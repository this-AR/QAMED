"""
QAMed — Retrieval & Reranking Service

Handles model/client initialization, document retrieval, reranking,
and parent-child context expansion.
"""

import re

from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
import pysbd
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    GROQ_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    DOC_STORE_PATH,
    DENSE_TOP_K,
    BM25_TOP_K,
    RRF_K,
    MAX_PARENT_CONTEXT_TOKENS,
)
from data.doc_store import DocumentStore

SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)
try:
    TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
except Exception:
    TOKENIZER = AutoTokenizer.from_pretrained("gpt2")

def token_length(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))


# ── Model + Client Loading ───────────────────────────────────────────────────
def load_models_and_clients():
    """Initialize and cache all external clients and models.

    Returns: (groq_client, vectorstore, reranker, doc_store, bm25_index, bm25_corpus)
    """
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in environment variables.")

    # Free-tier TPM limits are hit within one multi-part question; the SDK honours
    # Groq's retry-after on 429s, so allow enough retries to wait out the window
    groq_client = Groq(api_key=GROQ_API_KEY, max_retries=5)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )
    
    doc_store = DocumentStore(DOC_STORE_PATH)

    # Build BM25 index at startup (always loaded for A/B testing)
    bm25_index, bm25_corpus = build_bm25_index(qdrant_client, COLLECTION_NAME)

    return groq_client, vectorstore, reranker, doc_store, bm25_index, bm25_corpus


# ── Hybrid Retrieval (BM25 + RRF) ────────────────────────────────────────────
def bm25_tokenize(text: str) -> list[str]:
    """Lowercase word tokens; punctuation is dropped so 'canal?' matches 'canal,'."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(qdrant_client: QdrantClient, collection_name: str):
    """Scroll Qdrant collection, tokenize text, build BM25 index."""
    points = []
    offset = None
    while True:
        page, offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=10000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(page)
        if offset is None:
            break

    corpus_docs = []
    tokenized_corpus = []

    for point in points:
        payload = point.payload or {}
        text = payload.get("page_content", "")
        metadata = payload.get("metadata", {})
        doc = Document(page_content=text, metadata=metadata)
        corpus_docs.append(doc)
        tokenized_corpus.append(bm25_tokenize(text))

    bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    return bm25_index, corpus_docs


def bm25_search(query: str, bm25_index: BM25Okapi, corpus_docs: list, top_k: int = 15) -> list:
    """Return top-K docs by BM25 score."""
    if not bm25_index or not corpus_docs:
        return []
    
    scores = bm25_index.get_scores(bm25_tokenize(query))
    
    ranked = sorted(zip(corpus_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k] if score > 0]


def rrf_fusion(dense_docs: list, sparse_docs: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion of dense and sparse results."""
    rrf_scores = {}

    for docs in (dense_docs, sparse_docs):
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_id]["score"] += 1.0 / (k + rank + 1)

    ranked_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in ranked_items]


# ── Full Retrieval Pipeline ──────────────────────────────────────────────────
def retrieve(subquery: str, vectorstore, reranker, doc_store: DocumentStore,
             bm25_index, bm25_corpus: list, hybrid: bool, top_n: int = 6) -> tuple[list, list]:
    """Dense (+ BM25 and RRF if hybrid) → rerank → expand to parent sections.

    Returns: (top_docs, parent_sections)
    """
    dense_docs = vectorstore.similarity_search("query: " + subquery, k=DENSE_TOP_K)
    if hybrid:
        sparse_docs = bm25_search(subquery, bm25_index, bm25_corpus, top_k=BM25_TOP_K)
        candidates = rrf_fusion(dense_docs, sparse_docs, k=RRF_K)
    else:
        candidates = dense_docs
    top_docs = rerank_docs(reranker, subquery, candidates, top_n=top_n)
    parent_sections = expand_to_parents(top_docs, doc_store, MAX_PARENT_CONTEXT_TOKENS)
    return top_docs, parent_sections


# ── Reranking Helper ─────────────────────────────────────────────────────────
def rerank_docs(reranker, query: str, docs: list, top_n: int | None = None) -> list:
    """Rerank documents using the CrossEncoder and return sorted list."""
    if not docs:
        return []

    try:
        scores = reranker.predict(
            [(query, doc.page_content) for doc in docs]
        )
        ranked = [
            doc for doc, _ in
            sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        ]
    except Exception:
        ranked = docs

    if top_n is not None:
        return ranked[:top_n]
    return ranked


# ── Parent Expansion & Truncation ────────────────────────────────────────────

def truncate_parent(parent_text: str, leaf_offset: int, leaf_length: int, max_tokens: int = 3000) -> str:
    """Positional window centered on the leaf's known offset within the parent.
    
    Args:
        parent_text:  Full parent section text.
        leaf_offset:  Character offset where this leaf starts within the parent.
        leaf_length:  Character length of the leaf chunk.
        max_tokens:   Token budget for the truncated output.
    """
    if not parent_text:
        return ""
        
    avg_chars_per_token = max(1.0, len(parent_text) / max(1, token_length(parent_text)))
    window_chars = int(max_tokens * avg_chars_per_token)
    
    if len(parent_text) <= window_chars:
        # Full text easily fits the budget character-wise, but double check tokens
        if token_length(parent_text) <= max_tokens:
            return parent_text
            
    half_window = window_chars // 2
    
    # Center window on the leaf
    start = max(0, leaf_offset - half_window)
    end = min(len(parent_text), leaf_offset + leaf_length + half_window)
    
    # Snap to sentence boundaries
    window_text = parent_text[start:end]
    sentences = SENTENCE_SEGMENTER.segment(window_text)
    if not sentences:
        return window_text

    # Post-truncation token count verification: trim whichever side has more
    # text left around the leaf, so the leaf stays centred and is cut last
    head = leaf_offset - start
    tail = end - (leaf_offset + leaf_length)
    while sentences and token_length(" ".join(sentences)) > max_tokens:
        if head >= tail:
            head -= len(sentences.pop(0))  # trim from start
        else:
            tail -= len(sentences.pop())   # trim from end
            
    return " ".join(sentences) if sentences else ""

def expand_to_parents(leaf_docs: list, doc_store: DocumentStore, max_tokens: int = 3000) -> list:
    """Expand retrieved leaf chunks to their parent sections.
    
    Returns list of parent section dicts with full text (truncated if necessary).
    """
    if not leaf_docs:
        return []
        
    expanded = []
    seen_parents = set()
    truncated_parents = set()

    for doc in leaf_docs:
        parent_id = doc.metadata.get("parent_id")
        if not parent_id:
            continue
        if parent_id in seen_parents:
            # The truncated window is centred on an earlier leaf and may not reach this one
            if parent_id in truncated_parents:
                expanded.append({"parent_id": parent_id, "text": doc.page_content})
            continue

        parent_data = doc_store.get_parent(parent_id)
        if parent_data:
            leaf_offset = doc.metadata.get("leaf_offset", 0)
            leaf_length = len(doc.page_content)
            
            if parent_data["token_count"] > max_tokens:
                parent_data["text"] = truncate_parent(
                    parent_data["text"], 
                    leaf_offset, 
                    leaf_length, 
                    max_tokens
                )
                truncated_parents.add(parent_id)

            expanded.append(parent_data)
            seen_parents.add(parent_id)
            
    return expanded
