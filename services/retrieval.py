"""
QAMed — Retrieval & Reranking Service

Handles model/client initialization, document retrieval, reranking,
and parent-child context expansion.
"""

from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
import pysbd

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    GROQ_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    DOC_STORE_PATH
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

    Returns: (groq_client, vectorstore, reranker, doc_store)
    """
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in environment variables.")

    groq_client = Groq(api_key=GROQ_API_KEY)
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    reranker = CrossEncoder("BAAI/bge-reranker-base")

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )
    
    doc_store = DocumentStore(DOC_STORE_PATH)

    return groq_client, vectorstore, reranker, doc_store


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
        
    result = " ".join(sentences)
    
    # Post-truncation token count verification
    while sentences and token_length(" ".join(sentences)) > max_tokens:
        if abs(start - leaf_offset) > abs(end - leaf_offset - leaf_length):
            sentences.pop(0)  # trim from start
        else:
            sentences.pop()   # trim from end
            
    return " ".join(sentences) if sentences else ""

def expand_to_parents(leaf_docs: list, doc_store: DocumentStore, max_tokens: int = 3000) -> list:
    """Expand retrieved leaf chunks to their parent sections.
    
    Returns list of parent section dicts with full text (truncated if necessary).
    """
    if not leaf_docs:
        return []
        
    expanded = []
    seen_parents = set()
    
    for doc in leaf_docs:
        parent_id = doc.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parents:
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
                
            expanded.append(parent_data)
            seen_parents.add(parent_id)
            
    return expanded
