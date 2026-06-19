"""
QAMed — Retrieval & Reranking Service

Handles model/client initialization, document retrieval, and reranking.
Pure Python — no UI framework dependencies. Caching is the caller's responsibility
(e.g. app.py uses @st.cache_resource when running under Streamlit).
"""

from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    GROQ_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


# ── Model + Client Loading ───────────────────────────────────────────────────
def load_models_and_clients():
    """Initialize and cache all external clients and models.

    Returns: (groq_client, vectorstore, reranker)
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

    return groq_client, vectorstore, reranker


# ── Reranking Helper ─────────────────────────────────────────────────────────
def rerank_docs(reranker, query: str, docs: list, top_n: int | None = None) -> list:
    """Rerank documents using the CrossEncoder and return sorted list.

    Args:
        reranker: CrossEncoder model instance
        query: the search query
        docs: list of retrieved documents
        top_n: number of top docs to return (None = return all, sorted)

    Returns:
        List of documents sorted by reranker score (descending), sliced to top_n.
    """
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
