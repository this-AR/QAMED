"""
QAMed — Centralized Configuration

All environment variables, constants, and shared settings live here.
Both the Streamlit app and the ingestion pipeline import from this module.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_documents")

# ── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Embeddings ────────────────────────────────────────────────────────────────
# Shared between app (query encoding) and ingest (document encoding).
# Change this once when migrating to MedCPT — takes effect everywhere.
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# ── Prompts ───────────────────────────────────────────────────────────────────
DEFAULT_PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3")

# ── Cache (v1.5) ──────────────────────────────────────────────────────────────
# REDIS_URL is optional. If absent, exact-match cache is silently disabled.
# Example: redis://default:<password>@<host>:<port>
REDIS_URL = os.getenv("REDIS_URL")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 hours
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_COLLECTION = os.getenv("SEMANTIC_CACHE_COLLECTION", "query_cache")

# ── Langfuse Tracing (v1.5) ───────────────────────────────────────────────────
# All three are optional. If absent, tracing is silently disabled.
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")  # defaults to Langfuse Cloud if unset

# ── Hierarchical Chunking (v2.0) & Hybrid Search ──────────────────────────────
PARAGRAPH_CHUNK_TOKENS = int(os.getenv("PARAGRAPH_CHUNK_TOKENS", "200"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "30"))
MAX_PARENT_CONTEXT_TOKENS = int(os.getenv("MAX_PARENT_CONTEXT_TOKENS", "2048"))
DOC_STORE_PATH = os.getenv("DOC_STORE_PATH", "data/doc_store.db")
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "75"))
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "30"))

# ── Contextual Micro-Headers & A/B Testing (v2.5) ────────────────────────────
CONTEXT_API_KEYS = os.getenv("CONTEXT_API_KEYS", "")
CONTEXT_WORKERS = int(os.getenv("CONTEXT_WORKERS", "5"))
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"

