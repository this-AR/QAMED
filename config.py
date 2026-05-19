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
DEFAULT_PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
