# data/ — Ingestion and preprocessing pipeline
#
# Current modules:
#   ingest.py    — PDF loading (Docling), hierarchical chunking, multi-label tagging, Qdrant upload
#   doc_store.py — SQLite storage for parent sections (hierarchical retrieval context)
