# data/ — Ingestion and preprocessing pipeline
#
# Current modules:
#   ingest.py — PDF loading, chunking, metadata tagging, Qdrant upload
#
# Future modules (per roadmap):
#   parsers.py  — Docling parser + PyMuPDF fallback (v2.0)
#   chunking.py — Multi-level chunking: sentence/paragraph/section (v2.0)
#   tagging.py  — LLM-based semantic metadata tagging (v2.0)
#   toc.py      — Automatic TOC chapter mapping (v2.0)
