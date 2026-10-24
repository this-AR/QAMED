# services/ — External integrations and core business logic
#
# Current modules:
#   llm.py           — Groq LLM interactions (decompose, stream, build_prompt)
#   retrieval.py     — Vectorstore loading, retrieve() pipeline, reranking
#   cache.py         — Redis exact + semantic cache (v1.5)
#   guardrails.py    — Output hallucination check (v1.5)
#   observability.py — Langfuse tracing (v1.5, opt-in)
