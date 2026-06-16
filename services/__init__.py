# services/ — External integrations and core business logic
#
# Current modules:
#   llm.py           — Groq LLM interactions (classify, decompose, stream, build_prompt)
#   retrieval.py     — Vectorstore loading, retrieval, reranking
#   cache.py         — Redis exact + semantic cache (v1.5)
#   guardrails.py    — Output hallucination check (v1.5)
#   observability.py — Langfuse tracing (v1.5, opt-in)
#
# Future modules (per roadmap):
#   hyde.py          — Hypothetical document embeddings (v2.0)
#   router.py        — Multi-book query routing (v2.0)
#   compression.py   — LLMLingua context compression (v2.0)
#   memory.py        — Persistent conversation memory (v2.0)
#   multihop.py      — LangGraph multi-hop agent (v3.0)
