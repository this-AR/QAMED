# services/ — External integrations and core business logic
#
# Current modules:
#   llm.py       — Groq LLM interactions (classify, decompose, stream, build_prompt)
#   retrieval.py — Vectorstore loading, retrieval, reranking
#
# Future modules (per roadmap):
#   cache.py         — Redis exact + semantic cache (v1.5)
#   memory.py        — Conversation memory (v1.0)
#   guardrails.py    — Output hallucination checks (v1.5)
#   observability.py — Langfuse tracing (v1.5)
#   hyde.py          — Hypothetical document embeddings (v2.0)
#   router.py        — Multi-book query routing (v2.0)
#   compression.py   — LLMLingua context compression (v2.0)
#   citations.py     — Citation builder (v1.0)
#   multihop.py      — LangGraph multi-hop agent (v3.0)
