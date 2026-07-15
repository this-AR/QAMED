"""
QAMed — Medical RAG Assistant (Streamlit Entry Point) v1.5

Orchestration layer — all business logic lives in services/, all rendering
helpers live in ui/, configuration in config.py.

v1.5 additions wired here:
  - Two-level cache (exact + semantic) before any pipeline work
  - Output guardrail (hallucination check) after generation
  - Langfuse tracing around all LLM calls
  - Async RAGAS evaluation after each subquery answer
"""

import time
from datetime import datetime, timezone

import streamlit as st

from config import GROQ_MODEL, DEFAULT_PROMPT_VERSION, MAX_PARENT_CONTEXT_TOKENS
from prompts import PROMPT_TEMPLATES, resolve_prompt_version

from services.llm import extract_subquestions, classify_query, build_prompt, stream_groq_answer
from services.retrieval import load_models_and_clients, rerank_docs, expand_to_parents
from services.cache import check_cache, store_in_cache, CacheResult
from services.guardrails import check_hallucination
from services.observability import get_tracer, Timer

from evaluation.ragas_eval import run_eval_async

from ui.components import (
    render_simple_answer,
    render_sources,
    render_rating,
    render_run_history,
    render_cache_badge,
    render_guardrail_badge,
    render_ragas_scores,
)


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="QAMed — Medical Assistant", layout="wide")
st.title("QAMed — Medical RAG Assistant")
st.caption("Streamed answers grounded in MBBS textbook chunks.")

# ── Session State Init ────────────────────────────────────────────────────────
if "ratings" not in st.session_state:
    st.session_state["ratings"] = {}
if "prompt_runs" not in st.session_state:
    st.session_state["prompt_runs"] = []
if "ragas_results" not in st.session_state:
    st.session_state["ragas_results"] = {}

# ── Cached wrapper for model loading (keeps heavy models alive across Streamlit reruns) ──
@st.cache_resource(show_spinner=False)
def _cached_load_models():
    return load_models_and_clients()


# ── Load Models on Startup ────────────────────────────────────────────────────
with st.spinner("Loading models and connections..."):
    try:
        groq_client, vectorstore, reranker, doc_store = _cached_load_models()
        st.session_state["groq_client"] = groq_client
        st.session_state["vectorstore"] = vectorstore
        st.session_state["reranker"] = reranker
        st.session_state["doc_store"] = doc_store
    except Exception as exc:
        st.error(str(exc))
        st.stop()

# ── Tracer (opt-in Langfuse, no-op if unconfigured) ──────────────────────────
tracer = get_tracer()

# ── Embedding function (used by semantic cache) ───────────────────────────────
# Reuse the embedding model that's already loaded inside the vectorstore.
# QdrantVectorStore keeps a .embeddings attribute that wraps the HF model.
def _embed_query(text: str) -> list[float]:
    """Embed a single string using the cached HF embedding model."""
    return vectorstore.embeddings.embed_query(text)

# ── Qdrant raw client (used by semantic cache) ────────────────────────────────
# QdrantVectorStore wraps a QdrantClient — access it for cache operations.
_qdrant_client = vectorstore.client


# ── UI Controls ───────────────────────────────────────────────────────────────
query = st.text_input(
    "Ask a medical question",
    placeholder="e.g. What is the inguinal canal and what are its contents?",
)
use_decomposition = st.checkbox("Use query decomposition", value=True)

prompt_options = list(PROMPT_TEMPLATES.keys())
default_version = resolve_prompt_version(DEFAULT_PROMPT_VERSION)
default_index = prompt_options.index(default_version) if default_version in prompt_options else 0
active_prompt_version = st.selectbox(
    "Prompt version",
    options=prompt_options,
    index=default_index,
    format_func=lambda version: f"{version} - {PROMPT_TEMPLATES[version]['name']}",
)
st.caption(f"Active prompt version: {active_prompt_version}")


# ── Main Query Handler ────────────────────────────────────────────────────────
if st.button("Ask", type="primary") and query.strip():
    start_time = time.time()
    st.write("---")

    with tracer.trace("query-pipeline", query=query) as trace:

        # ── Level 1 & 2: Cache check ──────────────────────────────────────
        cache_result: CacheResult | None = check_cache(query, _embed_query, _qdrant_client)

        if cache_result is not None:
            render_cache_badge(cache_result.hit_type, cache_result.similarity)
            st.markdown(cache_result.answer)
            if cache_result.sources_text:
                with st.expander("Sources (cached)"):
                    st.markdown(cache_result.sources_text)
            st.session_state["prompt_runs"].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    "query": query,
                    "subquery": None,
                    "prompt_version": cache_result.prompt_version,
                    "model": "cache",
                    "sources_used": 0,
                    "cache_hit": True,
                }
            )
            trace_id = getattr(trace, "id", None) if trace else None
            render_rating(0, query, trace_id)
            
            st.caption(f"Returned from cache in {time.time() - start_time:.2f}s")
            render_run_history(st.session_state["prompt_runs"])
            st.stop()

        # ── Query decomposition + classification ──────────────────────────
        t_decomp = Timer()
        subqueries = (
            extract_subquestions(groq_client, GROQ_MODEL, query)
            if use_decomposition
            else [query]
        )
        tracer.log_generation(
            trace,
            name="decomposition",
            prompt=query,
            completion=str(subqueries),
            model=GROQ_MODEL,
            latency_ms=t_decomp.elapsed_ms(),
        )

        t_classify = Timer()
        classification_label, classifier_raw = classify_query(groq_client, GROQ_MODEL, query)
        tracer.log_generation(
            trace,
            name="classification",
            prompt=query,
            completion=classification_label,
            model=GROQ_MODEL,
            latency_ms=t_classify.elapsed_ms(),
            metadata={"label": classification_label},
        )

        # Log classification decision
        st.session_state["prompt_runs"].append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "query": query,
                "subquery": None,
                "prompt_version": None,
                "model": "groq-classifier",
                "sources_used": 0,
                "classification": classification_label,
                "classifier_output": classifier_raw,
                "skipped_rag": classification_label == "SIMPLE",
                "cache_hit": False,
            }
        )

        # ── SIMPLE path: extracted definition, skip full RAG ──────────────
        if classification_label == "SIMPLE":
            st.info("Detected SIMPLE question — returning extracted definition without full RAG generation.")

            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke("query: " + query)

            if not docs:
                st.warning("No matching documents found for the query.")
                st.stop()

            top_doc = rerank_docs(reranker, query, docs, top_n=1)[0]
            render_simple_answer(top_doc, query, st.session_state["doc_store"])
            
            trace_id = getattr(trace, "id", None) if trace else None
            render_rating(0, query, trace_id)
            
            st.caption(f"Completed in {time.time() - start_time:.2f}s")
            st.stop()

        # ── COMPLEX path: full RAG pipeline ──────────────────────────────
        if not subqueries:
            st.warning("No subquestions generated. Try rephrasing your query.")
            st.stop()

        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

        # Accumulate full answer for cache storage after all subqueries
        full_answer_parts: list[str] = []
        full_sources_text_parts: list[str] = []

        for idx, subquery in enumerate(subqueries, start=1):
            st.subheader(f"Subquestion {idx}: {subquery}")

            docs = retriever.invoke("query: " + subquery)
            top_docs = rerank_docs(reranker, subquery, docs, top_n=4)
            context_chunks = [doc.page_content for doc in top_docs]

            parent_sections = expand_to_parents(top_docs, st.session_state["doc_store"], MAX_PARENT_CONTEXT_TOKENS)

            system_msg, user_msg, used_prompt_version = build_prompt(
                subquery, parent_sections, top_docs, active_prompt_version,
            )

            # ── Stream the answer ─────────────────────────────────────────
            t_gen = Timer()
            answer_box = st.empty()
            answer_text = ""
            for token in stream_groq_answer(groq_client, GROQ_MODEL, system_msg, user_msg):
                answer_text += token
                answer_box.markdown(answer_text)

            tracer.log_generation(
                trace,
                name=f"generation-{idx}",
                prompt=user_msg,
                completion=answer_text,
                model=GROQ_MODEL,
                latency_ms=t_gen.elapsed_ms(),
                metadata={"prompt_version": used_prompt_version, "subquery": subquery},
            )

            # ── Output guardrail ──────────────────────────────────────────
            t_guard = Timer()
            guardrail = check_hallucination(answer_text, context_chunks, groq_client, GROQ_MODEL)
            tracer.log_generation(
                trace,
                name=f"guardrail-{idx}",
                prompt=user_msg[:500],
                completion=guardrail.raw_output,
                model=GROQ_MODEL,
                latency_ms=t_guard.elapsed_ms(),
                metadata={"label": guardrail.label},
            )

            # Retry once with stricter prompt if hallucinated
            if not guardrail.is_grounded:
                st.warning("⚠️ Guardrail triggered — regenerating with stricter prompt…")
                stricter_system = (
                    system_msg
                    + " IMPORTANT: Only state facts that appear word-for-word in the numbered sources. "
                    "If the information is not explicitly in the sources, say so. "
                    "Do not infer or extrapolate under any circumstances."
                )
                t_regen = Timer()
                retry_box = st.empty()
                retry_text = ""
                for token in stream_groq_answer(groq_client, GROQ_MODEL, stricter_system, user_msg):
                    retry_text += token
                    retry_box.markdown(retry_text)
                answer_text = retry_text

                # Re-check after retry
                guardrail = check_hallucination(answer_text, context_chunks, groq_client, GROQ_MODEL)
                tracer.log_generation(
                    trace,
                    name=f"guardrail-retry-{idx}",
                    prompt=user_msg[:500],
                    completion=guardrail.raw_output,
                    model=GROQ_MODEL,
                    latency_ms=t_regen.elapsed_ms(),
                    metadata={"label": guardrail.label, "retry": True},
                )

            render_guardrail_badge(guardrail.is_grounded, guardrail.explanation)
            st.caption(f"Prompt version used: {used_prompt_version}")

            # ── Log run ───────────────────────────────────────────────────
            st.session_state["prompt_runs"].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    "query": query,
                    "subquery": subquery,
                    "prompt_version": used_prompt_version,
                    "model": GROQ_MODEL,
                    "sources_used": len(top_docs),
                    "cache_hit": False,
                    "guardrail": guardrail.label,
                }
            )

            # ── Async RAGAS eval ──────────────────────────────────────────
            ragas_key = f"ragas_{idx}_{hash(subquery)}"
            run_eval_async(
                query=subquery,
                answer=answer_text,
                contexts=context_chunks,
                session_state=st.session_state,
                session_key=ragas_key,
                tracer=tracer,
            )

            # trace is from `with tracer.trace(...) as trace:`
            trace_id = getattr(trace, "id", None) if trace else None
            render_rating(idx, subquery, trace_id)
            render_sources(top_docs)
            render_ragas_scores(st.session_state.get(ragas_key))

            full_answer_parts.append(answer_text)
            full_sources_text_parts.append(
                "\n".join(
                    f"[{i+1}] {doc.page_content[:200]}…"
                    for i, doc in enumerate(top_docs)
                )
            )

        # ── Store in cache after full pipeline ────────────────────────────
        combined_answer = "\n\n---\n\n".join(full_answer_parts)
        combined_sources = "\n\n".join(full_sources_text_parts)
        store_in_cache(
            query=query,
            answer=combined_answer,
            sources_text=combined_sources,
            prompt_version=active_prompt_version,
            embedding_fn=_embed_query,
            qdrant_client=_qdrant_client,
        )

    st.caption(f"Completed in {time.time() - start_time:.2f}s")
    render_run_history(st.session_state["prompt_runs"])