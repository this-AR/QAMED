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

from config import GROQ_MODEL, DEFAULT_PROMPT_VERSION, MAX_PARENT_CONTEXT_TOKENS, BM25_TOP_K, DENSE_TOP_K, RRF_K, USE_HYBRID_SEARCH
from prompts import PROMPT_TEMPLATES, resolve_prompt_version

from services.llm import extract_subquestions, classify_query, build_prompt, stream_groq_answer
from services.retrieval import load_models_and_clients, rerank_docs, expand_to_parents, bm25_search, rrf_fusion
from services.cache import check_cache, store_in_cache, CacheResult
from services.guardrails import check_hallucination
from services.observability import get_tracer, Timer

from evaluation.ragas_eval import run_eval_async

from ui.components import (
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
if "active_response" not in st.session_state:
    st.session_state["active_response"] = None

# ── Cached wrapper for model loading (keeps heavy models alive across Streamlit reruns) ──
@st.cache_resource(show_spinner=False)
def _cached_load_models():
    return load_models_and_clients()


# ── Load Models on Startup ────────────────────────────────────────────────────
with st.spinner("Loading models and connections..."):
    try:
        groq_client, vectorstore, reranker, doc_store, bm25_index, bm25_corpus = _cached_load_models()
        st.session_state["groq_client"] = groq_client
        st.session_state["vectorstore"] = vectorstore
        st.session_state["reranker"] = reranker
        st.session_state["doc_store"] = doc_store
        st.session_state["bm25_index"] = bm25_index
        st.session_state["bm25_corpus"] = bm25_corpus
    except Exception as exc:
        st.error(str(exc))
        st.stop()

# ── Tracer (opt-in Langfuse, no-op if unconfigured) ──────────────────────────
tracer = get_tracer()

# ── Embedding function (used by semantic cache) ───────────────────────────────
def _embed_query(text: str) -> list[float]:
    """Embed a single string using the cached HF embedding model."""
    return vectorstore.embeddings.embed_query(text)

# ── Qdrant raw client (used by semantic cache) ────────────────────────────────
_qdrant_client = vectorstore.client


# ── Response Renderer Helper (Reruns without re-executing LLM) ────────────────
def render_active_response(data: dict):
    if not data:
        return
    st.write("---")
    if data.get("is_cache"):
        c = data["cache_result"]
        render_cache_badge(c.hit_type, c.similarity)
        st.markdown(c.answer)
        if c.sources_text:
            with st.expander("Sources (cached)"):
                st.markdown(c.sources_text)
        render_rating(0, data["query"], data.get("trace_id"))
        st.caption(f"Returned from cache in {data['elapsed']:.2f}s")
    else:
        for item in data.get("subquery_data", []):
            st.subheader(f"Subquestion {item['idx']}: {item['subquery']}")
            if item.get("ab_testing"):
                col_hybrid, col_dense = st.columns(2)
                with col_hybrid:
                    st.markdown("#### 🔵 Hybrid (BM25 + Dense + RRF)")
                    st.markdown(item["hybrid_text"])
                    render_sources(item["hybrid_top"])
                with col_dense:
                    st.markdown("#### 🟢 Pure Dense")
                    st.markdown(item["dense_text"])
                    render_sources(item["dense_top"])
            else:
                st.markdown(item["answer_text"])
                if item.get("guardrail"):
                    g = item["guardrail"]
                    render_guardrail_badge(g.is_grounded, g.explanation)
                st.caption(f"Prompt version used: {item['used_prompt_version']}")
                render_rating(item["idx"], item["subquery"], item.get("trace_id"))
                render_sources(item["top_docs"])
                render_ragas_scores(st.session_state.get(item["ragas_key"]))

        st.caption(f"Completed in {data['elapsed']:.2f}s")

    render_run_history(st.session_state["prompt_runs"])


# ── UI Controls ───────────────────────────────────────────────────────────────
query = st.text_input(
    "Ask a medical question",
    placeholder="e.g. What is the inguinal canal and what are its contents?",
)
use_decomposition = st.checkbox("Use query decomposition", value=True)
ab_testing = st.checkbox("🔬 A/B Test: Compare Hybrid vs Pure Dense", value=False)

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
        trace_id = getattr(trace, "id", None) if trace else None

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
            render_rating(0, query, trace_id)
            
            elapsed = time.time() - start_time
            st.caption(f"Returned from cache in {elapsed:.2f}s")
            render_run_history(st.session_state["prompt_runs"])

            st.session_state["active_response"] = {
                "is_cache": True,
                "query": query,
                "cache_result": cache_result,
                "elapsed": elapsed,
                "trace_id": trace_id,
            }
        else:
            # ── Query classification + decomposition ──────────────────────────
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

            t_decomp = Timer()
            if classification_label == "SIMPLE":
                subqueries = [query]
            elif use_decomposition:
                subqueries = extract_subquestions(groq_client, GROQ_MODEL, query)
            else:
                subqueries = [query]
                
            tracer.log_generation(
                trace,
                name="decomposition",
                prompt=query,
                completion=str(subqueries),
                model=GROQ_MODEL,
                latency_ms=t_decomp.elapsed_ms(),
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

            if not subqueries:
                st.warning("No subquestions generated. Try rephrasing your query.")
            else:
                retriever = vectorstore.as_retriever(search_kwargs={"k": DENSE_TOP_K})
                full_answer_parts: list[str] = []
                full_sources_text_parts: list[str] = []
                subquery_data = []

                for idx, subquery in enumerate(subqueries, start=1):
                    st.subheader(f"Subquestion {idx}: {subquery}")

                    # ── A/B Testing Mode ──────────────────────────────────────────
                    if ab_testing:
                        col_hybrid, col_dense = st.columns(2)
                        dense_docs = retriever.invoke("query: " + subquery)

                        with col_hybrid:
                            st.markdown("#### 🔵 Hybrid (BM25 + Dense + RRF)")
                            sparse_docs = bm25_search(subquery, st.session_state["bm25_index"], st.session_state["bm25_corpus"], top_k=BM25_TOP_K)
                            fused_docs = rrf_fusion(dense_docs, sparse_docs, k=RRF_K)
                            hybrid_top = rerank_docs(reranker, subquery, fused_docs, top_n=6)
                            hybrid_parents = expand_to_parents(hybrid_top, st.session_state["doc_store"], MAX_PARENT_CONTEXT_TOKENS)
                            h_sys, h_usr, h_ver = build_prompt(subquery, hybrid_parents, hybrid_top, active_prompt_version)
                            h_text = st.write_stream(stream_groq_answer(groq_client, GROQ_MODEL, h_sys, h_usr))
                            render_sources(hybrid_top)

                        with col_dense:
                            st.markdown("#### 🟢 Pure Dense")
                            dense_top = rerank_docs(reranker, subquery, dense_docs, top_n=6)
                            dense_parents = expand_to_parents(dense_top, st.session_state["doc_store"], MAX_PARENT_CONTEXT_TOKENS)
                            d_sys, d_usr, d_ver = build_prompt(subquery, dense_parents, dense_top, active_prompt_version)
                            d_text = st.write_stream(stream_groq_answer(groq_client, GROQ_MODEL, d_sys, d_usr))
                            render_sources(dense_top)

                        top_docs = hybrid_top
                        parent_sections = hybrid_parents
                        context_chunks = [p["text"] for p in parent_sections] if parent_sections else [doc.page_content for doc in top_docs]
                        answer_text = h_text
                        used_prompt_version = h_ver

                        st.caption(f"Prompt version used: {used_prompt_version}")
                        st.session_state["prompt_runs"].append({
                            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                            "query": query, "subquery": subquery,
                            "prompt_version": used_prompt_version, "model": GROQ_MODEL,
                            "sources_used": len(top_docs), "cache_hit": False,
                            "mode": "ab_test",
                        })
                        render_rating(idx, subquery, trace_id)
                        full_answer_parts.append(answer_text)
                        full_sources_text_parts.append(
                            "\n".join(f"[{i+1}] {doc.page_content[:200]}…" for i, doc in enumerate(top_docs))
                        )

                        subquery_data.append({
                            "idx": idx,
                            "subquery": subquery,
                            "ab_testing": True,
                            "hybrid_text": h_text,
                            "hybrid_top": hybrid_top,
                            "dense_text": d_text,
                            "dense_top": dense_top,
                            "top_docs": top_docs,
                            "answer_text": h_text,
                            "used_prompt_version": h_ver,
                            "guardrail": None,
                            "ragas_key": None,
                            "trace_id": trace_id,
                        })
                        continue

                    # ── Normal single-path retrieval ──────────────────────────────
                    dense_docs = retriever.invoke("query: " + subquery)
                    if USE_HYBRID_SEARCH:
                        sparse_docs = bm25_search(subquery, st.session_state["bm25_index"], st.session_state["bm25_corpus"], top_k=BM25_TOP_K)
                        fused_docs = rrf_fusion(dense_docs, sparse_docs, k=RRF_K)
                    else:
                        fused_docs = dense_docs
                    top_docs = rerank_docs(reranker, subquery, fused_docs, top_n=6)
                    parent_sections = expand_to_parents(top_docs, st.session_state["doc_store"], MAX_PARENT_CONTEXT_TOKENS)
                    context_chunks = [p["text"] for p in parent_sections] if parent_sections else [doc.page_content for doc in top_docs]

                    system_msg, user_msg, used_prompt_version = build_prompt(
                        subquery, parent_sections, top_docs, active_prompt_version,
                    )

                    t_gen = Timer()
                    answer_text = st.write_stream(stream_groq_answer(groq_client, GROQ_MODEL, system_msg, user_msg))

                    tracer.log_generation(
                        trace,
                        name=f"generation-{idx}",
                        prompt=user_msg,
                        completion=answer_text,
                        model=GROQ_MODEL,
                        latency_ms=t_gen.elapsed_ms(),
                        metadata={"prompt_version": used_prompt_version, "subquery": subquery},
                    )

                    t_guard = Timer()
                    guardrail = check_hallucination(answer_text, context_chunks, groq_client, GROQ_MODEL, embedding_fn=_embed_query)
                    tracer.log_generation(
                        trace,
                        name=f"guardrail-{idx}",
                        prompt=user_msg[:500],
                        completion=guardrail.raw_output,
                        model=GROQ_MODEL,
                        latency_ms=t_guard.elapsed_ms(),
                        metadata={"label": guardrail.label},
                    )

                    if not guardrail.is_grounded:
                        st.warning("⚠️ Guardrail triggered — regenerating with stricter prompt…")
                        stricter_system = (
                            system_msg
                            + " IMPORTANT: Only state facts that appear word-for-word in the numbered sources. "
                            "If the information is not explicitly in the sources, say so. "
                            "Do not infer or extrapolate under any circumstances."
                        )
                        t_regen = Timer()
                        answer_text = st.write_stream(stream_groq_answer(groq_client, GROQ_MODEL, stricter_system, user_msg))

                        guardrail = check_hallucination(answer_text, context_chunks, groq_client, GROQ_MODEL, embedding_fn=_embed_query)
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

                    ragas_key = f"ragas_{idx}_{hash(subquery)}"
                    run_eval_async(
                        query=subquery,
                        answer=answer_text,
                        contexts=context_chunks,
                        session_state=st.session_state,
                        session_key=ragas_key,
                        tracer=tracer,
                    )

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

                    subquery_data.append({
                        "idx": idx,
                        "subquery": subquery,
                        "ab_testing": False,
                        "top_docs": top_docs,
                        "answer_text": answer_text,
                        "used_prompt_version": used_prompt_version,
                        "guardrail": guardrail,
                        "ragas_key": ragas_key,
                        "trace_id": trace_id,
                    })

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

                elapsed = time.time() - start_time
                st.caption(f"Completed in {elapsed:.2f}s")
                render_run_history(st.session_state["prompt_runs"])

                st.session_state["active_response"] = {
                    "is_cache": False,
                    "query": query,
                    "subquery_data": subquery_data,
                    "elapsed": elapsed,
                }

elif st.session_state.get("active_response"):
    render_active_response(st.session_state["active_response"])