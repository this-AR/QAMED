"""
QAMed — Medical RAG Assistant (Streamlit Entry Point)

This is the slim orchestrator. All business logic lives in services/,
all rendering helpers live in ui/, configuration in config.py.
"""

import time
from datetime import datetime

import streamlit as st

from config import GROQ_MODEL, DEFAULT_PROMPT_VERSION
from prompts import PROMPT_TEMPLATES, resolve_prompt_version
from services.llm import extract_subquestions, classify_query, build_prompt, stream_groq_answer
from services.retrieval import load_models_and_clients, rerank_docs
from ui.components import render_simple_answer, render_sources, render_rating, render_run_history


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="QAMed — Medical Assistant", layout="wide")
st.title("QAMed — Medical RAG Assistant")
st.caption("Streamed answers grounded in MBBS textbook chunks.")

# ── Session State Init ────────────────────────────────────────────────────────
if "ratings" not in st.session_state:
    st.session_state["ratings"] = {}
if "prompt_runs" not in st.session_state:
    st.session_state["prompt_runs"] = []

# ── Load Models on Startup ────────────────────────────────────────────────────
with st.spinner("Loading models and connections..."):
    try:
        groq_client, vectorstore, reranker = load_models_and_clients()
        st.session_state["groq_client"] = groq_client
        st.session_state["vectorstore"] = vectorstore
        st.session_state["reranker"] = reranker
    except Exception as exc:
        st.error(str(exc))
        st.stop()

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

    subqueries = extract_subquestions(groq_client, GROQ_MODEL, query) if use_decomposition else [query]

    # Always run Groq classifier to decide SIMPLE vs COMPLEX
    classification_label, classifier_raw = classify_query(groq_client, GROQ_MODEL, query)

    # Log the classification decision
    st.session_state["prompt_runs"].append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "query": query,
            "subquery": None,
            "prompt_version": None,
            "model": "groq-classifier",
            "sources_used": 0,
            "classification": classification_label,
            "classifier_output": classifier_raw,
            "skipped_rag": classification_label == "SIMPLE",
        }
    )

    # ── SIMPLE path: extract top doc definition, skip full RAG ────────────
    if classification_label == "SIMPLE":
        st.info("Detected SIMPLE question — returning extracted definition without full RAG generation.")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke("query: " + query)

        if not docs:
            st.warning("No matching documents found for the query.")
            st.stop()

        top_doc = rerank_docs(reranker, query, docs, top_n=1)[0]
        render_simple_answer(top_doc, query)
        st.stop()

    # ── COMPLEX path: full RAG pipeline ───────────────────────────────────
    if not subqueries:
        st.warning("No subquestions generated. Try rephrasing your query.")
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    for idx, subquery in enumerate(subqueries, start=1):
        st.subheader(f"Subquestion {idx}: {subquery}")

        docs = retriever.invoke("query: " + subquery)
        top_docs = rerank_docs(reranker, subquery, docs, top_n=4)

        system_msg, user_msg, used_prompt_version = build_prompt(
            subquery, top_docs, active_prompt_version,
        )

        # Stream the answer
        answer_box = st.empty()
        answer_text = ""
        for token in stream_groq_answer(groq_client, GROQ_MODEL, system_msg, user_msg):
            answer_text += token
            answer_box.markdown(answer_text)

        st.caption(f"Prompt version used: {used_prompt_version}")

        # Log the run
        st.session_state["prompt_runs"].append(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "query": query,
                "subquery": subquery,
                "prompt_version": used_prompt_version,
                "model": GROQ_MODEL,
                "sources_used": len(top_docs),
            }
        )

        render_rating(idx, subquery)
        render_sources(top_docs)

    st.caption(f"Completed in {time.time() - start_time:.2f}s")
    render_run_history(st.session_state["prompt_runs"])