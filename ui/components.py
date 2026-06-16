"""
QAMed — Streamlit UI Components

Reusable rendering functions for the Streamlit interface.
Each function handles a self-contained UI block.
"""

import streamlit as st
import pysbd

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def render_simple_answer(top_doc, query: str):
    """Render the short extracted answer for SIMPLE queries."""
    # PySBD-aware sentence splitting — handles Dr., Fig., M. tuberculosis etc.
    sentences = _SEGMENTER.segment(top_doc.page_content.strip())
    short_answer = " ".join(s.strip() for s in sentences[:2] if s.strip())

    st.subheader("Extracted answer (SIMPLE)")
    st.markdown(short_answer)

    meta = top_doc.metadata or {}
    st.markdown(
        f"**Source:** Chapter: {meta.get('chapter', 'N/A')} | "
        f"Page: {meta.get('page_number', 'N/A')} | "
        f"Book: {meta.get('book_name', 'N/A')}"
    )


def render_sources(top_docs: list):
    """Render the 'View sources' expander with document metadata."""
    with st.expander("View sources"):
        for doc in top_docs:
            meta = doc.metadata or {}
            st.markdown(
                f"**Chapter:** {meta.get('chapter', 'N/A')} &nbsp;|&nbsp; "
                f"**Page:** {meta.get('page_number', 'N/A')} &nbsp;|&nbsp; "
                f"**Book:** {meta.get('book_name', 'N/A')} &nbsp;|&nbsp; "
                f"**Labels:** {', '.join(meta.get('labels', [])) if meta.get('labels') else 'N/A'}"
            )
            st.write(doc.page_content)
            st.write("---")


def render_rating(idx: int, subquery: str):
    """Render the answer rating slider and submit button."""
    rating_key = f"rating_{idx}_{hash(subquery)}"
    col1, col2 = st.columns([3, 1])
    with col1:
        rating = st.slider("Rate this answer", 1, 10, 7, key=rating_key)
    with col2:
        if st.button("Submit", key=f"submit_{rating_key}"):
            st.session_state["ratings"][subquery] = rating
            st.success(f"Saved {rating}/10")


def render_run_history(prompt_runs: list):
    """Render the 'Prompt run history' expander."""
    with st.expander("Prompt run history"):
        for item in reversed(prompt_runs[-20:]):
            classification = item.get("classification", "")
            skipped = item.get("skipped_rag", False)
            cache_hit = item.get("cache_hit", False)
            badge = ""
            if cache_hit:
                badge = "⚡ cached"
            elif skipped:
                badge = "→ SIMPLE"
            st.markdown(
                f"- **{item['timestamp']}** | version={item.get('prompt_version')} | "
                f"model={item['model']} | sources={item.get('sources_used', 0)} "
                f"{'| ' + badge if badge else ''}"
            )


def render_cache_badge(hit_type: str, similarity: float | None = None):
    """Show a green badge when an answer is returned from cache."""
    if hit_type == "exact":
        st.success("⚡ **Exact cache hit** — answer returned instantly from cache.")
    elif hit_type == "semantic":
        sim_str = f" (similarity: {similarity:.3f})" if similarity else ""
        st.success(f"⚡ **Semantic cache hit**{sim_str} — answer retrieved from similar past query.")


def render_guardrail_badge(is_grounded: bool, explanation: str):
    """Show grounded / hallucination warning badge below an answer."""
    if is_grounded:
        st.caption("✅ Guardrail: answer is grounded in retrieved sources.")
    else:
        st.warning(
            f"⚠️ **Guardrail warning**: potential hallucination detected.\n\n"
            f"_{explanation}_\n\n"
            "Answer shown with caution — verify against sources."
        )


def render_ragas_scores(scores):
    """
    Render RAGAS scores in a collapsed expander.
    `scores` is a RagasScores dataclass or None (pending).
    """
    with st.expander("📊 Evaluation scores (RAGAS)"):
        if scores is None:
            st.info("Evaluation running in background… refresh in a moment.")
        elif scores.error:
            st.warning(f"RAGAS evaluation failed: {scores.error}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                _score_metric("Faithfulness", scores.faithfulness)
            with col2:
                _score_metric("Answer Relevancy", scores.answer_relevancy)
            with col3:
                _score_metric("Context Precision", scores.context_precision)


def _score_metric(label: str, value: float | None):
    if value is None:
        st.metric(label, "N/A")
    else:
        color = "🟢" if value >= 0.7 else ("🟡" if value >= 0.4 else "🔴")
        st.metric(label, f"{color} {value:.2f}")
