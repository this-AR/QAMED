"""
QAMed — Streamlit UI Components

Reusable rendering functions for the Streamlit interface.
Each function handles a self-contained UI block.
"""

import streamlit as st
import pysbd

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def render_simple_answer(top_doc, query: str, doc_store=None):
    """Render the short extracted answer for SIMPLE queries."""
    # Start with the chunk content
    text_to_segment = top_doc.page_content.strip()
    
    # If we have access to the SQLite doc store, fetch the original un-chopped text
    # starting exactly at the chunk's offset to guarantee full sentences!
    if doc_store:
        parent_id = top_doc.metadata.get("parent_id")
        if parent_id:
            parent_data = doc_store.get_parent(parent_id)
            if parent_data:
                offset = top_doc.metadata.get("leaf_offset", 0)
                # Take a generous slice of the parent text starting from the exact chunk offset
                text_to_segment = parent_data["text"][offset:offset+1000].strip()

    # PySBD-aware sentence splitting — handles Dr., Fig., M. tuberculosis etc.
    sentences = _SEGMENTER.segment(text_to_segment)
    # Increase from 2 to 5 sentences to provide a more detailed definition
    short_answer = " ".join(s.strip() for s in sentences[:5] if s.strip())

    st.subheader("Extracted answer (SIMPLE)")
    st.markdown(short_answer)

    meta = top_doc.metadata or {}
    st.markdown(
        f"**Source:** Chapter: {meta.get('chapter', 'N/A')} | "
        f"Page: {meta.get('book_page', 'N/A')} | "
        f"Book: {meta.get('book_name', 'N/A')}"
    )


def render_sources(top_docs: list):
    """Render the 'View sources' expander with document metadata."""
    with st.expander("View sources"):
        for doc in top_docs:
            meta = doc.metadata or {}
            st.markdown(
                f"**Chapter:** {meta.get('chapter', 'N/A')} &nbsp;|&nbsp; "
                f"**Page:** {meta.get('book_page', 'N/A')} &nbsp;|&nbsp; "
                f"**Book:** {meta.get('book_name', 'N/A')} &nbsp;|&nbsp; "
                f"**Labels:** {', '.join(meta.get('labels', [])) if meta.get('labels') else 'N/A'}"
            )
            st.write(doc.page_content)
            st.write("---")


from services.observability import get_tracer

@st.fragment
def render_rating(idx: int, subquery: str, trace_id: str | None = None):
    """Render the answer rating slider and submit button as an independent fragment."""
    rating_key = f"rating_{idx}_{hash(subquery)}"
    col1, col2 = st.columns([3, 1])
    with col1:
        rating = st.slider("Rate this answer", 1, 10, 7, key=rating_key)
    with col2:
        if st.button("Submit", key=f"submit_{rating_key}"):
            st.session_state["ratings"][subquery] = rating
            
            # Log score to Langfuse if tracing is enabled
            if trace_id:
                tracer = get_tracer()
                if tracer.enabled and hasattr(tracer._client, "score"):
                    try:
                        tracer._client.score(
                            trace_id=trace_id,
                            name="user-rating",
                            value=rating,
                            comment=f"Rating for subquery: {subquery}"
                        )
                    except Exception as e:
                        pass # Silently ignore network errors for metrics
            
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
