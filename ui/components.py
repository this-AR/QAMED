"""
QAMed — Streamlit UI Components

Reusable rendering functions for the Streamlit interface.
Each function handles a self-contained UI block.
"""

import re
import streamlit as st


def render_simple_answer(top_doc, query: str):
    """Render the short extracted answer for SIMPLE queries."""
    # Extract first 2 sentences as a concise definition
    sentences = re.split(r'(?<=[.!?])\s+', top_doc.page_content.strip())
    short_answer = " ".join(sentences[:2]).strip()

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
            st.markdown(
                f"- **{item['timestamp']}** | version={item['prompt_version']} | "
                f"model={item['model']} | sources={item['sources_used']}"
            )
