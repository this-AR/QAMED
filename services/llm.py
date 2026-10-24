"""
QAMed — LLM Service Layer

All Groq LLM interactions: query decomposition,
prompt building, and streaming generation.

Functions accept groq_client and model as explicit parameters so they
are decoupled from Streamlit and can be tested / reused independently.
"""

import re

from prompts import PROMPT_TEMPLATES, resolve_prompt_version


# ── Query Decomposition ──────────────────────────────────────────────────────
def extract_subquestions(groq_client, model: str, query: str) -> list[str]:
    """Break a complex query into focused subquestions via Groq."""
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert medical query decomposer. "
                    "Break complex or multi-part questions into individual, highly focused subquestions (one concept each). "
                    "This is critical for vector database retrieval. "
                    "If a question asks for a definition AND contents/causes/effects, YOU MUST SPLIT THEM into separate questions. "
                    "If the question is a simple definition or single-concept query, return it as a single question — do NOT split it. "
                    "Return only a numbered list. Maximum 3 subquestions. Do not add explanations.\n\n"
                    "Example 1:\n"
                    "Question: what is inguinal canal and its content?\n"
                    "1. What is the inguinal canal?\n"
                    "2. What are the contents of the inguinal canal?\n\n"
                    "Example 2:\n"
                    "Question: symptoms of malaria and how is it treated\n"
                    "1. What are the symptoms of malaria?\n"
                    "2. How is malaria treated?"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\nSubquestions:",
            },
        ],
        temperature=0,
        max_tokens=1024,  # headroom: reasoning models spend tokens before answering
    )

    output = response.choices[0].message.content
    raw = re.findall(r"\d+\.\s*(.+)", output)
    result = [q.strip() for q in raw if q.strip()]
    return result if result else [query]


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(subquery: str, parent_sections: list[dict], leaf_docs: list, prompt_version: str) -> tuple[str, str, str]:
    """Build the system and user messages for the generation step.

    Returns: (system_msg, user_msg, resolved_version)
    """
    context_blocks = []
    
    # We use parent sections for rich context
    if parent_sections:
        for i, parent in enumerate(parent_sections, 1):
            passage = parent["text"].strip().replace("\n", " ")
            context_blocks.append(f"[{i}] {passage}")
    else:
        # Fallback if no parents (e.g. legacy chunks or test data without parents)
        for i, doc in enumerate(leaf_docs, 1):
            passage = doc.page_content.strip().replace("\n", " ")
            context_blocks.append(f"[{i}] {passage}")

    version = resolve_prompt_version(prompt_version)
    template = PROMPT_TEMPLATES[version]

    system_msg = template["system"]
    user_msg = template["user"].format(
        sources=chr(10).join(context_blocks),
        question=subquery,
    )

    return system_msg, user_msg, version


# ── Streaming Generation ─────────────────────────────────────────────────────
def stream_groq_answer(groq_client, model: str, system_msg: str, user_msg: str):
    """Stream tokens from Groq for the given prompt. Yields token strings."""
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        top_p=0.9,
        max_tokens=2048,  # headroom: reasoning models spend tokens before answering
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
