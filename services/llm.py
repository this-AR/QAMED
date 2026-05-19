"""
QAMed — LLM Service Layer

All Groq LLM interactions: query decomposition, classification,
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
                    "You are a medical assistant. "
                    "Break the question into focused subquestions, one concept each. "
                    "Return only a numbered list. Maximum 4 subquestions. "
                    "Do not add explanations or preamble."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\nSubquestions:",
            },
        ],
        temperature=0,
        max_tokens=200,
    )

    output = response.choices[0].message.content
    raw = re.findall(r"\d+\.\s*(.+)", output)
    result = [q.strip() for q in raw if q.strip()]
    return result if result else [query]


# ── Query Classification ─────────────────────────────────────────────────────
def classify_query(groq_client, model: str, query: str) -> tuple[str, str]:
    """Classify the query as SIMPLE or COMPLEX using the Groq API.

    Returns a tuple: (label, raw_text)
    label: 'SIMPLE' or 'COMPLEX' (defaults to 'COMPLEX' on ambiguity)
    raw_text: raw model output
    """
    prompt = (
        "Classify this MBBS query as SIMPLE or COMPLEX.\n\n"
        "SIMPLE:\n"
        "- single fact\n"
        "- direct definition\n"
        "- localized retrieval\n\n"
        "COMPLEX:\n"
        "- multi-hop reasoning\n"
        "- comparison\n"
        "- requires synthesis\n\n"
        f"Query:\n{query}\n\n"
        "Answer with a single token: SIMPLE or COMPLEX. Do not add any other text."
    )

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an instruction-following classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=4,
        )
        out = response.choices[0].message.content.strip()
        label = "SIMPLE" if re.search(r"SIMPLE", out, re.IGNORECASE) else "COMPLEX"
        return label, out
    except Exception:
        return "COMPLEX", ""


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(subquery: str, docs: list, prompt_version: str) -> tuple[str, str, str]:
    """Build the system and user messages for the generation step.

    Returns: (system_msg, user_msg, resolved_version)
    """
    context_blocks = []
    for i, doc in enumerate(docs, 1):
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
        max_tokens=512,
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
