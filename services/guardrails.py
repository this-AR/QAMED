"""
QAMed — Output Guardrails (v1.5)

Detects hallucinations before an answer reaches the user.

Strategy: Lightweight Groq call that compares the generated answer against
the retrieved context chunks. Returns GROUNDED or HALLUCINATED.

Only fires on the COMPLEX path (simple queries return extracted text, no risk).
One automatic retry with a stricter prompt if HALLUCINATED is detected.

Public API:
    check_hallucination(answer, context_chunks, groq_client, model) -> GuardrailResult
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    is_grounded: bool       # True = safe to show, False = potential hallucination
    label: str              # "GROUNDED" or "HALLUCINATED"
    explanation: str        # Brief explanation from the model
    raw_output: str         # Full raw model output for debugging


# ── Prompt ────────────────────────────────────────────────────────────────────

_GUARDRAIL_SYSTEM = (
    "You are a strict medical fact-checker. "
    "Your job is to determine whether an AI-generated answer contains any claims "
    "that are NOT supported by the provided source excerpts. "
    "You must reply with exactly one word on the first line: GROUNDED or HALLUCINATED. "
    "Then on the second line, give a one-sentence explanation."
)

_GUARDRAIL_USER_TEMPLATE = """\
SOURCE EXCERPTS:
{context}

AI ANSWER:
{answer}

Does the AI answer contain any claims not supported by the source excerpts?
Reply GROUNDED if everything is supported. Reply HALLUCINATED if any claim lacks support.
"""


# ── Core function ─────────────────────────────────────────────────────────────

def check_hallucination(
    answer: str,
    context_chunks: list[str],
    groq_client,
    model: str,
) -> GuardrailResult:
    """
    Check whether the answer is grounded in the provided context.

    Args:
        answer:         The generated answer text.
        context_chunks: List of raw chunk strings used as context.
        groq_client:    Initialized Groq client.
        model:          Groq model name to use for the check.

    Returns:
        GuardrailResult with is_grounded, label, explanation, raw_output.
    """
    if not answer or not context_chunks:
        # Nothing to check — treat as grounded
        return GuardrailResult(
            is_grounded=True,
            label="GROUNDED",
            explanation="No answer or context provided.",
            raw_output="",
        )

    # Truncate context to ~2000 chars to stay within token limits
    combined_context = "\n---\n".join(c.strip() for c in context_chunks)
    if len(combined_context) > 2000:
        combined_context = combined_context[:2000] + "\n[...truncated for guardrail check]"

    user_msg = _GUARDRAIL_USER_TEMPLATE.format(
        context=combined_context,
        answer=answer.strip(),
    )

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _GUARDRAIL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=80,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Guardrail check failed — treating as grounded: %s", exc)
        return GuardrailResult(
            is_grounded=True,
            label="GROUNDED",
            explanation="Guardrail check failed, defaulting to grounded.",
            raw_output=str(exc),
        )

    # Parse response: first token is GROUNDED or HALLUCINATED
    first_line = raw.split("\n")[0].strip().upper()
    rest_lines = "\n".join(raw.split("\n")[1:]).strip()
    explanation = rest_lines if rest_lines else raw

    is_hallucinated = bool(re.search(r"HALLUCINATED", first_line))
    label = "HALLUCINATED" if is_hallucinated else "GROUNDED"

    return GuardrailResult(
        is_grounded=not is_hallucinated,
        label=label,
        explanation=explanation,
        raw_output=raw,
    )
