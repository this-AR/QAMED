"""
QAMed — Output Guardrails (v2.5)

Detects hallucinations before an answer reaches the user.

Strategy (two-tier):
  Tier 1 (Fast — always runs):
      Split the generated answer into sentences. For each sentence,
      compute cosine similarity against all retrieved context chunks.
      Classify the answer as GROUNDED if the average max-similarity
      across sentences is above GUARDRAIL_SIM_THRESHOLD (default 0.30).

      This eliminates false-positive hallucination flags that the old
      LLM-based checker produced against v3 (Expert Medical Synthesis)
      prompts, which write fluid narratives rather than verbatim quotes.

  Tier 2 (LLM fallback — only fires in ambiguous zone):
      If the cosine score falls in the grey zone [LOW_THRESHOLD, HIGH_THRESHOLD),
      fall back to the original LLM checker for a second opinion.

Benefits over the old pure-LLM approach:
  - No extra Groq API call in the common case (saves latency + tokens).
  - Mathematically objective — not confused by paraphrasing or synthesis.
  - Configurable thresholds via config.py.

Public API:
    check_hallucination(answer, context_chunks, groq_client, model,
                        embedding_fn=None) -> GuardrailResult
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pysbd

logger = logging.getLogger(__name__)

# Cosine similarity thresholds
# Sentences scoring above HIGH are definitely grounded.
# Sentences scoring below LOW are definitely hallucinated.
# Scores in [LOW, HIGH) go to the LLM fallback.
HIGH_THRESHOLD = 0.30   # average max-sim above this → GROUNDED (no LLM call)
LOW_THRESHOLD  = 0.18   # average max-sim below this → HALLUCINATED (no LLM call)
# Between LOW and HIGH → LLM fallback

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    is_grounded: bool       # True = safe to show, False = potential hallucination
    label: str              # "GROUNDED" or "HALLUCINATED"
    explanation: str        # Brief explanation
    raw_output: str         # Full raw output for debugging
    avg_similarity: float | None = None   # cosine similarity score (Tier 1)
    method: str = "cosine"               # "cosine" or "llm_fallback"


# ── Tier 1: Cosine Similarity Guardrail ──────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Fast dot-product cosine similarity for unit vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _cosine_guardrail(
    answer: str,
    context_chunks: list[str],
    embedding_fn,
) -> tuple[str, float]:
    """
    Returns (verdict, avg_max_similarity).
    verdict is one of: "GROUNDED", "HALLUCINATED", "AMBIGUOUS"
    """
    sentences = _SEGMENTER.segment(answer)
    # Filter out very short sentences (likely citations or fragments)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 5]

    if not sentences:
        return "GROUNDED", 1.0   # nothing to check

    # Embed all context chunks once
    chunk_vecs = [embedding_fn(c) for c in context_chunks]

    max_sims = []
    for sent in sentences:
        sent_vec = embedding_fn(sent)
        sims = [_cosine_similarity(sent_vec, cv) for cv in chunk_vecs]
        max_sims.append(max(sims) if sims else 0.0)

    avg_max = sum(max_sims) / len(max_sims)

    if avg_max >= HIGH_THRESHOLD:
        return "GROUNDED", avg_max
    elif avg_max < LOW_THRESHOLD:
        return "HALLUCINATED", avg_max
    else:
        return "AMBIGUOUS", avg_max


# ── Tier 2: LLM Fallback (ambiguous zone only) ───────────────────────────────

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


def _llm_guardrail(
    answer: str,
    context_chunks: list[str],
    groq_client,
    model: str,
) -> tuple[bool, str, str]:
    """
    Returns (is_grounded, label, explanation).
    """
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
        logger.warning("LLM guardrail fallback failed — treating as grounded: %s", exc)
        return True, "GROUNDED", f"LLM fallback failed: {exc}"

    first_line = raw.split("\n")[0].strip().upper()
    rest_lines = "\n".join(raw.split("\n")[1:]).strip()
    explanation = rest_lines if rest_lines else raw

    is_hallucinated = bool(re.search(r"HALLUCINATED", first_line))
    label = "HALLUCINATED" if is_hallucinated else "GROUNDED"
    return not is_hallucinated, label, explanation


# ── Public API ────────────────────────────────────────────────────────────────

def check_hallucination(
    answer: str,
    context_chunks: list[str],
    groq_client,
    model: str,
    embedding_fn=None,
) -> GuardrailResult:
    """
    Check whether the answer is grounded in the provided context.

    Args:
        answer:         The generated answer text.
        context_chunks: List of raw chunk strings used as context.
        groq_client:    Initialized Groq client (used only for LLM fallback).
        model:          Groq model name (used only for LLM fallback).
        embedding_fn:   Optional callable (str -> list[float]). If provided,
                        uses fast cosine similarity as Tier 1. If None,
                        falls back directly to LLM-only mode.

    Returns:
        GuardrailResult with is_grounded, label, explanation, avg_similarity, method.
    """
    if not answer or not context_chunks:
        return GuardrailResult(
            is_grounded=True,
            label="GROUNDED",
            explanation="No answer or context provided.",
            raw_output="",
            avg_similarity=None,
            method="cosine",
        )

    # ── Tier 1: Cosine Similarity ─────────────────────────────────────────────
    if embedding_fn is not None:
        try:
            verdict, avg_sim = _cosine_guardrail(answer, context_chunks, embedding_fn)
            logger.info("Cosine guardrail: verdict=%s avg_sim=%.3f", verdict, avg_sim)

            if verdict == "GROUNDED":
                return GuardrailResult(
                    is_grounded=True,
                    label="GROUNDED",
                    explanation=f"Cosine similarity {avg_sim:.3f} ≥ {HIGH_THRESHOLD} threshold.",
                    raw_output=f"avg_max_sim={avg_sim:.4f}",
                    avg_similarity=avg_sim,
                    method="cosine",
                )
            elif verdict == "HALLUCINATED":
                return GuardrailResult(
                    is_grounded=False,
                    label="HALLUCINATED",
                    explanation=f"Cosine similarity {avg_sim:.3f} < {LOW_THRESHOLD} threshold.",
                    raw_output=f"avg_max_sim={avg_sim:.4f}",
                    avg_similarity=avg_sim,
                    method="cosine",
                )
            # else: AMBIGUOUS → fall through to LLM
            logger.info("Cosine score %.3f in ambiguous zone — running LLM fallback.", avg_sim)

        except Exception as exc:
            logger.warning("Cosine guardrail failed — falling back to LLM: %s", exc)
            avg_sim = None

    else:
        avg_sim = None

    # ── Tier 2: LLM Fallback ─────────────────────────────────────────────────
    is_grounded, label, explanation = _llm_guardrail(
        answer, context_chunks, groq_client, model
    )
    return GuardrailResult(
        is_grounded=is_grounded,
        label=label,
        explanation=explanation,
        raw_output=explanation,
        avg_similarity=avg_sim,
        method="llm_fallback",
    )
