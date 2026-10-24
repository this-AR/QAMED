"""
QAMed — RAGAS Async Evaluation Pipeline (v1.5)

Runs RAGAS faithfulness + answer_relevancy metrics asynchronously in a
background thread so the user sees their answer immediately and scores
appear on the next rerun.

LLM backend: langchain-groq (uses existing GROQ_API_KEY, no new cost beyond
~500 extra tokens per evaluation call).

Public API:
    run_eval_async(query, answer, contexts, results, key, embeddings, tracer)
        → fires background thread, writes RagasScores to results[key]

    RagasScores  — dataclass with faithfulness, answer_relevancy
                   fields (all 0.0–1.0 or None on error)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class RagasScores:
    faithfulness: float | None = None          # Is the answer grounded in context?
    answer_relevancy: float | None = None      # Does the answer address the question?
    error: str | None = None                   # Non-None if evaluation failed
    query: str = ""


# ── Core evaluation (runs in background thread) ───────────────────────────────

def _run_ragas(
    query: str,
    answer: str,
    contexts: list[str],
    results: dict,
    key: str,
    embeddings,
    tracer=None,
    trace_id: str | None = None,
) -> None:
    """
    Execute RAGAS evaluation and write results to results[key].
    Always runs in a daemon thread — never blocks the Streamlit main thread.
    """
    from config import GROQ_API_KEY, GROQ_MODEL  # imported inside thread to avoid circular

    scores = RagasScores(query=query)

    try:
        # ── Lazy imports to avoid slowing app startup ──────────────────────
        import pandas as pd
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            faithfulness,
            answer_relevancy,
        )
        from langchain_groq import ChatGroq  # type: ignore

        # ── Build dataset ──────────────────────────────────────────────────
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            # ground_truth is optional for faithfulness + relevancy
        }
        dataset = Dataset.from_dict(data)

        # ── LLM + embeddings for RAGAS ─────────────────────────────────────
        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
        )

        # ── Run evaluation ─────────────────────────────────────────────────
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
            ],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        df = result.to_pandas()
        row = df.iloc[0]

        scores.faithfulness = _safe_float(row.get("faithfulness"))
        scores.answer_relevancy = _safe_float(row.get("answer_relevancy"))

        logger.info(
            "RAGAS scores — faithfulness=%.2f relevancy=%.2f",
            scores.faithfulness or 0,
            scores.answer_relevancy or 0,
        )

        # ── Log to Langfuse if available ───────────────────────────────────
        if tracer and tracer.enabled and trace_id and hasattr(tracer._client, "create_score"):
            try:
                if scores.faithfulness is not None:
                    tracer._client.create_score(
                        trace_id=trace_id, name="faithfulness", value=scores.faithfulness, data_type="NUMERIC"
                    )
                if scores.answer_relevancy is not None:
                    tracer._client.create_score(
                        trace_id=trace_id, name="answer_relevancy", value=scores.answer_relevancy, data_type="NUMERIC"
                    )
            except Exception as exc:
                logger.warning("RAGAS Langfuse logging failed: %s", exc)

    except ImportError as exc:
        scores.error = f"RAGAS not installed: {exc}"
        logger.warning("RAGAS import error: %s", exc)
    except Exception as exc:
        scores.error = str(exc)
        logger.warning("RAGAS evaluation failed: %s", exc)
    finally:
        results[key] = scores


def _safe_float(val: Any) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def run_eval_async(
    query: str,
    answer: str,
    contexts: list[str],
    results: dict,
    key: str,
    embeddings,
    tracer=None,
    trace_id: str | None = None,
) -> None:
    """
    Fire a background thread to evaluate the answer with RAGAS.
    Results are written to results[key] when done.

    The caller should read results[key] in a later render pass to display scores.

    Args:
        query:       The user's original question.
        answer:      The generated answer text.
        contexts:    List of raw chunk strings used as retrieval context.
        results:     A plain dict to write into. Pass a dict stored in
                     st.session_state, not st.session_state itself: writes to
                     st.session_state from a background thread have no script
                     context and never reach the user's session.
        key:         Key to write the RagasScores result into.
        embeddings:  Already-loaded LangChain embeddings (reused, not reloaded).
        tracer:      Optional Langfuse tracer for logging.
        trace_id:    Optional Langfuse trace ID to attach scores to.
    """
    results[key] = None  # mark as "pending"

    thread = threading.Thread(
        target=_run_ragas,
        args=(query, answer, contexts, results, key, embeddings, tracer, trace_id),
        daemon=True,
        name=f"ragas-eval-{key}",
    )
    thread.start()
