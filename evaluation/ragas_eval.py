"""
QAMed — RAGAS Async Evaluation Pipeline (v1.5)

Runs RAGAS faithfulness + answer_relevancy + context_precision metrics
asynchronously in a background thread so the user sees their answer
immediately and scores appear after a few seconds.

LLM backend: langchain-groq (uses existing GROQ_API_KEY, no new cost beyond
~500 extra tokens per evaluation call).

Public API:
    run_eval_async(query, answer, contexts, session_state_key, tracer)
        → fires background thread, writes results to st.session_state

    RagasScores  — dataclass with faithfulness, answer_relevancy,
                   context_precision fields (all 0.0–1.0 or None on error)
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
    context_precision: float | None = None     # Were the retrieved chunks useful?
    error: str | None = None                   # Non-None if evaluation failed
    query: str = ""


# ── Core evaluation (runs in background thread) ───────────────────────────────

def _run_ragas(
    query: str,
    answer: str,
    contexts: list[str],
    session_state: dict,
    session_key: str,
    tracer,
) -> None:
    """
    Execute RAGAS evaluation and write results to session_state[session_key].
    Always runs in a daemon thread — never blocks the Streamlit main thread.
    """
    from config import GROQ_API_KEY, GROQ_MODEL  # imported inside thread to avoid circular

    scores = RagasScores(query=query)

    try:
        # ── Lazy imports to avoid slowing app startup ──────────────────────
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            faithfulness,
            answer_relevancy,
            context_precision,
        )
        from langchain_groq import ChatGroq  # type: ignore
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

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

        # Reuse the same embedding model config as the main app
        from config import EMBEDDING_MODEL_NAME
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # ── Run evaluation ─────────────────────────────────────────────────
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        df = result.to_pandas()
        row = df.iloc[0]

        scores.faithfulness = _safe_float(row.get("faithfulness"))
        scores.answer_relevancy = _safe_float(row.get("answer_relevancy"))
        scores.context_precision = _safe_float(row.get("context_precision"))

        logger.info(
            "RAGAS scores — faithfulness=%.2f relevancy=%.2f precision=%.2f",
            scores.faithfulness or 0,
            scores.answer_relevancy or 0,
            scores.context_precision or 0,
        )

        # ── Log to Langfuse if available ───────────────────────────────────
        if tracer and tracer.enabled:
            try:
                with tracer.trace("ragas-eval", query=query) as tr:
                    tracer.log_generation(
                        tr,
                        name="ragas",
                        prompt=query,
                        completion=answer,
                        model=GROQ_MODEL,
                        latency_ms=0,
                        metadata={
                            "faithfulness": scores.faithfulness,
                            "answer_relevancy": scores.answer_relevancy,
                            "context_precision": scores.context_precision,
                        },
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
        session_state[session_key] = scores


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
    session_state: dict,
    session_key: str,
    tracer=None,
) -> None:
    """
    Fire a background thread to evaluate the answer with RAGAS.
    Results are written to session_state[session_key] when done.

    The caller should poll session_state[session_key] (e.g. via st.rerun
    or checking in a later render pass) to display scores.

    Args:
        query:         The user's original question.
        answer:        The generated answer text.
        contexts:      List of raw chunk strings used as retrieval context.
        session_state: Reference to st.session_state (or any dict).
        session_key:   Key to write the RagasScores result into.
        tracer:        Optional Langfuse tracer for logging.
    """
    session_state[session_key] = None  # mark as "pending"

    thread = threading.Thread(
        target=_run_ragas,
        args=(query, answer, contexts, session_state, session_key, tracer),
        daemon=True,
        name=f"ragas-eval-{session_key}",
    )
    thread.start()
