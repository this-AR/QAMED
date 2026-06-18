"""
QAMed — Observability Service (v1.5)

Langfuse tracing — opt-in.

Setup:
  Add to .env:
      LANGFUSE_PUBLIC_KEY=pk-lf-...
      LANGFUSE_SECRET_KEY=sk-lf-...
      LANGFUSE_HOST=https://cloud.langfuse.com   (optional, defaults to cloud)

If credentials are absent, all functions return no-op stubs and the app
runs exactly as before with zero overhead.

Public API:
    get_tracer() -> LangfuseTracer | NoOpTracer
    tracer.trace(name, query) -> Trace context manager
    tracer.log_generation(trace, name, prompt, completion, model, latency_ms, metadata)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

logger = logging.getLogger(__name__)

# ── Singleton ─────────────────────────────────────────────────────────────────
_tracer_instance = None


# ── No-Op Tracer (used when Langfuse is not configured) ───────────────────────

class _NoOpTrace:
    """Returned by NoOpTracer.trace() — all methods do nothing."""
    id = "noop"

    def log_generation(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass


class NoOpTracer:
    """Returned when LANGFUSE_PUBLIC_KEY is absent. Zero overhead."""
    enabled = False

    @contextmanager
    def trace(self, name: str, query: str = "") -> Generator[_NoOpTrace, None, None]:
        yield _NoOpTrace()

    def log_generation(self, *args, **kwargs):
        pass

    def flush(self):
        pass


# ── Real Langfuse Tracer ──────────────────────────────────────────────────────

class LangfuseTracer:
    """Thin wrapper around the Langfuse Python SDK."""
    enabled = True

    def __init__(self, client):
        self._client = client

    @contextmanager
    def trace(self, name: str, query: str = "") -> Generator[Any, None, None]:
        """
        Context manager for a top-level Langfuse trace.

        Usage:
            with tracer.trace("query-pipeline", query=user_query) as tr:
                tracer.log_generation(tr, ...)
        """
        with self._client.start_as_current_observation(as_type="span", name=name, input=query) as span:
            try:
                yield span
            except Exception:
                span.update(status_message="error")
                raise

    def log_generation(
        self,
        trace,
        name: str,
        prompt: str,
        completion: str,
        model: str,
        latency_ms: float,
        metadata: dict | None = None,
    ):
        """
        Log a single LLM generation span under an existing trace.

        Args:
            trace:       Trace object from tracer.trace() context.
            name:        Span name (e.g. "decomposition", "generation", "guardrail").
            prompt:      Full prompt sent to the model.
            completion:  Model response text.
            model:       Model name used.
            latency_ms:  Round-trip latency in milliseconds.
            metadata:    Optional extra fields (e.g. cache_hit, prompt_version).
        """
        if trace is None:
            return
        try:
            with trace.start_as_current_observation(
                as_type="generation",
                name=name,
                model=model,
                input=prompt,
            ) as gen:
                gen.update(
                    output=completion,
                    metadata={
                        "latency_ms": round(latency_ms, 1),
                        **(metadata or {}),
                    },
                )
        except Exception as exc:
            logger.warning("Langfuse log_generation failed: %s", exc)

    def flush(self):
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_tracer() -> LangfuseTracer | NoOpTracer:
    """
    Return the singleton tracer.
    Builds a real LangfuseTracer if credentials are present, otherwise NoOpTracer.
    Safe to call repeatedly — always returns the same instance.
    """
    global _tracer_instance
    if _tracer_instance is not None:
        return _tracer_instance

    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        logger.info("Langfuse credentials not set — tracing disabled.")
        _tracer_instance = NoOpTracer()
        return _tracer_instance

    try:
        from langfuse import Langfuse  # type: ignore
        kwargs: dict = {
            "public_key": LANGFUSE_PUBLIC_KEY,
            "secret_key": LANGFUSE_SECRET_KEY,
        }
        if LANGFUSE_HOST:
            kwargs["host"] = LANGFUSE_HOST
        client = Langfuse(**kwargs)
        _tracer_instance = LangfuseTracer(client)
        logger.info("Langfuse tracing enabled.")
    except ImportError:
        logger.warning("langfuse package not installed — tracing disabled.")
        _tracer_instance = NoOpTracer()
    except Exception as exc:
        logger.warning("Langfuse init failed — tracing disabled: %s", exc)
        _tracer_instance = NoOpTracer()

    return _tracer_instance


# ── Utility: latency timer ────────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer. Usage: t = Timer(); t.elapsed_ms()."""

    def __init__(self):
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000
