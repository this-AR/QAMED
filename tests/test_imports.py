"""
QAMed — Import Smoke Tests

Run this after ANY code change to verify the module structure is intact.

Usage:
    python -m pytest tests/test_imports.py -v
    OR
    python tests/test_imports.py
"""

import sys
import os

# Ensure the project root is on sys.path so imports work
# regardless of where the script is run from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_config_imports():
    from config import (
        QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY, GROQ_MODEL,
        COLLECTION_NAME, EMBEDDING_MODEL_NAME, DEFAULT_PROMPT_VERSION,
        REDIS_URL, CACHE_TTL_SECONDS, SEMANTIC_CACHE_THRESHOLD,
        SEMANTIC_CACHE_COLLECTION, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
    )
    assert isinstance(GROQ_MODEL, str)
    assert isinstance(COLLECTION_NAME, str)
    assert isinstance(EMBEDDING_MODEL_NAME, str)
    assert isinstance(DEFAULT_PROMPT_VERSION, str)


def test_prompts_imports():
    from prompts import PROMPT_TEMPLATES, resolve_prompt_version
    assert "v1" in PROMPT_TEMPLATES
    assert "v2" in PROMPT_TEMPLATES
    assert resolve_prompt_version("v1") == "v1"
    assert resolve_prompt_version("v999") == "v1"  # fallback


def test_services_llm_imports():
    from services.llm import (
        extract_subquestions, classify_query, build_prompt, stream_groq_answer,
    )
    assert callable(extract_subquestions)
    assert callable(classify_query)
    assert callable(build_prompt)
    assert callable(stream_groq_answer)


def test_services_retrieval_imports():
    from services.retrieval import load_models_and_clients, rerank_docs
    assert callable(load_models_and_clients)
    assert callable(rerank_docs)


def test_services_cache_imports():
    from services.cache import check_cache, store_in_cache, clear_cache, CacheResult
    assert callable(check_cache)
    assert callable(store_in_cache)
    assert callable(clear_cache)


def test_services_guardrails_imports():
    from services.guardrails import check_hallucination, GuardrailResult
    assert callable(check_hallucination)


def test_services_observability_imports():
    from services.observability import get_tracer, Timer, LangfuseTracer, NoOpTracer
    assert callable(get_tracer)


def test_ui_components_imports():
    from ui.components import (
        render_simple_answer, render_sources, render_rating, render_run_history,
        render_cache_badge, render_guardrail_badge, render_ragas_scores,
    )
    assert callable(render_simple_answer)
    assert callable(render_sources)
    assert callable(render_rating)
    assert callable(render_run_history)
    assert callable(render_cache_badge)
    assert callable(render_guardrail_badge)
    assert callable(render_ragas_scores)


def test_data_ingest_imports():
    from data.ingest import (
        load_and_chunk_pdf, detect_labels, get_chapter_by_page,
        segment_text_with_pysbd, ensure_collection, CHAPTER_MAP,
    )
    assert callable(load_and_chunk_pdf)
    assert callable(detect_labels)
    assert len(CHAPTER_MAP) > 0


def test_evaluation_ragas_imports():
    from evaluation.ragas_eval import run_eval_async, RagasScores
    assert callable(run_eval_async)


def test_rerank_docs_empty():
    """rerank_docs should handle empty input gracefully."""
    from services.retrieval import rerank_docs
    result = rerank_docs(None, "test query", [], top_n=3)
    assert result == []


def test_detect_labels():
    """Verify label detection works for known patterns."""
    from data.ingest import detect_labels
    assert "@definition" in detect_labels("Hernia is defined as a protrusion")
    assert "@treatment" in detect_labels("Treatment includes surgery")
    assert "@general" in detect_labels("The abdomen is a body cavity")


def test_resolve_prompt_version_edge_cases():
    from prompts import resolve_prompt_version
    assert resolve_prompt_version("") == "v1"
    assert resolve_prompt_version("v2") == "v2"
    assert resolve_prompt_version("nonexistent") == "v1"


if __name__ == "__main__":
    tests = [
        test_config_imports,
        test_prompts_imports,
        test_services_llm_imports,
        test_services_retrieval_imports,
        test_services_cache_imports,
        test_services_guardrails_imports,
        test_services_observability_imports,
        test_ui_components_imports,
        test_data_ingest_imports,
        test_evaluation_ragas_imports,
        test_rerank_docs_empty,
        test_detect_labels,
        test_resolve_prompt_version_edge_cases,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{len(tests)} passed, {failed} failed.")
