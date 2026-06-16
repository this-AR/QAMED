"""
QAMed — Cache Service (v1.5)

Two-level caching:
  Level 1: Exact-match via Redis (Upstash or any Redis-compatible host).
            Key = SHA-256 of normalized query. TTL = CACHE_TTL_SECONDS.
  Level 2: Semantic cache via Qdrant query_cache collection.
            Embeds the query → cosine similarity check → threshold 0.92.

Both levels gracefully degrade: if credentials are absent the cache is
simply skipped and the full pipeline runs as normal.

Public API:
    check_cache(query, embedding_fn, qdrant_client) -> CacheResult | None
    store_in_cache(query, answer, sources_text, prompt_version, embedding_fn, qdrant_client)
    clear_cache(qdrant_client)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from config import (
    REDIS_URL,
    CACHE_TTL_SECONDS,
    SEMANTIC_CACHE_THRESHOLD,
    SEMANTIC_CACHE_COLLECTION,
)

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(query: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(query.lower().strip().split())


def _query_key(query: str) -> str:
    return "qamed:" + hashlib.sha256(_normalize(query).encode()).hexdigest()


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class CacheResult:
    answer: str
    sources_text: str
    prompt_version: str | None
    hit_type: str  # "exact" or "semantic"
    similarity: float | None = None  # only set for semantic hits


# ── Redis (Level 1 — Exact Match) ────────────────────────────────────────────

def _get_redis():
    """Return a Redis client or None if REDIS_URL is not configured."""
    if not REDIS_URL:
        return None
    try:
        import redis  # type: ignore
        client = redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception as exc:
        logger.warning("Redis unavailable, skipping exact-match cache: %s", exc)
        return None


def _exact_cache_get(query: str) -> CacheResult | None:
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.get(_query_key(query))
        if raw:
            data = json.loads(raw)
            return CacheResult(
                answer=data["answer"],
                sources_text=data.get("sources_text", ""),
                prompt_version=data.get("prompt_version"),
                hit_type="exact",
            )
    except Exception as exc:
        logger.warning("Exact cache read error: %s", exc)
    return None


def _exact_cache_set(query: str, answer: str, sources_text: str, prompt_version: str | None):
    r = _get_redis()
    if r is None:
        return
    try:
        payload = json.dumps({
            "answer": answer,
            "sources_text": sources_text,
            "prompt_version": prompt_version,
            "cached_at": time.time(),
        })
        r.setex(_query_key(query), CACHE_TTL_SECONDS, payload)
    except Exception as exc:
        logger.warning("Exact cache write error: %s", exc)


# ── Qdrant (Level 2 — Semantic Cache) ────────────────────────────────────────

def _ensure_semantic_collection(qdrant_client, vector_size: int = 768):
    """Create the query_cache collection if it doesn't exist."""
    try:
        from qdrant_client.models import Distance, VectorParams  # type: ignore
        existing = {c.name for c in qdrant_client.get_collections().collections}
        if SEMANTIC_CACHE_COLLECTION not in existing:
            qdrant_client.create_collection(
                collection_name=SEMANTIC_CACHE_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("Created semantic cache collection: %s", SEMANTIC_CACHE_COLLECTION)
    except Exception as exc:
        logger.warning("Could not ensure semantic cache collection: %s", exc)


def _semantic_cache_get(query: str, embedding_fn: Callable, qdrant_client) -> CacheResult | None:
    if qdrant_client is None:
        return None
    try:
        vector = embedding_fn(query)
        results = qdrant_client.search(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            query_vector=vector,
            limit=1,
            score_threshold=SEMANTIC_CACHE_THRESHOLD,
        )
        if results:
            hit = results[0]
            payload = hit.payload or {}
            return CacheResult(
                answer=payload.get("answer", ""),
                sources_text=payload.get("sources_text", ""),
                prompt_version=payload.get("prompt_version"),
                hit_type="semantic",
                similarity=round(hit.score, 4),
            )
    except Exception as exc:
        logger.warning("Semantic cache read error: %s", exc)
    return None


def _semantic_cache_set(
    query: str,
    answer: str,
    sources_text: str,
    prompt_version: str | None,
    embedding_fn: Callable,
    qdrant_client,
):
    if qdrant_client is None:
        return
    try:
        vector = embedding_fn(query)
        _ensure_semantic_collection(qdrant_client, vector_size=len(vector))
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, _normalize(query)))
        from qdrant_client.models import PointStruct  # type: ignore
        qdrant_client.upsert(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "query": _normalize(query),
                        "answer": answer,
                        "sources_text": sources_text,
                        "prompt_version": prompt_version,
                        "cached_at": time.time(),
                    },
                )
            ],
        )
    except Exception as exc:
        logger.warning("Semantic cache write error: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def check_cache(
    query: str,
    embedding_fn: Callable[[str], list[float]],
    qdrant_client,
) -> CacheResult | None:
    """
    Check both cache levels in order.
    Returns CacheResult on hit, None on miss.

    Args:
        query:         Raw user query string.
        embedding_fn:  Callable that takes a string and returns a float list.
        qdrant_client: Initialized Qdrant client (used for semantic cache).
    """
    # Level 1: exact match
    result = _exact_cache_get(query)
    if result:
        return result

    # Level 2: semantic match
    result = _semantic_cache_get(query, embedding_fn, qdrant_client)
    return result


def store_in_cache(
    query: str,
    answer: str,
    sources_text: str,
    prompt_version: str | None,
    embedding_fn: Callable[[str], list[float]],
    qdrant_client,
):
    """
    Store the answer in both cache levels.
    Safe to call unconditionally — will no-op if credentials are absent.
    """
    _exact_cache_set(query, answer, sources_text, prompt_version)
    _semantic_cache_set(query, answer, sources_text, prompt_version, embedding_fn, qdrant_client)


def clear_cache(qdrant_client=None):
    """
    Flush both caches. Useful for testing.
    """
    r = _get_redis()
    if r:
        try:
            keys = r.keys("qamed:*")
            if keys:
                r.delete(*keys)
                logger.info("Cleared %d exact-match cache entries.", len(keys))
        except Exception as exc:
            logger.warning("Could not clear Redis cache: %s", exc)

    if qdrant_client:
        try:
            existing = {c.name for c in qdrant_client.get_collections().collections}
            if SEMANTIC_CACHE_COLLECTION in existing:
                qdrant_client.delete_collection(SEMANTIC_CACHE_COLLECTION)
                logger.info("Cleared semantic cache collection.")
        except Exception as exc:
            logger.warning("Could not clear semantic cache: %s", exc)
