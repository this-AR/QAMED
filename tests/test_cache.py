"""Cache key and semantic-cache behaviour, using fakes (no Redis/Qdrant needed)."""
from types import SimpleNamespace

import services.cache as cache


class FakeQdrant:
    def __init__(self, points=None):
        self.points = points or []
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.points)


def test_query_key_depends_on_variant():
    assert cache._query_key("What is X?", "v3|decomp=True") != cache._query_key("What is X?", "v1|decomp=True")
    assert cache._query_key("What is X?", "v3|decomp=True") != cache._query_key("What is X?", "v3|decomp=False")
    assert cache._query_key("what is  x?", "v3") == cache._query_key("What is X?", "v3")


def test_semantic_cache_uses_query_points_with_variant_filter(monkeypatch):
    monkeypatch.setattr(cache, "_get_redis", lambda: None)
    hit = SimpleNamespace(score=0.95, payload={"answer": "A", "sources_text": "S", "prompt_version": "v3"})
    client = FakeQdrant([hit])

    result = cache.check_cache("What is X?", "v3|decomp=True", lambda q: [0.1, 0.2], client)

    assert result is not None and result.hit_type == "semantic" and result.answer == "A"
    variant_cond, age_cond = client.calls[0]["query_filter"].must
    assert variant_cond.key == "variant" and variant_cond.match.value == "v3|decomp=True"
    assert age_cond.key == "cached_at" and age_cond.range.gte > 0  # entries older than the TTL never match


def test_semantic_threshold_separates_different_questions():
    """Real embedder: a different anatomical question must not hit the other's cached answer."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    from config import EMBEDDING_MODEL_NAME

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    a, b = model.encode([
        "What is the inguinal canal and what are its contents?",
        "What is the femoral canal and what are its contents?",
    ])
    assert float(cos_sim(a, b)) < cache.SEMANTIC_CACHE_THRESHOLD
