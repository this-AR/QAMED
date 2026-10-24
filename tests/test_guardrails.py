"""Cosine guardrail tier, using a fake batch embedder (no model or LLM needed)."""
from services.guardrails import check_hallucination


def test_cosine_guardrail_embeds_in_one_batch():
    calls = []

    def embed_batch(texts):
        calls.append(list(texts))
        return [[1.0, 0.0] if "kidney" in t else [0.0, 1.0] for t in texts]

    answer = "The kidney filters blood through the nephrons. The kidney also regulates electrolytes."
    result = check_hallucination(answer, ["The kidney filters blood."], None, "model", embed_batch_fn=embed_batch)

    assert result.is_grounded and result.method == "cosine"
    assert len(calls) == 1 and len(calls[0]) == 3  # 1 context + 2 sentences


def test_failed_llm_fallback_is_not_grounded():
    """Grey-zone score + Groq error (e.g. 429) must not pass as grounded."""
    class FailingGroq:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("429 rate limited")

    # Cosine 0.3 sits between LOW (0.18) and HIGH (0.45), so the LLM fallback runs
    embed_batch = lambda texts: [[1.0, 0.0]] + [[0.3, 0.954]] * (len(texts) - 1)
    answer = "The kidney filters blood through the nephrons every single day."
    result = check_hallucination(answer, ["The kidney filters blood."], FailingGroq(), "model", embed_batch_fn=embed_batch)

    assert not result.is_grounded and result.label == "UNVERIFIED"
