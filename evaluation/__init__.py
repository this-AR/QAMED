# evaluation/ — Metrics, validation, and quality tracking
#
# Current modules:
#   ragas_eval.py      — RAGAS async evaluation pipeline (v1.5)
#                        Metrics: faithfulness, answer_relevancy, context_precision
#                        Runs in background thread, results stored in session_state
#
# Future modules (per roadmap):
#   golden_dataset.py  — LLM-generated QA pairs for eval (v2.0)
#   exam_eval.py       — MBBS exam question evaluation (v2.0)
#   ab_testing.py      — A/B prompt version testing (v3.0)
