# evaluation/ — Metrics, validation, and quality tracking
#
# Current modules:
#   ragas_eval.py      — RAGAS async evaluation pipeline (v1.5)
#                        Metrics: faithfulness, answer_relevancy
#                        Runs in background thread, results stored in session_state["ragas_results"]
#   benchmark.py       — Offline hybrid vs dense × prompt-version benchmark
#   compare.py         — Summarizes benchmark results into a report
#   test_questions.py  — Benchmark question set
