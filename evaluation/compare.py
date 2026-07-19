import os
import json
import pandas as pd

def generate_report():
    results_file = os.path.join(os.path.dirname(__file__), "results", "benchmark_results.json")
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # Group by method and prompt_version
    summary = df.groupby(["method", "prompt_version"]).agg(
        avg_retrieval_latency=("retrieval_latency_ms", "mean"),
        avg_generation_latency=("generation_latency_ms", "mean"),
        avg_faithfulness=("faithfulness", "mean"),
        avg_answer_relevancy=("answer_relevancy", "mean"),
        grounded_ratio=("guardrail_is_grounded", "mean"),
        avg_overlap_ratio=("overlap_ratio", "mean")
    ).reset_index()

    # Format numbers
    summary["avg_retrieval_latency"] = summary["avg_retrieval_latency"].round(2)
    summary["avg_generation_latency"] = summary["avg_generation_latency"].round(2)
    summary["avg_faithfulness"] = summary["avg_faithfulness"].round(4)
    summary["avg_answer_relevancy"] = summary["avg_answer_relevancy"].round(4)
    summary["grounded_ratio"] = summary["grounded_ratio"].round(2)
    summary["avg_overlap_ratio"] = summary["avg_overlap_ratio"].round(2)

    report_lines = [
        "# Benchmark Results: Retrieval Methods × Prompt Versions",
        "",
        "## Summary Metrics",
        "",
        summary.to_markdown(index=False),
        "",
        "## Detailed Results",
        ""
    ]

    for q_idx, query in enumerate(df["question"].unique(), 1):
        report_lines.append(f"### Question {q_idx}: {query}")
        q_df = df[df["question"] == query]
        
        table_df = q_df[["method", "prompt_version", "guardrail_is_grounded", "faithfulness", "answer_relevancy", "retrieval_latency_ms", "overlap_ratio"]].copy()
        report_lines.append(table_df.to_markdown(index=False))
        report_lines.append("")

    report_path = os.path.join(os.path.dirname(__file__), "results", "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    generate_report()
