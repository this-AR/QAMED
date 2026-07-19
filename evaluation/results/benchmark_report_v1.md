# Benchmark Results: Retrieval Methods × Prompt Versions

## Summary Metrics

| method   | prompt_version   |   avg_retrieval_latency |   avg_generation_latency |   avg_faithfulness |   avg_answer_relevancy |   grounded_ratio |   avg_overlap_ratio |
|:---------|:-----------------|------------------------:|-------------------------:|-------------------:|-----------------------:|-----------------:|--------------------:|
| dense    | v1               |                 16876.3 |                  1642.26 |             0.5    |                    nan |             1    |                0.75 |
| dense    | v2               |                 16876.3 |                   639.51 |             0.4083 |                    nan |             1    |                0.75 |
| dense    | v3               |                 16876.3 |                   995.56 |             0.5511 |                    nan |             0.75 |                0.75 |
| hybrid   | v1               |                 23372   |                   990.82 |             0.5402 |                    nan |             0.75 |                0.75 |
| hybrid   | v2               |                 23372   |                   750.84 |             0.4449 |                    nan |             0.75 |                0.75 |
| hybrid   | v3               |                 23372   |                  1103.84 |             0.7222 |                    nan |             0.75 |                0.75 |

## Detailed Results

### Question 1: what is inguinal canal and what are its content ?
| method   | prompt_version   | guardrail_is_grounded   |   faithfulness |   answer_relevancy |   retrieval_latency_ms |   overlap_ratio |
|:---------|:-----------------|:------------------------|---------------:|-------------------:|-----------------------:|----------------:|
| hybrid   | v1               | True                    |       nan      |                nan |                37620.8 |               1 |
| hybrid   | v2               | True                    |         0.375  |                nan |                37620.8 |               1 |
| hybrid   | v3               | False                   |       nan      |                nan |                37620.8 |               1 |
| dense    | v1               | True                    |       nan      |                nan |                26635.9 |               1 |
| dense    | v2               | True                    |         0.3333 |                nan |                26635.9 |               1 |
| dense    | v3               | False                   |       nan      |                nan |                26635.9 |               1 |

### Question 2: Why is the appendix called a "surgical organ"?
| method   | prompt_version   | guardrail_is_grounded   |   faithfulness |   answer_relevancy |   retrieval_latency_ms |   overlap_ratio |
|:---------|:-----------------|:------------------------|---------------:|-------------------:|-----------------------:|----------------:|
| hybrid   | v1               | False                   |         0.6207 |                nan |                20304.8 |             0.5 |
| hybrid   | v2               | True                    |         0.5714 |                nan |                20304.8 |             0.5 |
| hybrid   | v3               | True                    |         0.5556 |                nan |                20304.8 |             0.5 |
| dense    | v1               | True                    |         0.5    |                nan |                14825.3 |             0.5 |
| dense    | v2               | True                    |         1      |                nan |                14825.3 |             0.5 |
| dense    | v3               | True                    |         0.7083 |                nan |                14825.3 |             0.5 |

### Question 3: Why is the left kidney located higher than the right kidney?
| method   | prompt_version   | guardrail_is_grounded   |   faithfulness |   answer_relevancy |   retrieval_latency_ms |   overlap_ratio |
|:---------|:-----------------|:------------------------|---------------:|-------------------:|-----------------------:|----------------:|
| hybrid   | v1               | True                    |         0      |                nan |                19498.7 |               1 |
| hybrid   | v2               | True                    |         0      |                nan |                19498.7 |               1 |
| hybrid   | v3               | True                    |         1      |                nan |                19498.7 |               1 |
| dense    | v1               | True                    |         0      |                nan |                13300.3 |               1 |
| dense    | v2               | True                    |         0      |                nan |                13300.3 |               1 |
| dense    | v3               | True                    |         0.4118 |                nan |                13300.3 |               1 |

### Question 4: Why are the renal arteries considered end arteries?
| method   | prompt_version   | guardrail_is_grounded   |   faithfulness |   answer_relevancy |   retrieval_latency_ms |   overlap_ratio |
|:---------|:-----------------|:------------------------|---------------:|-------------------:|-----------------------:|----------------:|
| hybrid   | v1               | True                    |         1      |                nan |                16063.6 |             0.5 |
| hybrid   | v2               | False                   |         0.8333 |                nan |                16063.6 |             0.5 |
| hybrid   | v3               | True                    |         0.6111 |                nan |                16063.6 |             0.5 |
| dense    | v1               | True                    |         1      |                nan |                12743.8 |             0.5 |
| dense    | v2               | True                    |         0.3    |                nan |                12743.8 |             0.5 |
| dense    | v3               | True                    |         0.5333 |                nan |                12743.8 |             0.5 |
