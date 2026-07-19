# Benchmark Results: Retrieval Methods × Prompt Versions

## Summary Metrics

| method   | prompt_version   |   avg_retrieval_latency |   avg_generation_latency |   avg_faithfulness |   avg_answer_relevancy |   grounded_ratio |   avg_overlap_ratio |
|:---------|:-----------------|------------------------:|-------------------------:|-------------------:|-----------------------:|-----------------:|--------------------:|
| dense    | v1               |                 3786.9  |                   724.44 |             0.4444 |                    nan |             0.75 |                0.92 |
| dense    | v2               |                 3786.9  |                   829.28 |             0.5962 |                    nan |             0.75 |                0.92 |
| dense    | v3               |                 3786.9  |                  1044.76 |             0.8314 |                      1 |             1    |                0.92 |
| hybrid   | v1               |                 3549.68 |                  7076.16 |             0.6413 |                    nan |             1    |                0.92 |
| hybrid   | v2               |                 3549.68 |                   698.22 |             0.696  |                    nan |             0.75 |                0.92 |
| hybrid   | v3               |                 3549.68 |                   986.44 |             0.8636 |                    nan |             0.75 |                0.92 |

## Detailed Results

### Question 1: what is inguinal canal and what are its content ?
| method   | prompt_version   | guardrail_is_grounded   |   faithfulness |   answer_relevancy |   retrieval_latency_ms |   overlap_ratio |
|:---------|:-----------------|:------------------------|---------------:|-------------------:|-----------------------:|----------------:|
| hybrid   | v1               | True                    |         0.7391 |                nan |                4021.82 |               1 |
| hybrid   | v2               | True                    |         0.8    |                nan |                4021.82 |               1 |
| hybrid   | v3               | False                   |         1      |                nan |                4021.82 |               1 |
| dense    | v1               | True                    |         0.3333 |                nan |                4024.68 |               1 |
| dense    | v2               | True                    |         0.7778 |                nan |                4024.68 |               1 |
| dense    | v3               | True                    |         1      |                nan |                4024.68 |               1 |

### Question 2: Why is the appendix called a "surgical organ"?
| method   | prompt_version   | guardrail_is_grounded   |   faithfulness |   answer_relevancy |   retrieval_latency_ms |   overlap_ratio |
|:---------|:-----------------|:------------------------|---------------:|-------------------:|-----------------------:|----------------:|
| hybrid   | v1               | True                    |         0.8261 |                nan |                4200.61 |            0.83 |
| hybrid   | v2               | False                   |         0.7619 |                nan |                4200.61 |            0.83 |
| hybrid   | v3               | True                    |       nan      |                nan |                4200.61 |            0.83 |
| dense    | v1               | False                   |       nan      |                nan |                4115.44 |            0.83 |
| dense    | v2               | False                   |         0.5217 |                nan |                4115.44 |            0.83 |
| dense    | v3               | True                    |       nan      |                nan |                4115.44 |            0.83 |

### Question 3: Why is the left kidney located higher than the right kidney?
| method   | prompt_version   | guardrail_is_grounded   |   faithfulness |   answer_relevancy |   retrieval_latency_ms |   overlap_ratio |
|:---------|:-----------------|:------------------------|---------------:|-------------------:|-----------------------:|----------------:|
| hybrid   | v1               | True                    |         0      |                nan |                3294.86 |               1 |
| hybrid   | v2               | True                    |         0.2222 |                nan |                3294.86 |               1 |
| hybrid   | v3               | True                    |         0.7273 |                nan |                3294.86 |               1 |
| dense    | v1               | True                    |         0      |                nan |                3654.28 |               1 |
| dense    | v2               | True                    |         0.2727 |                nan |                3654.28 |               1 |
| dense    | v3               | True                    |         0.6667 |                  1 |                3654.28 |               1 |

### Question 4: Why are the renal arteries considered end arteries?
| method   | prompt_version   | guardrail_is_grounded   |   faithfulness |   answer_relevancy |   retrieval_latency_ms |   overlap_ratio |
|:---------|:-----------------|:------------------------|---------------:|-------------------:|-----------------------:|----------------:|
| hybrid   | v1               | True                    |         1      |                nan |                2681.44 |            0.83 |
| hybrid   | v2               | True                    |         1      |                nan |                2681.44 |            0.83 |
| hybrid   | v3               | True                    |       nan      |                nan |                2681.44 |            0.83 |
| dense    | v1               | True                    |         1      |                nan |                3353.21 |            0.83 |
| dense    | v2               | True                    |         0.8125 |                nan |                3353.21 |            0.83 |
| dense    | v3               | True                    |         0.8276 |                nan |                3353.21 |            0.83 |
