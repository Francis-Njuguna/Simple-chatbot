# LLM Provider Benchmark

Timestamp: 2026-08-20T07:04:41.043662+00:00
Git commit: `e16517cc6463cceca7e828180142395a5d08fcd0`
Python: 3.14.6 (tags/v3.14.6:c63aec6, Jun 10 2026, 10:26:10) [MSC v.1944 64 bit (AMD64)]
Questions: 20; trials: 5

| Metric | NVIDIA | AgentRouter |
|---|---:|---:|
| Model | meta/llama-3.1-8b-instruct | claude-opus-5 |
| Sequential requests | 100 | 100 |
| TTFT p50 (s) | 1.00 | 12.49 |
| TTFT p95 (s) | 1.16 | 23.06 |
| TTFT p99 (s) | 1.21 | 27.64 |
| LLM total p50 (s) | 3.42 | 22.19 |
| LLM total p95 (s) | 9.92 | 38.23 |
| LLM total p99 (s) | 11.76 | 43.76 |
| Success rate | 100.0% | 72.0% |
| Quality score | 56.60 | 31.95 |
| Overall score | 89.15 | 32.57 |

## Concurrency

| Provider | Users | Success | TTFT p50 | TTFT p95 | Total p50 | Total p95 | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| nvidia | 1 | 100.0% | 0.98s | 1.15s | 3.04s | 4.20s | 0.317 answers/s |
| nvidia | 5 | 100.0% | 0.83s | 1.21s | 3.94s | 6.34s | 0.841 answers/s |
| nvidia | 10 | 100.0% | 0.76s | 0.89s | 3.83s | 12.88s | 1.547 answers/s |
| agentrouter | 1 | 80.0% | 14.64s | 26.92s | 25.11s | 46.68s | 0.028 answers/s |
| agentrouter | 5 | 60.0% | 12.40s | 22.16s | 24.88s | 31.87s | 0.106 answers/s |
| agentrouter | 10 | 75.0% | 12.52s | 26.15s | 22.76s | 35.46s | 0.231 answers/s |

## Winner

Single-user: **nvidia**
Concurrent: **nvidia**
Quality: **nvidia**
Overall: **nvidia**

Recommendation: Keep NVIDIA as the primary provider; retain configurable AgentRouter fallback.

Quality metrics are repeatable automated proxies (term coverage, lexical grounding, citations, completeness and URL hallucination), not a substitute for human review of the included raw answers.
