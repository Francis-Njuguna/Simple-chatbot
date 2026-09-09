# NVIDIA Prompt Optimization A/B

This report compares the same representative queries through the real retrieval, NVIDIA streaming, and persistence path. Quality values are automated proxies, not a substitute for human review.

## Configuration

- Model: `nvidia/nemotron-3-super-120b-a12b`
- NVIDIA base URL: `https://integrate.api.nvidia.com/v1`
- Requests per variant: `20`
- Concurrency: `1` (sequential, to avoid provider throttling)

## Before / after

| Metric | Legacy | Compact | Change |
|---|---:|---:|---:|
| End-to-end median | 15.19s | 8.51s | -44.0% |
| NVIDIA generation median | 11.74s | 7.72s | -34.2% |
| TTFT median | 6.67s | 4.57s | -31.5% |
| Retrieval median | 1.38s | 0.01s | -99.6% |
| Persistence median | 0.68s | 1.16s | +69.7% |
| Input tokens median | 2291 | 1141 | -50.2% |
| Output tokens median | n/a | n/a | n/a |
| Answer chars median | 1613 | 864 | -46.4% |
| Success rate | 90.0% | 95.0% | +5.0% |
| HTTP 429 rate | 0.0% | 0.0% | +0.0% |

## p95

- Legacy end-to-end p95: **38.93s**; compact: **31.58s**.
- Legacy NVIDIA p95: **28.54s**; compact: **30.66s**.

## Prompt composition

- Legacy system prompt: **1326 tokens**; compact system prompt: **220 tokens**.
- Legacy full prompt: **2291 tokens**; compact full prompt: **1141 tokens**.

## Quality proxies

| Proxy | Legacy | Compact |
|---|---:|---:|
| relevance | 50.0% | 50.0% |
| groundedness | 45.0% | 60.0% |
| completeness_proxy | 90.0% | 95.0% |
| short_query_handled | 0.0% | 0.0% |

## Interpretation

Prompt reduction is retained only if latency improves without a material drop in the quality proxies. `output_tokens` and `reasoning_chars` are provider-reported when NVIDIA includes them in stream usage; otherwise the JSON records null/zero rather than estimating them.
