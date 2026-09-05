# NVIDIA Concurrency Benchmark

Timestamp: 2026-09-05T05:07:46.456275+00:00
Model: `nvidia/nemotron-3-super-120b-a12b`
Frozen contexts: 20
Requests per level: 10

| Concurrency | Success | TTFT p50 | TTFT p95 | Gen p50 | Gen p95 | Total p50 | Total p95 | Throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.0% | 7.10s | 16.03s | 10.42s | 17.31s | 12.15s | 18.23s | 0.091/s |
| 5 | 90.0% | 60.00s | 72.21s | 60.00s | 85.28s | 60.91s | 86.74s | 0.066/s |
| 10 | 100.0% | 54.23s | 62.42s | 57.89s | 64.84s | 58.90s | 65.74s | 0.145/s |

Generation latency is measured from request initiation until the final streamed text chunk.
Total answer latency adds the frozen context's original local retrieval time once; retrieval was not rerun per provider request.
