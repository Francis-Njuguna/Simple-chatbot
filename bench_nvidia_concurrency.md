# NVIDIA Concurrency Benchmark

Timestamp: 2026-08-20T09:00:38.151239+00:00
Model: `meta/llama-3.1-8b-instruct`
Frozen contexts: 20
Requests per level: 100

| Concurrency | Success | TTFT p50 | TTFT p95 | Gen p50 | Gen p95 | Total p50 | Total p95 | Throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.0% | 0.79s | 1.01s | 3.31s | 10.75s | 4.65s | 11.70s | 0.244/s |
| 5 | 60.0% | 0.81s | 1.76s | 3.34s | 11.91s | 4.58s | 13.21s | 0.849/s |
| 10 | 16.0% | 1.37s | 1.59s | 5.10s | 5.72s | 6.06s | 6.67s | 0.820/s |
| 20 | 16.0% | 0.84s | 1.02s | 8.07s | 8.62s | 9.30s | 10.63s | 1.098/s |
| 50 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | 0.000/s |
| 100 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | 0.000/s |

Generation latency is measured from request initiation until the final streamed text chunk.
Total answer latency adds the frozen context's original local retrieval time once; retrieval was not rerun per provider request.
