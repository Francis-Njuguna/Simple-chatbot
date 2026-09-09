# NVIDIA NIM LLM Performance Diagnostic

Timestamp: 2026-09-05T09:24:20.724447+00:00

## 1. Current NVIDIA configuration

| Setting | Value |
|---|---|
| `provider` | `NVIDIA NIM (OpenAI-compatible transport)` |
| `model` | `nvidia/nemotron-3-super-120b-a12b` |
| `api_base` | `https://integrate.api.nvidia.com/v1` |
| `llm_timeout_s` | `25` |
| `first_token_timeout_s` | `30.0` |
| `stream_stall_timeout_s` | `30.0` |
| `max_output_tokens` | `2048` |
| `temperature` | `0.0` |
| `max_retries` | `1` |
| `llm_max_concurrency` | `1` |
| `llm_queue_timeout_s` | `60.0` |
| `http_max_connections` | `1000` |
| `http_max_keepalive` | `100` |
| `streaming_path` | `LLMService.stream_answer / NVIDIA SSE` |
| `embedding_provider` | `sentence-transformers` |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` |
| `embedding_device` | `cpu` |
| `rerank_enabled` | `True` |
| `rerank_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `rerank_quantize` | `True` |
| `rerank_shortlist` | `8` |
| `rerank_query_forms` | `2` |
| `top_k_retrieval` | `5` |
| `db_pool_size` | `5` |
| `db_max_overflow` | `10` |
| `db_pool_timeout_s` | `30` |
| `chroma_mode` | `auto` |
| `chroma_persist_dir` | `./data/chroma` |

## 2. Isolated LLM benchmark

Fixed context; 20 representative queries; direct NVIDIA SSE; no retrieval.

| Concurrency | Requests | Success | TTFT after headers p50 | Total generation p50 | Output tokens p50 | Tok/s p50 | Throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 100.0% | 1.34s | 2.23s | 162 | 73.6 | 0.246/s |
| 2 | 20 | 95.0% | 1.03s | 2.35s | 141 | 74.4 | 0.543/s |
| 5 | 20 | 100.0% | 1.26s | 2.55s | 145 | 64.6 | 0.538/s |
| 10 | 20 | 70.0% | 0.99s | 2.23s | 156 | 83.4 | 1.513/s |

## 3. Concurrency results

Concurrency was run against NVIDIA directly, with 20 queries per level. HTTP errors, timeouts, and provider quota responses are retained in the JSON observations.

## 4. Full RAG latency breakdown

| Stage | Median | p95 | Share of median end-to-end |
|---|---:|---:|---:|
| `session_history_ms` | 43 ms | 2571 ms | 0.3% |
| `embedding_ms` | 51 ms | 98 ms | 0.4% |
| `retrieval_wrapper_ms` | 959 ms | 4395 ms | 7.1% |
| `hydration_ms` | 10 ms | 1571 ms | 0.1% |
| `context_build_ms` | 0 ms | 0 ms | 0.0% |
| `llm_generation_ms` | 10748 ms | 36651 ms | 79.8% |
| `persist_ms` | 1370 ms | 3007 ms | 10.2% |
| `total_end_to_end_ms` | 13473 ms | 39703 ms | - |

### Retrieval sub-stages

| Stage | Median | p95 |
|---|---:|---:|
| `bm25` | 2 ms | 6 ms |
| `query_processing` | 3 ms | 154 ms |
| `rerank` | 785 ms | 951 ms |
| `search` | 955 ms | 1590 ms |
| `vector` | 85 ms | 538 ms |

## 5. Streaming analysis

`connection_to_headers_s` is measured separately from `ttft_after_headers_s`; TTFT excludes the request-to-headers interval. `reasoning_content` is counted from raw SSE deltas when NVIDIA returns it. The app's `LLMService.stream_answer` yields only answer text, so reasoning deltas are not shown to students.

## 6. Error and timeout analysis

| Concurrency | Success | Timeout rate | HTTP statuses | Errors |
|---:|---:|---:|---|---|
| 1 | 20/20 | 0.0% | 200 x20 | none |
| 2 | 19/20 | 0.0% | 200 x20 | unknown x1 |
| 5 | 20/20 | 0.0% | 200 x20 | none |
| 10 | 14/20 | 0.0% | 200 x14, 429 x6 | HTTP 429: {"status":429,"title":"Too Many Requests"} x6 |

Full RAG answer success rate: **90.0%**. Queue-status prose is not counted as an isolated answer.

## 7. Bottleneck identification

The median NVIDIA generation share of end-to-end RAG latency was **79.8%**; retrieval wrapper share was **7.1%**. The database-scoped stages (session/history, hydration, and persistence) together consumed **10.6%** of median end-to-end latency.

## 8. Theoretical <10s calculation

Measured median non-LLM stages: **2.43s**; measured median NVIDIA generation: **10.75s**; theoretical best-case with those healthy medians retained: **13.18s** before network/client variance.
The target is realistically achievable only if this best-case value is below 10s and p95 is separately controlled; median alone does not guarantee the target for every request.

## 9. Recommended optimizations ranked by expected impact

1. Reduce NVIDIA answer-generation time first: it measured 79.8% of median end-to-end latency, and the measured median was 10.75s.
2. Eliminate avoidable provider failures before raising concurrency: the isolated run reached 100.0% at its best level but fell to 70.0% at concurrency 10 because of HTTP 429 responses.
3. Investigate persistence and database variance next: the measured persistence median was 1.37s and p95 was 3.01s.
4. Profile reranking only after the NVIDIA path: reranking measured 0.78s median, materially smaller than NVIDIA generation but still the largest CPU retrieval sub-stage.

## Blunt conclusion

**NVIDIA LLM is YES the main median-latency bottleneck** in this run. Its measured contribution is approximately **79.8%** of median end-to-end latency.
