# NVIDIA NIM LLM Performance Diagnostic

Timestamp: 2026-09-05T12:44:48.105254+00:00

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
| 1 | 5 | 80.0% | 1.16s | 2.09s | 154 | 72.8 | 0.324/s |
| 2 | 5 | 100.0% | 1.02s | 1.70s | 166 | 80.5 | 0.696/s |
| 5 | 5 | 80.0% | 5.42s | 7.46s | 146 | 27.5 | 0.158/s |

## 3. Concurrency results

Concurrency was run against NVIDIA directly, with 20 queries per level. HTTP errors, timeouts, and provider quota responses are retained in the JSON observations.

## 4. Full RAG latency breakdown

| Stage | Median | p95 | Share of median end-to-end |
|---|---:|---:|---:|
| `session_history_ms` | 45 ms | 525 ms | 0.3% |
| `embedding_ms` | 46 ms | 135 ms | 0.3% |
| `retrieval_wrapper_ms` | 1160 ms | 3796 ms | 8.3% |
| `hydration_ms` | 10 ms | 163 ms | 0.1% |
| `context_build_ms` | 0 ms | 0 ms | 0.0% |
| `llm_generation_ms` | 11346 ms | 27148 ms | 80.9% |
| `persist_ms` | 215 ms | 2296 ms | 1.5% |
| `total_end_to_end_ms` | 14024 ms | 31839 ms | - |

### Retrieval sub-stages

| Stage | Median | p95 |
|---|---:|---:|
| `bm25` | 2 ms | 10 ms |
| `query_processing` | 3 ms | 137 ms |
| `rerank` | 903 ms | 1770 ms |
| `search` | 1155 ms | 3727 ms |
| `vector` | 101 ms | 997 ms |

## 5. Streaming analysis

`connection_to_headers_s` is measured separately from `ttft_after_headers_s`; TTFT excludes the request-to-headers interval. `reasoning_content` is counted from raw SSE deltas when NVIDIA returns it. The app's `LLMService.stream_answer` yields only answer text, so reasoning deltas are not shown to students.

## 6. Error and timeout analysis

| Concurrency | Success | Timeout rate | HTTP statuses | Errors |
|---:|---:|---:|---|---|
| 1 | 4/5 | 0.0% | 200 x5 | unknown x1 |
| 2 | 5/5 | 0.0% | 200 x5 | none |
| 5 | 4/5 | 0.0% | 200 x5 | unknown x1 |

Full RAG answer success rate: **95.0%**. Queue-status prose is not counted as an isolated answer.

## 7. Bottleneck identification

The median NVIDIA generation share of end-to-end RAG latency was **80.9%**; retrieval wrapper share was **8.3%**. The database-scoped stages (session/history, hydration, and persistence) together consumed **1.9%** of median end-to-end latency.

## 8. Theoretical <10s calculation

Measured median non-LLM stages: **1.48s**; measured median NVIDIA generation: **11.35s**; theoretical best-case with those healthy medians retained: **12.82s** before network/client variance.
The target is realistically achievable only if this best-case value is below 10s and p95 is separately controlled; median alone does not guarantee the target for every request.

## 9. Recommended optimizations ranked by expected impact

1. Reduce NVIDIA answer-generation time first: it measured 80.9% of median end-to-end latency, and the measured median was 11.35s.
2. Eliminate avoidable provider failures before raising concurrency: the isolated run reached 100.0% at its best level but fell to 0.0% at concurrency 10 because of HTTP 429 responses.
3. Investigate persistence and database variance next: the measured persistence median was 0.21s and p95 was 2.30s.
4. Profile reranking only after the NVIDIA path: reranking measured 0.90s median, materially smaller than NVIDIA generation but still the largest CPU retrieval sub-stage.

## Blunt conclusion

**NVIDIA LLM is YES the main median-latency bottleneck** in this run. Its measured contribution is approximately **80.9%** of median end-to-end latency.
