# NVIDIA Concurrency Benchmark

Timestamp: 2026-08-20T09:00:38.151239+00:00
Model: `meta/llama-3.1-8b-instruct`
Frozen contexts: 20
Requests per level: 100

> **The Success column below measures NVIDIA's free-tier request quota, not this
> server's capacity.** Read the correction underneath before using any of it.
> Verified 2026-09-02 from `bench_nvidia_concurrency.json`: **every one of the
> 408 failures carries `http_statuses: [429, 429]`** — the provider's rate
> limiter, returned after one retry. Not a timeout, not a server error, not
> queueing. The TTFT and generation columns remain valid, because they were
> measured only on requests that actually succeeded.

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

## Correction (2026-09-02) — this table and `bench_llm_providers.md` do not disagree

Both documents were committed as findings and appear to contradict each other on
the same provider, model and box, two hours apart:

| | `bench_llm_providers.md` 07:04 | this run 09:00 |
|---|---:|---:|
| Requests **per level** | 20 | **100** |
| Levels | 1, 5, 10, 20 | 1, 5, 10, 20, 50, 100 |
| Total NVIDIA requests | 60 | **600** |
| 429s observed | **0** | **826** |
| Success at concurrency 10 | 100% | 16% |

**The variable is total request volume, not concurrency.** The 07:04 run stayed
under the free-tier quota and saw zero 429s; this run blew through it and saw
almost nothing else. Two details in the table above prove concurrency is not
what's being measured:

1. **Levels 10 and 20 both scored exactly 16/100.** Doubling concurrency changed
   the success rate not at all — a capacity ceiling would not behave that way.
2. **Level 1 scored 100/100 on the same 100 requests.** Identical work, one at a
   time, no failures. Only the arrival *rate* differed.

So the success column is a chart of the provider's quota policy. It is not
evidence that this server collapses at 50 concurrent users, and it must not be
cited as a capacity limit.

### What this does and does not license

- **Valid from both runs:** NVIDIA TTFT p50 ~0.8-1.0s, and NVIDIA being
  decisively faster and more reliable than AgentRouter (12.49s TTFT, 72%
  success). The provider switch in `8b12229` is well supported.
- **Not established by either run:** the concurrency at which *this application*
  degrades. Nothing here measured it, because the provider's limiter always
  bound first. `CHAT_RATE_LIMIT=5/minute` and `LLM_MAX_CONCURRENCY=1` are
  therefore reasonable **quota-protection** settings, not capacity-derived ones —
  and they are the reason the app does not currently hand students the
  "could not generate an answer: the openai endpoint is rate limiting requests"
  prose that fills this run's failures.
- **To actually measure app capacity** you need either a paid NVIDIA tier or a
  local/stubbed model endpoint, so the limiter stops being the binding
  constraint. Re-running this script against the free tier will only redraw the
  same quota curve.

### Reproducing the diagnosis

```bash
./.venv/Scripts/python.exe -c "
import json, collections
d = json.load(open('bench_nvidia_concurrency.json'))
for lv in sorted({o['concurrency'] for o in d['raw_observations']}):
    rows = [o for o in d['raw_observations'] if o['concurrency'] == lv]
    bad = [o for o in rows if not o['ok']]
    st = collections.Counter(s for o in bad for s in (o.get('http_statuses') or []))
    print(f'concurrency={lv:3d} ok={len(rows)-len(bad):3d}/{len(rows)} statuses={dict(st)}')
"
```
