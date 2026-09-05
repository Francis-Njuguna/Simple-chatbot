# The latency is host-level, not the cross-encoder

**Conclusion: the slow requests are not a reranker problem, and no amount of
model optimisation will fix them.** Every stage of the pipeline is
intermittently 10-100x slow, including stages that contain no model, no I/O and
no network. The cross-encoder is merely the largest single block of CPU work in
the request, so it shows the biggest absolute number when the process is starved.

This document exists because the optimisation plan was built on one trace
(request `48a1655868a3`, 2026-08-17 15:16) that attributed 8.7s to rerank and
concluded "this CPU is 4-15x slower than the code's assumption". That conclusion
was wrong. What follows is the evidence, because acting on the original reading
would have meant swapping in a 2-layer model and re-calibrating the
`rerank_min_score=-8.0` off-topic gate — real, permanent quality risk taken on
to fix something that was never the cause.

## 2026-08-18 — what closing Brave/Codex/Docker did and did not show

Brave, Codex and the Docker containers were closed (Postgres left running) and
the pipeline re-measured. **The result does not support "the other applications
were the cause", and the reason is worth recording carefully.**

`scripts/_diag_worker_thread.py`, same 32 pairs (shortlist 16 x 2 forms), int8,
4 torch threads:

| When | Other apps running | 32 pairs |
|---|---|---|
| Contended era (measurements in this doc) | Brave + Codex + Docker | **620-740ms** |
| 2026-08-18 | none | **708-805ms** |

**Identical, if anything marginally slower with the box empty.** So the
in-process cost never depended on what else was running. Closing the
applications changed nothing measurable here.

The 12.3x gap (271ms/pair logged vs ~22ms/pair measured) is therefore **not**
"busy box vs quiet box". It is **in-process vs inside the uvicorn server** — the
same gap this document reported before, still unexplained, and not narrowed by
quieting the host.

Full in-process pipeline for reference, `diag_latency.py --concurrency 1,1,1
--no-cache`, four cold requests (cache cleared, so not cache hits):

| Stage | In-server 2026-08-17 | In-process 2026-08-18 |
|---|---|---|
| session_history | 144ms | 54-267ms |
| embedding | 629ms | 25-62ms |
| **retrieval** | **9,480ms** | **732-943ms** |
| └ rerank | 8,669ms | 686-866ms |
| hydration | 209ms | 4-6ms |
| **pre-LLM subtotal** | **10,182ms** | **~965ms** |

In-process retrieval also has a **1.29x** spread across the four, versus the
140x-11,228x spreads that characterise the in-server logs. Whatever produces
those spreads is absent from this harness — which is precisely why this harness
cannot settle the question.

**`--reload` and the uvicorn process are now the leading suspects, promoted
rather than demoted by this run.** Every fast measurement ever taken of this
pipeline has been in-process without uvicorn; every slow one has been inside the
server. Memory pressure is correspondingly demoted: it should have degraded the
in-process run too, and did not.

> **SUPERSEDED 2026-09-02 — the decisive test was run the same evening this was
> written, and it clears `--reload`.** See "2026-09-02 — the decisive test, found
> in the logs" below. The paragraph immediately following was true when written
> and is false now; it is kept because acting on a stale "not yet run" is exactly
> how the reranker got blamed the first time.

**The decisive test has still not been run.** It requires uvicorn, started
without `--reload`, on the now-quiet box, reading the real `[timing:chat]` line
— see "Reproducing" below. `diag_latency.py` is not a substitute for it; it was
used as one here, and that was a mistake.

Two side findings from the same run, both worth keeping:

1. The **first** request in a fresh process measured `hydration=2,337ms`, worse
   than anything in the contended logs; the next three measured 4ms, 4ms, 6ms.
   First-request-in-process artefact, not a cost. Always use `1,1,1`, never a
   single sample.
2. `llm` measured 12,850 / 15,808 / 43,906 / 17,055 ms — a **3.4x spread** while
   local stages held to 1.29x, so this variance is the gateway's, not the host's.
   The 43,906ms request is the retry path working as designed (25s timeout +
   ~0.5s backoff + ~18s successful retry, inside the 50.5s bound that replaced
   ~91s) and it **succeeded** — but agentrouter.org failed to answer within 25s
   on **1 of 4** requests. At 93-97.5% of request time the LLM is both the
   latency problem and a reliability one.

## The single most diagnostic line in the log

```
2026-07-22 08:17:03 [timing:chat] session_history=293ms embedding=2291ms
    retrieval=4763ms context_build=1848ms llm=2128ms persist=158ms TOTAL=11485ms
```

`context_build` assembles the prompt string from already-fetched chunks. Pure
in-memory string work: no torch, no database, no network, no thread dispatch. It
measures **0ms** in almost every other request in the log. Here it took
**1,848ms**, in a request where embedding and retrieval were also elevated.

There is no code-level explanation for that. A stage with no I/O and no model
takes 1.8 seconds only if the process was descheduled or was waiting on memory
page-in. Once you accept that, the rest of the log reads differently.

## The evidence

**1. Every stage swings wildly, including non-torch ones.** Extremes observed:

| Stage | What it does | Fast | Slow | Spread |
|---|---|---|---|---|
| `context_build` | string formatting | 0ms | 1,848ms | ∞ |
| `hydration` | fetch chunk text by id | 5ms | 1,202ms | 240x |
| `embedding` | 1 short query, MiniLM | 95ms | 4,452ms | 47x |
| `session_history` | one Postgres SELECT | 10ms | 14,953ms | 1,495x |
| `persist` | Postgres INSERTs | 23ms | 7,652ms | 333x |
| `retrieval` | vector+bm25+rerank | 10ms | 112,284ms | 11,228x |

`session_history` and `persist` are Postgres round-trips with no torch anywhere
near them. They spike into multiple seconds. **A cause that hits a Postgres
SELECT cannot be a cross-encoder problem.**

**2. Same warm process, 78 seconds apart, same work:**

```
2026-08-17 15:11:40  embedding=95ms    hydration=1202ms  session_history=131ms
2026-08-17 15:10:06  embedding=4452ms  hydration=11ms    session_history=699ms
```

`embedding=95ms` proves the model *can* run at full speed in that exact process.
When the identical call takes 4,452ms a minute later, nothing about the model,
the batch, the threads or the passages changed. The CPU was taken.

**3. Startup — which contains no inference at all — degraded the same way:**

| Server start | → `cross-encoder reranker ready` | Elapsed |
|---|---|---|
| 2026-08-10 15:13:42 | 15:13:57 | **15s** |
| 2026-08-10 13:50:22 | 13:50:49 | **27s** |
| 2026-08-17 13:17:58 | 13:18:58 | **60s** |
| 2026-08-17 20:23:26 | 20:26:39 | **3m13s** |

Startup is disk reads plus huggingface.co revalidation. It got 2-13x slower on
the same box with the same warm HF cache.

**4. Retrieval degraded progressively over a week on identical work** — `pool=41`
constant, so the same 41 candidates and the same pair count throughout:

| Date | `retrieval` | `variants` |
|---|---|---|
| Aug 10 | 1518, 1705, 1978 ms | 4, 2, 3 |
| Aug 11 (isolated requests) | 2793, 3030, 3167, 5040 ms | 3 |
| Aug 17 | 5353, 7604, 9390 ms | 5, 3, 4 |

Note `variants=1` took **4,738ms** while `variants=4` took **1,518ms** — the
query-variant count does not drive the time, which rules out the expansion
fan-out as the cause.

**5. Two distinct regimes, do not confuse them.** The 2026-08-11 12:37-12:42
block shows dozens of requests all completing within the same second with
`retrieval=112,284ms`. That is a **load test** — ~40 concurrent requests
queueing on 4 cores — and its numbers are pure queueing, not per-request cost.
It is unrelated to the isolated slow requests on Aug 17, which arrived hours
apart (09:02, 11:29, 12:54, 13:29, 14:24, 14:41, 15:09, 15:11, 15:16).

## What this rules out

Six model-level hypotheses were measured and refuted before the host-level
pattern was spotted. Recorded so they are not re-run:

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | Concurrent requests contending | Timestamps of the 9 slow requests | Hours apart, isolated. **Refuted** |
| 2 | Cold model / first-request graph setup | Startup 14:09:02, warm 14:09:57, slow request 15:16 | Warm by an hour. **Refuted** |
| 3 | Varying tensor shapes defeating oneDNN's kernel cache | `scripts/_diag_shape_cache.py` | Fixed 193ms vs varying 207ms = **1.07x**. **Refuted.** Bonus: forcing `padding="max_length", max_length=256` cost 1002ms — **5.19x worse**. Do not "optimise" by padding. |
| 4 | torch losing intra-op parallelism in anyio worker threads | `scripts/_diag_worker_thread.py`, 32 pairs | main 620ms, worker 739ms, worker+`set_num_threads` 572ms = **1.19x**. **Refuted** |
| 5 | Passages longer than the diagnostics assumed | Whole KB: 11,058 chars / 20 articles, largest 1,731, `CHUNK_SIZE=500` | Real chunks ≤500 chars; test passages were representative. **Refuted** |
| 6 | Wrong pair count (all 41 fused scored, not the shortlist) | `retriever.py:565` `selected_ids = fused_ids[:settings.rerank_shortlist]` | Shortlist caps it. **Refuted** |
| 7 | torch thread count | 1 vs 2 vs 4 threads | 1.27x spread. Real, but an order of magnitude too small. **Insufficient** |
| 8 | Batch size / one `predict()` vs two | `scripts/_verify_score_multi.py` | Sequential 371.5ms vs batched 385.4ms = **0.96x**. **Insufficient** |

In the server's exact execution context — anyio worker thread, int8, 4 torch
threads, 32 pairs — the work measures **0.62-0.74s** in-process, and the
in-server warmup `score_multi` of 16 pairs completed in **≤1s**. Production
logged **8.669s** for the same thing.

**Therefore Phase 2 of the plan (ONNX backend / 2-layer model) should not
proceed as a latency fix.** It targets a 0.7s stage, would buy at most ~3x of
that 0.7s, and the 2-layer swap requires re-calibrating the absolute
`rerank_min_score=-8.0` gate that is the only mechanism declining off-topic
questions. On a quiet box retrieval is already 1.5s and often far less; there is
no 8s of cross-encoder to reclaim.

## Where the latency actually is: the LLM gateway

`/chat/stream` records `llm_first_token`, measured from the moment the LLM call
starts. Six streaming requests exist in the log, all 2026-08-12:

| Request | `retrieval` | `llm_first_token` | tail | Outcome |
|---|---|---|---|---|
| 14:45:49 | 167ms *(cache hit)* | **94,930ms** | 21ms | **FAILED — error shown** |
| 14:43:09 | 17,427ms | **95,560ms** | 263ms | **FAILED — HTTP 429** |
| 09:03:47 | 7,136ms | **90,897ms** | 8,839ms | succeeded on 3rd attempt |
| 13:29:19 | 13,808ms | 4,706ms | 10,302ms | succeeded |
| 11:30:03 | 14,705ms | 15,904ms | 8,077ms | succeeded |
| 14:24:38 | 8,531ms | 11,593ms | 4,957ms | succeeded |

**The ~90s cluster is a timeout cascade, and it is fully explained by config:**

```
config.py:113   llm_timeout: int = Field(default=30, alias="LLM_TIMEOUT")
config.py:114   llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")
```

30s timeout x (1 attempt + 2 retries) + ~1.2s of backoff = **~91s**. At the time
neither variable was set in `.env`, so these defaults were live (both are now
pinned explicitly — see the fix below). The log shows the cascade directly —
e.g. request `a1837ba48b96`: retry at 14:44:45, retry at 14:45:16, give up at
14:45:48, `ERROR ... LLM streaming failed`.

**What the attempts are waiting on: nothing.** An earlier version of this
section blamed HTTP 429 and recommended honouring `Retry-After` to "fail fast in
~2s". The log does not support that, and the correction matters because it points
at a different fix:

- Every retry sleep logged by the SDK, across the whole file, is **0.37-0.99s**
  (`Retrying request to /chat/completions in 0.42 seconds`). That is plain
  exponential backoff from `INITIAL_RETRY_DELAY=0.5`. This gateway never sends
  `Retry-After`, so honouring it would change nothing.
- Request `a1837ba48b96` has **no httpx status line at all** — not a 429, not
  anything. All three attempts simply expired.
- Request `f46c5ed2ed9c` did get a 429, but it arrived **~30s after the request
  was sent**, i.e. at the timeout boundary. So "fail fast in ~2s" was never
  available: the gateway takes ~30s to say anything, including *no*.
- The 429s cluster in the **2026-08-11 load-test window** (12:37-12:43, ~40
  concurrent requests on one key), which is a self-inflicted and separate
  problem from the Aug 12 single-request cascades.

So the cost is the per-attempt timeout times the attempt count, full stop —
`30 × 3`. The 21ms / 263ms tails are `stream_answer`'s
`yield self._error_message(exc)`, not fast answers.

**Fixed (2026-08-17)** in `config.py`, `rag/llm.py`, `rag/sse_repair.py`, `.env`,
with `tests/test_llm_timeout_budget.py` holding the bounds:

| | Before | After |
|---|---|---|
| transport worst case | ~91s | **50.5s** (`LLM_TIMEOUT=25` × 2 attempts) |
| what a student waits before seeing text | ~91s | **30s** (`LLM_FIRST_TOKEN_TIMEOUT`) |
| unreachable gateway | ~91s | **~10.5s** (`LLM_CONNECT_TIMEOUT=5`, split out) |

`LLM_TIMEOUT` is 25 and not lower because the one attempt that *did* succeed took
~22s (09:03:15 → 09:03:37); a shorter budget converts that success into a
failure. `llm_transport_worst_case_seconds` now computes the product and prints
it at startup, so the number is asserted rather than emergent — the ~91s was
never a decision anyone made.

Consequences for the "first token <3s" target:

- **This fix bounds the failure path. It saves 0ms on a healthy request.** The
  three ~91s requests were cascades, not slow generation; the four that worked
  took 4.7-15.9s to first token. Capping a cascade cannot make a working request
  faster.
- What remains on the healthy path is the model itself, and it was measured
  rather than assumed — `scripts/probe_llm_latency.py`, 3 trials per model,
  round-robin so a host stall cannot be charged to one model, 2026-08-18:

  | Model | first *visible* text | full answer | usable |
  |---|---|---|---|
  | `claude-opus-5` | **6.96s** (6.37 / 6.96 / 7.04) | **14.45s** | 3/3 |
  | `claude-haiku-4-5` | — | — | **0/3 — HTTP 403** |
  | `claude-sonnet-4-5` | — | — | **0/3 — HTTP 403** |

  Two findings, and the second is the one that constrains the plan:

  1. `claude-opus-5` is a reasoning model. It streams `reasoning_content` deltas
     carrying no answer text, which `stream_answer` correctly filters out, so the
     entire reasoning phase lands inside time-to-first-token: **~7s before a
     student sees one character, on a quiet box with no retrieval contention.**
     `first token <3s` is therefore unreachable with this model — a model choice,
     not a tuning problem.
  2. **The model swap is blocked.** Both faster models return
     `403 该令牌无权访问模型 <model>` — "this token is not authorised for model
     X". The key is valid (opus-5 answers on it), so this is a plan limit.
     Getting under 3s needs an upgraded AgentRouter key or the `gemini` provider
     (`GEMINI_API_KEY` / `gemini-2.0-flash` are already configured).

  Note the probe measures *visible* text deliberately. An earlier version scored
  a 403 as the fastest model at 0.99s with a perfect 3/3, because
  `stream_answer` never raises — it yields the error as prose. Any per-model
  timing table that cannot tell an answer from an error message is worthless.
- Streaming the widget is still correct and still worth having, but it cannot
  deliver <3s while first token is gated behind the reasoning phase.
- `LLM_MAX_TOKENS=2048` is not currently a constraint: measured answers are
  1,825-2,036 characters, roughly 500 tokens. Capping it near ~700 bounds the
  worst case without touching any answer produced today.
- Caveat on the 167ms row: that request logged `Retrieval cache HIT`, so it did
  no retrieval work. It is **not** evidence that cold retrieval costs 167ms —
  cold retrieval on a quiet box is ~1.5s per the Aug 10 data. It does show a
  retrieval cache exists and is effective for repeat questions.
- Note the cascades coexist with host starvation in the same requests
  (`session_history=11,383ms`, `embedding=7,227ms`), so both problems are real
  and independent. Fixing the box will not fix a 429 cascade, and fixing the
  gateway will not fix 1,848ms of string formatting.

## 2026-09-02 — the decisive test, found in the logs

**It was run on 2026-08-18 at 16:44, hours after this document declared it "still
not run", and nobody folded the result back in.** `logs/uvicorn_test1.log` is a
real uvicorn server on port 8001 with **no reloader** (`--reload` prints
`Started reloader process ... WatchFiles`; that line is absent) and it carries
four genuine `[timing:chat]` / `[timing:chat_stream]` lines.

| When | idle before | session_history | embedding | retrieval | hydration | context_build |
|---|---:|---:|---:|---:|---:|---:|
| 08-18 20:57:29 | 253 min | 4,595ms | 5,028ms | 5,048ms | 557ms | 27ms |
| 08-19 12:21:17 | 924 min | 2,096ms | 4,538ms | 5,671ms | 276ms | 9ms |
| 08-19 12:22:44 | **1.4 min** | 764ms | **88ms** | **1,142ms** | 196ms | 7ms |
| 08-19 12:24:25 | **1.7 min** | **117ms** | **102ms** | 4,911ms | 26ms | 0ms |

**`--reload` is refuted.** The server had no reloader and still produced
`embedding=5,028ms` and `session_history=4,595ms` — the same class of stall this
document catalogues. The leading suspect from the section above is dead.

**What the four rows actually track is idle time, not concurrency or config.**
The two requests that followed a long idle gap (253 min, 924 min) show embedding
at 4.5-5.0s; the two that arrived 1-2 minutes behind another request show it at
88ms and 102ms — a **50x** difference on identical work in one warm process, with
the fast pair sandwiched between the slow ones so ordering and warmup cannot
explain it. That is the signature of the process being **paged out while idle**
and faulted back in on the next request.

**This promotes memory pressure / paging back to leading candidate** — the
opposite of what the 2026-08-18 in-process run concluded. It also explains why
that run saw nothing: `diag_latency.py` fires its requests back-to-back, so the
process never sits idle long enough to be evicted. The harness could not have
reproduced this bug.

Consistent with it: `retrieval=4,911ms` in the last row while `embedding=102ms`
and `context_build=0ms`. Under the paging story the residency of *one* stage's
pages need not match another's; under a CPU-contention story every stage in the
same request should degrade together, and here they plainly do not.

**Still not established:** the paging hypothesis is now well supported but not
directly confirmed — no one has watched this process's working set across an idle
gap. That measurement (working set immediately before and after a post-idle
request, plus system commit vs physical) is the remaining step, and it is
cheap. Until then, treat "idle-time paging" as the best explanation rather than a
finding.

**Consequence for the reranker, unchanged and now firmer:** the stalls hit
`session_history` and `embedding` — a Postgres SELECT and a 384-dim encode —
just as hard as they hit rerank. Nothing here is a cross-encoder problem, and
Phase 2 (ONNX / 2-layer model) still should not proceed as a latency fix.



The evidence establishes *that* the host stalls the whole process. It does not
yet establish *why*. Candidates, and how to tell them apart:

- **Memory pressure / paging (leading candidate).** Explains every symptom at
  once: slow startup (disk), variable embedding (weight page-in), variable
  Postgres stages (disk), 1.8s string formatting (page fault), and the
  progressive week-over-week decay. This box has 4 cores and the log shows
  ~20 server restarts on Aug 10 alone; leftover Python processes each holding
  torch plus two models would accumulate. **Test:** `Get-Process python*` for
  count and working set, and total committed memory vs physical.
- **CPU contention from another process** (Defender scan, Search Indexer,
  Windows Update). **Test:** sample CPU by process during a slow request.
- **`--reload`.** ~~`README.md:151` documents the start command with `--reload`,
  and every in-process measurement above was taken without it.~~ **REFUTED
  2026-09-02** — `logs/uvicorn_test1.log` is a no-reloader server that stalled
  anyway (see the 2026-09-02 section). Kept only so it is not re-proposed.
- **Thermal throttling / disk near full.** Least likely to produce 1,495x
  spreads on a single SELECT, but cheap to check.

Note `show_progress_bar` is no longer a candidate for the general case:
`embeddings.py:167` already passed `show_progress_bar=False` and that stage is
slow too. It is now explicitly `False` in `score()` and `score_multi()` anyway.

Also note `timings["rerank_ms"]` brackets `await anyio.to_thread.run_sync(...)`,
so it includes dispatch + compute + *time for the event loop to resume this
coroutine*. Under host starvation that resume delay lands inside the number,
which is part of why rerank absorbs the blame.

## Reproducing

```bash
# Warm in-process cost in the server's exact context (expect ~0.6-0.7s / 32 pairs):
./.venv/Scripts/python.exe -u scripts/_diag_worker_thread.py
./.venv/Scripts/python.exe -u scripts/_diag_shape_cache.py
./.venv/Scripts/python.exe -u scripts/_verify_score_multi.py

# The real gate — the in-server number is the only one that counts.
# Start WITHOUT --reload, and check the box is otherwise quiet first:
./.venv/Scripts/python.exe -u -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"message":"How do I download and install SMOWL for my exam?"}'
# then read the [timing:chat] line in logs/app.log
```

Interpretation: if `retrieval` comes back ~1.5s on a quiet box, the host
hypothesis is confirmed and the latency work belongs on the LLM stage and
streaming, not the reranker.

**Still not run as of 2026-08-18.** It was substituted with
`scripts/diag_latency.py`, which measures the same stages in-process and came
back at 732-943ms — but in-process was *always* fast, so that proves nothing
about the server. Only the uvicorn measurement above can separate `--reload` and
the server process from the host. `diag_latency.py` remains the right tool for
attributing stages *within* a process:

```bash
# Four independent cold requests with the retrieval cache cleared between them.
# Use 1,1,1 rather than a single 1: one sample cannot distinguish a real
# regression from a first-request artefact (it reported hydration=2,337ms).
./.venv/Scripts/python.exe -u scripts/diag_latency.py --concurrency 1,1,1 --no-cache
```

## Separately actionable, found while investigating

**`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.** The 3m13s startup on
2026-08-17 was overwhelmingly huggingface.co `HEAD`/`GET` revalidation
round-trips; the weights themselves load from cache in under a second. Beyond
the ~2 minutes, this is a **production availability risk**: the app currently
cannot boot if huggingface.co is unreachable.

## Two open correctness items — both CLOSED 2026-09-02

1. **`score_multi` is not numerically identical to the sequential path it
   replaced.** `_verify_score_multi.py` measured max abs difference **2.527e-01**
   against a 1e-3 tolerance. Almost certainly int8 dynamic quantization computing
   activation scales per batch, so 16-scored-together != 8+8-scored-separately.
   Not cosmetic: `rerank_min_score=-8.0` is an **absolute** comparison on these
   logits and is what declines off-topic questions, so a 0.25 drift can flip a
   decline. ~~**The fp32 control run has not been done**~~ — **DONE 2026-09-02,
   and it clears the reshape:**

   | Config | max abs difference | verdict |
   |---|---:|---|
   | `RERANK_QUANTIZE=false` (fp32) | **9.537e-07** | PASS — batching is exact |
   | `RERANK_QUANTIZE=true` (int8) | **2.527e-01** | drift is the quantization |

   So the reshape in `score_multi` is **correct** and the drift is a pure int8
   artefact, exactly as hypothesised. The fix was therefore to settle the flag,
   not to touch the code.

2. **Batching earns 0.96x**, so it currently carries item 1's risk for no speed.
   ~~Keep it only if the fp32 control clears it.~~ **The control cleared it, and
   the re-measured speedup is better than 0.96x:** 1.01x in fp32 (785.7ms →
   781.4ms) and **1.18x** in int8 (370.6ms → 314.8ms), 16 pairs, best of 3. With
   the reshape proven exact in fp32, batching carries no correctness risk and is
   kept.

## The int8 flag: measured unsafe against the gate, 2026-09-02

The fp32 control above says the *code* is right. It says nothing about whether
the int8 drift can reach the decision boundary, which is the question that
actually affects students. `scripts/_verify_gate_margin.py` measures that
directly — for all 56 eval queries it scores the real shortlist in both
precisions and reports each query's distance from `rerank_min_score=-8.0`:

```
queries scored              56
worst int8 drift observed   1.223
smallest gate margin        0.333   ("moddle login")
gate verdict flips          0
safety ratio                0.3x     -> FAIL
```

**Zero verdicts flipped, and that is not reassuring.** The worst drift (1.223) is
**3.7x larger than the smallest margin** (0.333). `moddle login` sits a third of
a logit from being declined while quantization moves scores by over a full logit;
no verdict flipped on *these* queries, but nothing prevents a slightly different
phrasing from landing on the wrong side. A no-regression pass with less margin
than drift is luck, not a property.

`bench_quality.py` agrees the two precisions are otherwise indistinguishable —
`bench_q_fp32_sep02.json` vs `bench_q_int8_sep02.json` differ by **+0.000 on
every** recall, precision, confidence and stability metric, with int8 ~300ms
faster on mean latency. That is precisely why the margin test was needed: an
eval-set comparison cannot see a risk that has not yet materialised on the eval
set.

**Conclusion: `RERANK_QUANTIZE=false`.** The flag buys ~1.18x on a stage that is
a fraction of a request dominated by ~1s of NVIDIA generation, and in exchange it
puts a >1-logit perturbation next to a 0.333-logit margin on the one mechanism
that can silently refuse a legitimate student question. Note `config.py:476`
already defaults it to `False` with the comment "stays off until the full eval
set confirms no recall regression" — the eval set confirms no regression, but the
gate margin does not confirm safety, so the default stands.

**Caveat on the harness:** `_verify_gate_margin.py`'s flip *detector* has not
been falsified — no run has been produced in which it reports a flip, so "0
flips" is measured but the detector is unproven. The margin and drift figures do
not depend on it.

```bash
# Both checks, in the order they should be read:
RERANK_QUANTIZE=false ./.venv/Scripts/python.exe -u scripts/_verify_score_multi.py  # reshape exact?
./.venv/Scripts/python.exe -u scripts/_verify_gate_margin.py                        # drift vs gate margin
```

