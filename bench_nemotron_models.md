# Nemotron Model Comparison (NVIDIA NIM)

Measured 2026-09-04 on `bench_nemotron_models.py`. Superseded fields in
`bench_nemotron_models.json`; this file is the human-readable finding.

## Why this benchmark was run

`OPENAI_MODEL` was set to `meta/llama-3.1-8b-instruct`, which NVIDIA retired on
2026-08-26. The endpoint now answers `410 Gone`, which is the 400-family error the
backend was surfacing. A replacement had to be chosen on measured evidence.

## Comparability contract

The old baseline **cannot be re-run** — the model is permanently gone. Its row is
therefore *quoted* from `bench_llm_providers.json` (measured 2026-08-20). That is
only legitimate because this run holds four things identical to it:

1. The same 20 frozen retrieval contexts, reloaded from that JSON and
   **sha256-verified unchanged** on load (the script aborts if any differ).
2. The same generation settings: `max_tokens=2048`, `temperature` omitted.
3. The same measurement function (`measure()` imported, not reimplemented), so
   TTFT and success are defined identically.
4. The same quality scoring (`quality_signals` / `quality_score` imported).

AgentRouter / `claude-opus-5` is **absent**: there is no key configured, so it
could not be re-measured. Its old numbers are not repeated here as if fresh.

## Results — production settings (`LLM_FIRST_TOKEN_TIMEOUT=30`)

| Model | Status | Reqs | Success | TTFT p50 | TTFT p95 | Total p50 | Quality | 30s cutoffs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `meta/llama-3.1-8b-instruct` | quoted, 410 Gone | 100 | 100% | **1.00s** | **1.16s** | **3.42s** | **56.6** | 0 |
| `nvidia/nemotron-3-super-120b-a12b` | measured | 28 | **75%** | 11.32s | 27.06s | 16.07s | 46.7 | 1 |
| `nvidia/nemotron-3-ultra-550b-a55b` | measured | 28 | 36% | 22.15s | 26.05s | 29.38s | 35.2 | 5 |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | measured | 28 | 11% | 14.62s | 27.70s | 14.69s | 34.1 | 20 |

The run was stopped at 46/120 sequential requests (the host slept mid-stream and
wedged a stream). The separation between candidates was already unambiguous and
stable, so it was not restarted; sample sizes are stated rather than rounded up.

**Zero 429s across the entire run.** Per `bench_nvidia_concurrency.md`, the free
tier binds on total request volume, so this confirms the latency figures above are
genuine model behaviour and not a quota curve. All HTTP statuses were `200`; the
failures are mid-stream `APIError: Service temporarily overloaded` from NVIDIA,
plus first-token budget cutoffs.

## Quality breakdown (successful answers only)

| Metric | llama-3.1-8b (quoted) | super-120b | ultra-550b |
|---|---:|---:|---:|
| Correctness | 0.808 | **0.817** | 0.667 |
| Groundedness | **0.573** | 0.496 | 0.546 |
| Citation rate | **0.470** | 0.143 | 0.200 |
| Completeness | 1.000 | 1.000 | 1.000 |
| Hallucinated URL rate | **0.190** | 0.238 | 0.400 |
| Mean answer chars | 1604 | 1604 | 1391 |

`super-120b` matches the retired model on correctness and answer length, but
**cites sources far less often** (0.143 vs 0.470) and invents URLs slightly more.
That is a real regression in a help-desk context and is the strongest argument for
tightening the citation instruction in `prompts/templates.py` after the switch.

## The confound, and why lightning is still rejected

Lightning's 11% success under production settings was **not** a fair verdict: 20 of
its 25 failures were the 30s first-token budget, not the model refusing. Re-running
it with `LLM_FIRST_TOKEN_TIMEOUT=180` gave **100% success (5/5)** — so the budget
was clipping it.

It is still unusable, for a worse reason found by that same relaxed run:

| Model | Successful answers | CoT leaked into user text | Hit `max_tokens` mid-answer | TTFT p50 |
|---|---:|---:|---:|---:|
| `nvidia/nemotron-3.5-lightning-30b-a3b` | 4 | **4/4** | **3/4** | 38.4s |
| `nvidia/nemotron-3-super-120b-a12b` | 3 | **0/3** | 0/3 | 10.8s |

Lightning emits its raw chain-of-thought into the `content` field, so answers begin:

```
Here's a thinking process:

1.  **Analyze User Input:**
   - User provided retrieved knowledge base context (3 articles)
```

…then consume the full 2048-token budget reasoning aloud and truncate mid-sentence
before delivering the procedure. A student would see the model's scratchpad and
never a finished answer. `super-120b` keeps reasoning in the separate
`reasoning_content` field, which `_extract_text` correctly does not yield.

This is why the earlier bare `ping` probe (2.8s, clean) was misleading: with a real
RAG prompt the reasoning phase dominates, and only the full harness exposes it.

## Recommendation

Use **`nvidia/nemotron-3-super-120b-a12b`**. It is the only candidate that is
simultaneously reliable (75%, best of the three), fast enough for the configured
budget, and free of chain-of-thought leakage.

Two caveats that must not be lost:

- **It is ~11x slower to first token than the retired model** (11.32s vs 1.00s)
  and its p95 of 27.06s sits under the 30s budget with almost no headroom. One
  slow request becomes a user-visible failure. Consider raising
  `LLM_FIRST_TOKEN_TIMEOUT` to 45-60s, which costs nothing when answers are fast.
- **75% success is not production-grade.** The residual failures are NVIDIA's own
  `Service temporarily overloaded`, so retry/fallback matters more than before —
  `LLM_MAX_RETRIES=1` is currently thin for a provider failing a quarter of calls.

## Scope

- Sequential only (concurrency 1), matching production `LLM_MAX_CONCURRENCY=1`.
- Concurrency scaling deliberately not re-measured: per
  `bench_nvidia_concurrency.md`, the free-tier limiter binds before app capacity,
  so such a sweep charts quota policy rather than this server.
- Retrieval excluded — contexts are frozen, so these are generation-only figures.
- Quality is a repeatable lexical proxy (term coverage, grounding, citations,
  completeness, URL hallucination), not human review.
