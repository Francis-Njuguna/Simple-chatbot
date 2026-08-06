"""Does batching the rerank query-forms into one predict() call beat looping?

THE HYPOTHESIS
--------------
retriever._score_all() currently does:

    for form in rerank_forms:            # 2 forms
        scores = reranker.score(form, texts)   # 16 pairs -> one predict()
    # then max-pools the two score lists

That is 2 sequential predict() calls of 16 pairs. sentence-transformers'
CrossEncoder defaults to batch_size=32, so each call submits a HALF-EMPTY
batch, pays its own tokenisation + forward-pass launch overhead, and the
per-batch fixed costs are paid twice.

Building all 32 (form, passage) pairs and issuing ONE predict() should produce
bit-identical scores — max-pooling is associative and order-independent — while
paying the fixed costs once and filling the batch.

This script checks BOTH halves of that claim:
  1. speed   — is batched actually faster, warmup-corrected, median-based?
  2. identity — are the max-pooled scores numerically identical?

A speedup that changes scores is not a free win, so (2) gates (1).

Usage:
    ./.venv/Scripts/python.exe -u scripts/bench_batching.py
"""

import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent.parent))

WARMUP = 3
TRIALS = 15


def timed(fn, trials: int = TRIALS) -> dict:
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "median": statistics.median(ts),
        "mean": statistics.fmean(ts),
        "min": ts[0],
        "p95": ts[max(0, int(len(ts) * 0.95) - 1)],
        "stdev": statistics.stdev(ts) if len(ts) > 1 else 0.0,
    }


def main() -> int:
    from backend.app.rag.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranker.warmup()
    model = reranker._ensure_model()
    if model is None:
        print("reranker unavailable — cannot benchmark")
        return 1

    # Two genuinely different phrasings, as the retriever would produce.
    form_a = "How do I reset my password for the learning management system?"
    form_b = "LMS password reset steps"
    passages = [
        f"This is passage {i} about password resets, LMS login, account recovery, "
        f"Moodle authentication, help desk procedures, IT support services, "
        f"and troubleshooting access issues for educational institutions."
        for i in range(16)
    ]
    forms = [form_a, form_b]

    # ---- the two implementations ----------------------------------------
    def looped() -> list[float]:
        """Current production shape: one predict() per form, then max-pool."""
        best = None
        for form in forms:
            scores = model.predict([(form, p) for p in passages])
            scores = [float(s) for s in scores]
            best = scores if best is None else [max(b, s) for b, s in zip(best, scores)]
        return best

    def batched() -> list[float]:
        """Proposed: all form x passage pairs in ONE predict(), then max-pool."""
        pairs = [(form, p) for form in forms for p in passages]
        flat = [float(s) for s in model.predict(pairs)]
        n = len(passages)
        best = flat[:n]
        for i in range(1, len(forms)):
            chunk = flat[i * n:(i + 1) * n]
            best = [max(b, s) for b, s in zip(best, chunk)]
        return best

    # ---- correctness gate FIRST -----------------------------------------
    print("=" * 72)
    print("CORRECTNESS: are the max-pooled scores identical?")
    print("=" * 72)
    a, b = looped(), batched()
    max_abs_diff = max(abs(x - y) for x, y in zip(a, b))
    identical = max_abs_diff == 0.0
    close = max_abs_diff < 1e-5
    print(f"  n_scores       : {len(a)}")
    print(f"  max |diff|     : {max_abs_diff:.3e}")
    print(f"  bit-identical  : {identical}")
    print(f"  within 1e-5    : {close}")
    # Ordering is what actually reaches the user; scores only matter via rank.
    rank_a = sorted(range(len(a)), key=lambda i: -a[i])
    rank_b = sorted(range(len(b)), key=lambda i: -b[i])
    print(f"  same ordering  : {rank_a == rank_b}")
    print()
    if not close:
        print("  !! scores differ materially — batching is NOT a free swap")
        return 1

    # ---- speed -----------------------------------------------------------
    print("=" * 72)
    print("SPEED (warmup-corrected, median of %d)" % TRIALS)
    print("=" * 72)

    # A/B/A: measure looped, then batched, then looped again for drift.
    s_loop_open = timed(looped)
    s_batch = timed(batched)
    s_loop_close = timed(looped)

    drift = (
        abs(s_loop_close["median"] - s_loop_open["median"]) / s_loop_open["median"]
        if s_loop_open["median"] else 0.0
    )

    print(f"  looped  (open) : median {s_loop_open['median']:7.1f}ms  "
          f"p95 {s_loop_open['p95']:7.1f}ms  stdev {s_loop_open['stdev']:5.1f}")
    print(f"  batched        : median {s_batch['median']:7.1f}ms  "
          f"p95 {s_batch['p95']:7.1f}ms  stdev {s_batch['stdev']:5.1f}")
    print(f"  looped  (close): median {s_loop_close['median']:7.1f}ms  <- drift check")
    print(f"  drift          : {drift*100:.1f}% "
          f"{'OK' if drift <= 0.25 else 'DRIFTED — rerun'}")
    print()

    loop_ref = statistics.median([s_loop_open["median"], s_loop_close["median"]])
    speedup = loop_ref / s_batch["median"] if s_batch["median"] else 0.0
    saved = loop_ref - s_batch["median"]
    print(f"  speedup        : {speedup:.2f}x   ({saved:+.0f}ms per request)")
    print()

    # ---- does batch_size matter? ----------------------------------------
    print("=" * 72)
    print("BATCH SIZE SWEEP (32 pairs, one predict())")
    print("=" * 72)
    pairs32 = [(form, p) for form in forms for p in passages]
    for bs in (8, 16, 32, 64):
        st = timed(lambda b=bs: model.predict(pairs32, batch_size=b), trials=10)
        print(f"  batch_size={bs:<3}   median {st['median']:7.1f}ms  p95 {st['p95']:7.1f}ms")
    print()

    if drift > 0.25:
        print("VERDICT: run drifted, numbers untrustworthy — rerun on a quiet machine")
        return 1
    if speedup > 1.05:
        print(f"VERDICT: batching wins ({speedup:.2f}x, {saved:.0f}ms saved) "
              f"with identical ranking — free throughput, zero quality cost")
    elif speedup < 0.95:
        print(f"VERDICT: batching is SLOWER ({speedup:.2f}x) — keep the loop")
    else:
        print(f"VERDICT: no material difference ({speedup:.2f}x) — not worth the change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
