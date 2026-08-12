"""Batched vs looped rerank scoring — noise-robust A/B under background load.

WHY A SECOND VERSION
--------------------
bench_batching.py measured looped-then-batched-then-looped in three blocks and
tripped its own drift guard at 26.4% (stdev 1573ms). The machine has ~39%
background CPU from a browser, an editor and Docker on 4 cores, so any BLOCK of
trials can land in a busy window and slander whichever variant it timed.

Block designs cannot fix that. Interleaving can: alternate A and B on every
single trial, so a busy window hits both variants nearly equally and cancels in
the paired difference. This is the standard defence when you cannot get a quiet
machine.

Reported statistic is the median of PAIRED differences (per-trial B - A) plus a
sign test on the pairs. "Batched won 47 of 60 paired trials" survives
background noise in a way that "median A vs median B" does not.

Usage:
    ./.venv/Scripts/python.exe -u scripts/bench_batching_paired.py
"""

import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent.parent))

WARMUP = 5
PAIRS = 60  # paired trials; each runs both variants once


def main() -> int:
    from backend.app.rag.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranker.warmup()
    model = reranker._ensure_model()
    if model is None:
        print("reranker unavailable")
        return 1

    form_a = "How do I reset my password for the learning management system?"
    form_b = "LMS password reset steps"
    passages = [
        f"This is passage {i} about password resets, LMS login, account recovery, "
        f"Moodle authentication, help desk procedures, IT support services, "
        f"and troubleshooting access issues for educational institutions."
        for i in range(16)
    ]
    forms = [form_a, form_b]
    n = len(passages)
    pairs32 = [(f, p) for f in forms for p in passages]

    def looped() -> list[float]:
        best = None
        for form in forms:
            s = [float(x) for x in model.predict([(form, p) for p in passages])]
            best = s if best is None else [max(b, v) for b, v in zip(best, s)]
        return best

    def batched() -> list[float]:
        flat = [float(x) for x in model.predict(pairs32, batch_size=16)]
        best = flat[:n]
        for i in range(1, len(forms)):
            best = [max(b, v) for b, v in zip(best, flat[i * n:(i + 1) * n])]
        return best

    # ---- correctness -----------------------------------------------------
    a, b = looped(), batched()
    max_diff = max(abs(x - y) for x, y in zip(a, b))
    same_order = (sorted(range(n), key=lambda i: -a[i])
                  == sorted(range(n), key=lambda i: -b[i]))
    print("=" * 74)
    print("CORRECTNESS")
    print("=" * 74)
    print(f"  max |score diff| : {max_diff:.3e}   (float assoc. in batched matmul)")
    print(f"  identical ranking: {same_order}")
    print()
    if not same_order or max_diff > 1e-4:
        print("  !! not a behaviour-preserving swap — stopping")
        return 1

    for _ in range(WARMUP):
        looped()
        batched()

    # ---- interleaved paired timing ---------------------------------------
    print("=" * 74)
    print(f"PAIRED INTERLEAVED TIMING ({PAIRS} pairs, alternating order per trial)")
    print("=" * 74)

    loop_ms: list[float] = []
    batch_ms: list[float] = []
    diffs: list[float] = []

    for i in range(PAIRS):
        # Alternate which variant goes first, so any within-pair ordering
        # advantage (cache state, turbo ramp) cancels across pairs too.
        if i % 2 == 0:
            t0 = time.perf_counter(); looped(); lm = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter(); batched(); bm = (time.perf_counter() - t0) * 1000
        else:
            t0 = time.perf_counter(); batched(); bm = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter(); looped(); lm = (time.perf_counter() - t0) * 1000
        loop_ms.append(lm)
        batch_ms.append(bm)
        diffs.append(bm - lm)  # negative => batched faster

    loop_med = statistics.median(loop_ms)
    batch_med = statistics.median(batch_ms)
    diff_med = statistics.median(diffs)
    batch_wins = sum(1 for d in diffs if d < 0)

    print(f"  looped   : median {loop_med:7.1f}ms   min {min(loop_ms):7.1f}   "
          f"stdev {statistics.stdev(loop_ms):6.1f}")
    print(f"  batched  : median {batch_med:7.1f}ms   min {min(batch_ms):7.1f}   "
          f"stdev {statistics.stdev(batch_ms):6.1f}")
    print()
    print(f"  median paired diff : {diff_med:+.1f}ms  (negative = batched faster)")
    print(f"  batched won        : {batch_wins}/{PAIRS} paired trials")

    # Sign test: under the null "no difference", wins ~ Binomial(n, 0.5).
    # Normal approximation is fine at n=60.
    expected = PAIRS / 2
    sd = (PAIRS * 0.25) ** 0.5
    z = (batch_wins - expected) / sd if sd else 0.0
    print(f"  sign-test z        : {z:+.2f}  "
          f"({'significant' if abs(z) > 2 else 'NOT significant'} at ~95%)")
    print()

    # min-vs-min is the least noise-contaminated comparison available: the
    # fastest observed run of each is the one that fought for cores the least.
    speedup_med = loop_med / batch_med if batch_med else 0.0
    speedup_min = min(loop_ms) / min(batch_ms) if min(batch_ms) else 0.0
    print(f"  speedup (median)   : {speedup_med:.3f}x")
    print(f"  speedup (min/min)  : {speedup_min:.3f}x   <- least noise-contaminated")
    print()

    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    if abs(z) <= 2:
        print(f"  No significant difference ({batch_wins}/{PAIRS} wins, z={z:+.2f}).")
        print("  Batching the query-forms is NOT a win on this model/size.")
        print("  Keep the existing loop in retriever._score_all() — changing it")
        print("  would add code for no measured gain.")
    elif z < -2:
        print(f"  Batched is significantly FASTER: {batch_wins}/{PAIRS} wins, "
              f"{diff_med:+.0f}ms median, {speedup_min:.2f}x on min/min.")
        print("  Ranking is identical, so this is free throughput.")
    else:
        print(f"  Batched is significantly SLOWER: {batch_wins}/{PAIRS} wins, "
              f"{diff_med:+.0f}ms median. Keep the loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
