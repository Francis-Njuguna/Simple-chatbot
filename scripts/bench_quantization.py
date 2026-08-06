"""INT8 dynamic quantization of the cross-encoder: speed AND ranking fidelity.

WHY THIS IS THE REMAINING LEVER
-------------------------------
profile_pipeline.py: rerank is 84% of retrieval wall time (2572ms/query on real
KB chunks). Everything else is rounding error (embed 424ms, chroma 72ms, pg
32ms, bm25 2ms).

The two ways to cut rerank cost by scoring FEWER pairs are both measured and
both cost recall:
  * RERANK_QUERY_FORMS 2->1  : partial recall 0.667 -> 0.000. Catastrophic.
  * RERANK_SHORTLIST 16->12/8: loses "exam registration" (short 1.000 -> 0.800).

So the only lever left is making each pair CHEAPER. Dynamic INT8 quantization
replaces Linear weights with int8 and computes in int8, which on CPU is the
standard 2-3x win for transformer inference. It changes the arithmetic, so it
is NOT automatically safe: this script measures fidelity before speed.

FIDELITY IS THE GATE
--------------------
Reranking only affects the answer through ORDER, so the metric that matters is
whether quantized scoring produces the same ranking, measured on REAL corpus
chunks (synthetic passages are shorter and understate the cost and the risk).
Reported:
  * exact top-k set agreement (k = RERANK_TOP_N) — what reaches the LLM
  * top-1 agreement           — dominates the assembled context
  * Spearman rank correlation — overall ordering drift
  * max/mean absolute score delta

Speed uses the same interleaved paired design as bench_batching_paired.py,
because this machine has ~39% background CPU and block designs get slandered
by busy windows (bench_batching.py tripped its own drift guard at 26.4%).

Usage:
    ./.venv/Scripts/python.exe -u scripts/bench_quantization.py
"""

import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent.parent))

WARMUP = 4
PAIRS = 40
TOP_K = 5


def _spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without scipy."""
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 2:
        return 1.0
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 1.0


def main() -> int:
    import torch
    from torch.quantization import quantize_dynamic

    from backend.app.database.chroma import get_text_collection
    from backend.app.rag.reranker import CrossEncoderReranker

    # ---- real corpus chunks, real token lengths --------------------------
    col = get_text_collection()
    got = col.get(limit=64, include=["documents"])
    docs = [d for d in (got.get("documents") or []) if d and d.strip()]
    if len(docs) < 16:
        print(f"only {len(docs)} chunks available — need >=16")
        return 1
    passages = docs[:16]
    avg_chars = statistics.fmean(len(p) for p in passages)

    reranker = CrossEncoderReranker()
    reranker.warmup()
    model = reranker._ensure_model()
    if model is None:
        print("reranker unavailable")
        return 1

    forms = [
        "How do I reset my password for the learning management system?",
        "LMS password reset steps",
    ]

    print("=" * 76)
    print("INT8 DYNAMIC QUANTIZATION — CROSS-ENCODER")
    print("=" * 76)
    print(f"  passages      : {len(passages)} real KB chunks, mean {avg_chars:.0f} chars")
    print(f"  query forms   : {len(forms)} (max-pooled, as production does)")
    print(f"  torch threads : {torch.get_num_threads()}")
    print()

    def score_with(m, form: str) -> list[float]:
        return [float(s) for s in m.predict([(form, p) for p in passages])]

    def maxpool(m) -> list[float]:
        best = None
        for f in forms:
            s = score_with(m, f)
            best = s if best is None else [max(x, y) for x, y in zip(best, s)]
        return best

    fp32 = maxpool(model)

    # ---- quantize --------------------------------------------------------
    # Getting this swap wrong produces a SILENT no-op that looks like a clean
    # experiment, so the mechanism is spelled out.
    #
    # A sentence-transformers 5.x CrossEncoder is a Sequential whose child "0"
    # is a Transformer wrapper. Two wrong targets:
    #   * `model.model = q`      -> `model` is a property; nn.Module.__setattr__
    #     registers q as a NEW Sequential child, so forward() feeds the raw HF
    #     module a features dict and dies on input_ids.size().
    #   * `wrapper.auto_model = q` -> `auto_model` is a read-only property alias;
    #     assignment creates a shadowing instance attribute while forward() keeps
    #     using the fp32 module. Scores come back BIT-IDENTICAL and the benchmark
    #     silently compares fp32 against fp32 (this actually happened).
    #
    # The real module is the registered child `wrapper._modules["model"]`.
    print("quantizing (Linear -> qint8)...")
    wrapper = model[0]
    if "model" not in wrapper._modules:
        print(f"unexpected wrapper layout: {list(wrapper._modules)} — aborting")
        return 1

    original_inner = wrapper._modules["model"]
    qinner = quantize_dynamic(original_inner, {torch.nn.Linear}, dtype=torch.qint8)
    n_quantized = sum(
        1 for _, mod in qinner.named_modules()
        if "quantized" in type(mod).__module__
    )
    print(f"  swapping {type(wrapper).__name__}._modules['model']  "
          f"quantized modules: {n_quantized}")

    def use_int8() -> None:
        wrapper._modules["model"] = qinner

    def use_fp32() -> None:
        wrapper._modules["model"] = original_inner

    use_int8()
    try:
        int8 = maxpool(model)
    finally:
        use_fp32()

    # Guard against the silent no-op: if int8 == fp32 exactly, the swap failed.
    if max(abs(x - y) for x, y in zip(fp32, int8)) == 0.0:
        print()
        print("  !! int8 scores are BIT-IDENTICAL to fp32 — the swap did not take")
        print("     effect. Any speed number below would be fp32-vs-fp32. Aborting.")
        return 1

    # sanity: fp32 path unchanged after the swap-back
    fp32_again = maxpool(model)
    drift_selfcheck = max(abs(x - y) for x, y in zip(fp32, fp32_again))

    # ---- fidelity gate ---------------------------------------------------
    print()
    print("=" * 76)
    print("FIDELITY (this gates everything below)")
    print("=" * 76)
    diffs = [abs(x - y) for x, y in zip(fp32, int8)]
    order_fp32 = sorted(range(len(fp32)), key=lambda i: -fp32[i])
    order_int8 = sorted(range(len(int8)), key=lambda i: -int8[i])
    topk_fp32, topk_int8 = set(order_fp32[:TOP_K]), set(order_int8[:TOP_K])
    rho = _spearman(fp32, int8)

    print(f"  fp32 self-check delta : {drift_selfcheck:.2e}  "
          f"({'clean swap-back' if drift_selfcheck < 1e-6 else 'MODEL MUTATED — abort'})")
    print(f"  max |score delta|     : {max(diffs):.4f}")
    print(f"  mean |score delta|    : {statistics.fmean(diffs):.4f}")
    print(f"  spearman rho          : {rho:.4f}")
    print(f"  top-1 identical       : {order_fp32[0] == order_int8[0]}")
    print(f"  top-{TOP_K} set identical    : {topk_fp32 == topk_int8}"
          f"  (overlap {len(topk_fp32 & topk_int8)}/{TOP_K})")
    print(f"  full order identical  : {order_fp32 == order_int8}")
    print()

    if drift_selfcheck >= 1e-6:
        print("  !! quantize_dynamic mutated the fp32 model — results invalid")
        return 1

    fidelity_ok = (topk_fp32 == topk_int8) and order_fp32[0] == order_int8[0]

    # ---- interleaved paired speed ---------------------------------------
    for _ in range(WARMUP):
        maxpool(model)

    print("=" * 76)
    print(f"PAIRED INTERLEAVED SPEED ({PAIRS} pairs)")
    print("=" * 76)

    fp32_ms: list[float] = []
    int8_ms: list[float] = []
    diffs_ms: list[float] = []

    for i in range(PAIRS):
        if i % 2 == 0:
            use_fp32()
            t0 = time.perf_counter(); maxpool(model); f = (time.perf_counter() - t0) * 1000
            use_int8()
            t0 = time.perf_counter(); maxpool(model); q = (time.perf_counter() - t0) * 1000
            use_fp32()
        else:
            use_int8()
            t0 = time.perf_counter(); maxpool(model); q = (time.perf_counter() - t0) * 1000
            use_fp32()
            t0 = time.perf_counter(); maxpool(model); f = (time.perf_counter() - t0) * 1000
        fp32_ms.append(f)
        int8_ms.append(q)
        diffs_ms.append(q - f)

    f_med, q_med = statistics.median(fp32_ms), statistics.median(int8_ms)
    wins = sum(1 for d in diffs_ms if d < 0)
    sd = (PAIRS * 0.25) ** 0.5
    z = (wins - PAIRS / 2) / sd if sd else 0.0

    print(f"  fp32 : median {f_med:7.1f}ms  min {min(fp32_ms):7.1f}  "
          f"stdev {statistics.stdev(fp32_ms):6.1f}")
    print(f"  int8 : median {q_med:7.1f}ms  min {min(int8_ms):7.1f}  "
          f"stdev {statistics.stdev(int8_ms):6.1f}")
    print()
    print(f"  median paired diff : {statistics.median(diffs_ms):+.1f}ms "
          f"(negative = int8 faster)")
    print(f"  int8 won           : {wins}/{PAIRS}   sign-test z = {z:+.2f} "
          f"({'significant' if abs(z) > 2 else 'NOT significant'})")
    print(f"  speedup (median)   : {f_med / q_med if q_med else 0:.2f}x")
    print(f"  speedup (min/min)  : "
          f"{min(fp32_ms) / min(int8_ms) if min(int8_ms) else 0:.2f}x")
    print()

    # ---- verdict ---------------------------------------------------------
    speedup = f_med / q_med if q_med else 0.0
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    # z is computed from INT8 wins, so z > +2 means int8 is significantly
    # faster. (An earlier revision tested `z >= -2` here and printed "NO WIN"
    # over a genuine z=+4.74 result.)
    if not fidelity_ok:
        print(f"  REJECT: ranking changed (top-{TOP_K} overlap "
              f"{len(topk_fp32 & topk_int8)}/{TOP_K}, top-1 "
              f"{'same' if order_fp32[0]==order_int8[0] else 'DIFFERENT'}).")
        print("  A faster reranker that reorders the shortlist is a quality")
        print("  regression, which the brief rules out. Do not ship on speed alone.")
    elif z <= 2:
        print(f"  NO WIN: int8 won only {wins}/{PAIRS} (z={z:+.2f}), "
              f"{speedup:.2f}x median.")
        print("  Fidelity is fine but there is no measurable speedup to buy.")
    else:
        rerank_share_ms = 2572.0
        new_rerank = rerank_share_ms / speedup
        saved = rerank_share_ms - new_rerank
        print(f"  SHIP-CANDIDATE: {speedup:.2f}x faster ({wins}/{PAIRS}, z={z:+.2f}) "
              f"with identical top-{TOP_K} and top-1.")
        print(f"  Projected rerank: {rerank_share_ms:.0f}ms -> {new_rerank:.0f}ms "
              f"({saved:.0f}ms saved/query)")
        print("  MUST still pass full scripts/bench_quality.py --compare before shipping:")
        print("  16 passages on one query is not the 56-query eval set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
