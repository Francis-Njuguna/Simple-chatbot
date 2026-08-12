"""Does INT8 raise the CONCURRENT throughput ceiling? Interleaved paired design.

WHY A THIRD CONCURRENCY BENCHMARK
---------------------------------
bench_concurrent.py sweeps (torch_threads x concurrency) in blocks. Two problems
now that the thread question is settled:

  1. Thread count is a measured non-lever (1/2/4 -> 4.69/4.60/4.68 req/s at
     conc=4, inside the noise band). The sweep spends 9 blocks re-answering it.
  2. Block designs die on this machine. The int8 run drifted 52.7% (anchor
     4.04 -> 1.91 req/s) because ~39% background CPU from a browser, an editor
     and Docker moves around during a multi-minute run. A 52.7% drift band
     swamps the ~25% effect being measured.

The only open question is whether int8 raises the SATURATED THROUGHPUT CEILING,
so this measures exactly that and nothing else: one fixed config, fp32 and int8
alternating batch-by-batch, sign test on the paired differences. A busy window
now hits both variants nearly equally and cancels in the pairing.

WHY THROUGHPUT AND NOT LATENCY
------------------------------
At saturation the queue absorbs speedups as shorter waits, not faster calls, so
per-call latency understates the win. Throughput (calls/wall) is what Little's
Law consumes and what the 50-user target is denominated in.

The per-request unit includes embedding, which int8 does NOT touch. That is
deliberate: diluting the effect with real untouched work is what makes the
number a system-level projection instead of a component microbenchmark.

Usage:
    ./.venv/Scripts/python.exe -u scripts/bench_concurrent_paired.py
"""

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent.parent))

PAIRS = 12          # paired batches; each runs both variants once
CALLS_PER_BATCH = 12
CONCURRENCY = 4     # at/near the measured saturation knee on 4 cores
WARMUP_BATCHES = 2


async def run_batch(fn, n_calls: int, concurrency: int) -> tuple[float, list[float]]:
    """Fire n_calls through a semaphore; return (throughput, per-call latencies)."""
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []

    async def one() -> None:
        async with sem:
            t0 = time.perf_counter()
            await asyncio.to_thread(fn)
            lat.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    await asyncio.gather(*[one() for _ in range(n_calls)])
    wall = time.perf_counter() - t0
    return (n_calls / wall if wall else 0.0), lat


async def main() -> int:
    import torch
    from torch.quantization import quantize_dynamic

    from backend.app.rag.embeddings import _SentenceTransformerBackend
    from backend.app.rag.reranker import CrossEncoderReranker

    # Load fp32 explicitly: this script owns the swap, so the config flag must
    # not pre-quantize the model out from under it.
    os.environ["RERANK_QUANTIZE"] = "false"

    embedder = _SentenceTransformerBackend()
    reranker = CrossEncoderReranker()
    reranker.warmup()
    model = reranker._ensure_model()
    if model is None:
        print("reranker unavailable")
        return 1

    wrapper = model[0]
    if "model" not in getattr(wrapper, "_modules", {}):
        print(f"unexpected wrapper layout: {list(wrapper._modules)}")
        return 1

    fp32_inner = wrapper._modules["model"]
    int8_inner = quantize_dynamic(fp32_inner, {torch.nn.Linear}, dtype=torch.qint8)
    n_q = sum(1 for _, m in int8_inner.named_modules()
              if "quantized" in type(m).__module__)
    if n_q == 0:
        print("quantization produced no quantized modules — aborting")
        return 1

    def use_fp32() -> None:
        wrapper._modules["model"] = fp32_inner

    def use_int8() -> None:
        wrapper._modules["model"] = int8_inner

    query = "How do I reset my password for the learning management system?"
    variants = [
        query,
        "How can I reset my LMS password?",
        "I forgot my Moodle password, what should I do?",
        "Steps to recover my learning platform login",
    ]

    # Real corpus chunks: synthetic passages are shorter than this KB's ~414
    # char average and understate rerank cost, which is the whole effect here.
    from backend.app.database.chroma import get_text_collection
    got = get_text_collection().get(limit=64, include=["documents"])
    docs = [d for d in (got.get("documents") or []) if d and d.strip()]
    if len(docs) < 16:
        print(f"only {len(docs)} chunks available — need >=16")
        return 1
    passages = docs[:16]

    def request_work() -> None:
        embedder.embed_texts(variants)
        reranker.score(query, passages)

    # Guard against a silent no-op: if int8 scores match fp32 bit-for-bit the
    # swap did not take and every number below would be fp32-vs-fp32.
    use_fp32()
    s_fp32 = reranker.score(query, passages)
    use_int8()
    s_int8 = reranker.score(query, passages)
    use_fp32()
    if s_fp32 is None or s_int8 is None:
        print("scoring unavailable — aborting")
        return 1
    if max(abs(a - b) for a, b in zip(s_fp32, s_int8)) == 0.0:
        print("!! int8 scores BIT-IDENTICAL to fp32 — swap did not take. Aborting.")
        return 1

    print("=" * 78)
    print("PAIRED CONCURRENT THROUGHPUT — fp32 vs INT8")
    print("=" * 78)
    print(f"  cpu_count      : {os.cpu_count()}   torch threads: {torch.get_num_threads()}")
    print(f"  concurrency    : {CONCURRENCY}   calls/batch: {CALLS_PER_BATCH}")
    print(f"  paired batches : {PAIRS}   quantized modules: {n_q}")
    print(f"  passages       : {len(passages)} real KB chunks, "
          f"mean {statistics.fmean(len(p) for p in passages):.0f} chars")
    print()

    for _ in range(WARMUP_BATCHES):
        use_fp32(); await run_batch(request_work, CALLS_PER_BATCH, CONCURRENCY)
        use_int8(); await run_batch(request_work, CALLS_PER_BATCH, CONCURRENCY)
    use_fp32()

    fp32_tp: list[float] = []
    int8_tp: list[float] = []
    fp32_p95: list[float] = []
    int8_p95: list[float] = []

    def p95(xs: list[float]) -> float:
        s = sorted(xs)
        return s[max(0, int(len(s) * 0.95) - 1)]

    for i in range(PAIRS):
        # Alternate within-pair order so turbo ramp / cache state cancels too.
        if i % 2 == 0:
            use_fp32(); f_tp, f_lat = await run_batch(request_work, CALLS_PER_BATCH, CONCURRENCY)
            use_int8(); q_tp, q_lat = await run_batch(request_work, CALLS_PER_BATCH, CONCURRENCY)
        else:
            use_int8(); q_tp, q_lat = await run_batch(request_work, CALLS_PER_BATCH, CONCURRENCY)
            use_fp32(); f_tp, f_lat = await run_batch(request_work, CALLS_PER_BATCH, CONCURRENCY)
        use_fp32()
        fp32_tp.append(f_tp); int8_tp.append(q_tp)
        fp32_p95.append(p95(f_lat)); int8_p95.append(p95(q_lat))
        print(f"  pair {i+1:>2}/{PAIRS}: fp32 {f_tp:5.2f} req/s   int8 {q_tp:5.2f} req/s   "
              f"{'int8' if q_tp > f_tp else 'fp32':>4} wins")

    f_med, q_med = statistics.median(fp32_tp), statistics.median(int8_tp)
    diffs = [q - f for q, f in zip(int8_tp, fp32_tp)]
    wins = sum(1 for d in diffs if d > 0)
    sd = (PAIRS * 0.25) ** 0.5
    z = (wins - PAIRS / 2) / sd if sd else 0.0

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  fp32 : median {f_med:5.2f} req/s  max {max(fp32_tp):5.2f}  "
          f"p95 lat median {statistics.median(fp32_p95):7.1f}ms")
    print(f"  int8 : median {q_med:5.2f} req/s  max {max(int8_tp):5.2f}  "
          f"p95 lat median {statistics.median(int8_p95):7.1f}ms")
    print()
    print(f"  median paired diff : {statistics.median(diffs):+.2f} req/s "
          f"(positive = int8 faster)")
    print(f"  int8 won           : {wins}/{PAIRS}   sign-test z = {z:+.2f} "
          f"({'significant' if abs(z) > 2 else 'NOT significant'})")
    print(f"  throughput ratio   : {q_med / f_med if f_med else 0:.2f}x (median)")
    print(f"  throughput ratio   : {max(int8_tp) / max(fp32_tp) if max(fp32_tp) else 0:.2f}x "
          f"(max/max — least noise-contaminated)")
    print()

    # Project against the target using the least-contaminated estimate: the best
    # batch each variant achieved is the one that fought background load least.
    print("  Little's Law, 50 users / p95 < 10s  =>  need 5.00 req/s")
    print(f"    fp32 best batch : {max(fp32_tp):.2f} req/s")
    print(f"    int8 best batch : {max(int8_tp):.2f} req/s")
    for label, tp in (("fp32", max(fp32_tp)), ("int8", max(int8_tp))):
        if tp >= 5.0:
            print(f"    {label}: target reachable on this box")
        else:
            print(f"    {label}: {5.0/tp:.2f}x short -> needs {5.0/tp:.1f}x the cores "
                  f"(or a GPU) to hit 50 users")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
