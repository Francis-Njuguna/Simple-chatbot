"""Concurrent throughput benchmark — measures the quantity that actually matters.

WHY THIS EXISTS
---------------
bench_optimizations_v2.py measured single-call latency at torch_threads =
1/2/4 and found threads=1 was ~18% SLOWER. That result is real but it answers
the wrong question. One isolated call obviously goes faster with more intra-op
threads; nothing is competing with it.

The production problem is different: anyio.to_thread.run_sync has a default
limiter of 40 threads, so up to 40 inference calls run at once, each spawning
`torch_threads` intra-op workers, on 4 physical cores. The binding constraint
under load is oversubscription, not per-call parallelism.

So this script measures AGGREGATE THROUGHPUT and TAIL LATENCY as a function of
(torch_threads x concurrency), which is what Little's Law actually consumes.
A config can lose on single-call latency and still win decisively here.

Usage:
    python scripts/bench_concurrent.py --output bench_concurrent.json
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

WARMUP_CALLS = 4
CALLS_PER_CONFIG = 32
DRIFT_TOLERANCE = 0.25


@dataclass
class ConcurrentResult:
    name: str
    torch_threads: int
    concurrency: int
    n_calls: int
    wall_s: float
    throughput_per_s: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    notes: str = ""


async def measure_concurrent(
    name: str,
    fn: Callable[[], Any],
    torch_threads: int,
    concurrency: int,
    n_calls: int = CALLS_PER_CONFIG,
    notes: str = "",
) -> ConcurrentResult:
    """Fire n_calls through a semaphore of width `concurrency`; time each call
    and the whole batch.

    Per-call latency is measured inside the worker thread, so it includes time
    spent fighting other threads for cores — that contention is the effect we
    are trying to see, not noise to be excluded.
    """
    # Warm up at this thread setting: changing torch_threads can trigger
    # re-planning inside the intra-op pool on the next forward pass.
    for _ in range(WARMUP_CALLS):
        await asyncio.to_thread(fn)

    sem = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []

    async def one_call() -> None:
        async with sem:
            start = time.perf_counter()
            await asyncio.to_thread(fn)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    wall_start = time.perf_counter()
    await asyncio.gather(*[one_call() for _ in range(n_calls)])
    wall_s = time.perf_counter() - wall_start

    latencies_ms.sort()
    p95_idx = max(0, int(len(latencies_ms) * 0.95) - 1)

    return ConcurrentResult(
        name=name,
        torch_threads=torch_threads,
        concurrency=concurrency,
        n_calls=n_calls,
        wall_s=wall_s,
        throughput_per_s=n_calls / wall_s if wall_s else 0.0,
        latency_median_ms=statistics.median(latencies_ms),
        latency_p95_ms=latencies_ms[p95_idx],
        latency_max_ms=latencies_ms[-1],
        notes=notes,
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="bench_concurrent.json")
    args = parser.parse_args()

    import torch

    from backend.app.rag.embeddings import _SentenceTransformerBackend
    from backend.app.rag.reranker import CrossEncoderReranker

    print("=" * 78)
    print("CONCURRENT THROUGHPUT BENCHMARK")
    print("=" * 78)
    print(f"  cpu_count          : {os.cpu_count()}")
    print(f"  torch default intra: {torch.get_num_threads()}")
    print(f"  calls per config   : {CALLS_PER_CONFIG} ({WARMUP_CALLS} warmup, discarded)")
    print()

    print("loading models...")
    embedder = _SentenceTransformerBackend()
    reranker = CrossEncoderReranker()
    reranker.warmup()
    print("models loaded")
    print()

    query = "How do I reset my password for the learning management system?"
    variants = [
        query,
        "How can I reset my LMS password?",
        "I forgot my Moodle password, what should I do?",
        "Steps to recover my learning platform login",
    ]
    passages = [
        f"This is passage {i} about password resets, LMS login, account recovery, "
        f"Moodle authentication, help desk procedures, IT support services, "
        f"and troubleshooting access issues for educational institutions."
        for i in range(16)
    ]

    default_threads = torch.get_num_threads()
    results: list[ConcurrentResult] = []

    # A realistic per-request unit of CPU work: embed the query set, then
    # score the shortlist. This is what one /chat request costs, minus the
    # network-bound LLM call.
    def request_work() -> None:
        embedder.embed_texts(variants)
        reranker.score(query, passages)

    THREAD_SETTINGS = (1, 2, 4)
    CONCURRENCY_LEVELS = (1, 4, 16)

    # ---- opening drift anchor --------------------------------------------
    torch.set_num_threads(default_threads)
    anchor_open = await measure_concurrent(
        "anchor_open", request_work, default_threads, 4, notes="opening drift anchor"
    )
    results.append(anchor_open)
    print(f"drift anchor (open): threads={default_threads} conc=4  "
          f"{anchor_open.throughput_per_s:.2f} req/s")
    print()

    print(f"{'threads':>8} {'conc':>5} {'req/s':>9} {'median ms':>11} "
          f"{'p95 ms':>9} {'max ms':>9}")
    print("-" * 78)

    for threads in THREAD_SETTINGS:
        torch.set_num_threads(threads)
        for conc in CONCURRENCY_LEVELS:
            r = await measure_concurrent(
                f"full_request_t{threads}_c{conc}",
                request_work,
                threads,
                conc,
            )
            results.append(r)
            print(f"{threads:>8} {conc:>5} {r.throughput_per_s:>9.2f} "
                  f"{r.latency_median_ms:>11.1f} {r.latency_p95_ms:>9.1f} "
                  f"{r.latency_max_ms:>9.1f}")
        print()

    # ---- closing drift anchor -------------------------------------------
    torch.set_num_threads(default_threads)
    anchor_close = await measure_concurrent(
        "anchor_close", request_work, default_threads, 4, notes="closing drift anchor"
    )
    results.append(anchor_close)

    drift = (
        abs(anchor_close.throughput_per_s - anchor_open.throughput_per_s)
        / anchor_open.throughput_per_s
        if anchor_open.throughput_per_s
        else 0.0
    )
    trustworthy = drift <= DRIFT_TOLERANCE
    print("=" * 78)
    print("DRIFT CHECK")
    print("=" * 78)
    print(f"  anchor open : {anchor_open.throughput_per_s:.2f} req/s")
    print(f"  anchor close: {anchor_close.throughput_per_s:.2f} req/s")
    print(f"  drift       : {drift*100:.1f}%  "
          f"{'OK' if trustworthy else 'DRIFTED — results unreliable'}")
    print()

    # ---- verdict ---------------------------------------------------------
    graded = [r for r in results if r.name.startswith("full_request")]
    best = max(graded, key=lambda r: r.throughput_per_s)
    print("=" * 78)
    print("BEST CONFIG BY THROUGHPUT")
    print("=" * 78)
    print(f"  torch_threads={best.torch_threads}  concurrency={best.concurrency}")
    print(f"  throughput : {best.throughput_per_s:.2f} req/s")
    print(f"  p95 latency: {best.latency_p95_ms:.0f}ms")
    print()

    # Compare the best against the default-thread config at the same concurrency,
    # which is the honest apples-to-apples "what does changing threads buy us".
    same_conc_default = next(
        (r for r in graded
         if r.torch_threads == default_threads and r.concurrency == best.concurrency),
        None,
    )
    if same_conc_default and same_conc_default.throughput_per_s:
        gain = best.throughput_per_s / same_conc_default.throughput_per_s
        print(f"  vs threads={default_threads} at same concurrency: {gain:.2f}x throughput")
        print()

    print("  Little's Law check for the 50-user / p95<10s target:")
    print(f"    required throughput = 50 / 10 = 5.00 req/s")
    print(f"    best measured       = {best.throughput_per_s:.2f} req/s")
    if best.throughput_per_s >= 5.0:
        print("    -> target reachable on CPU with this config")
    else:
        short = 5.0 / best.throughput_per_s
        print(f"    -> still {short:.1f}x short; needs workload cuts and/or GPU")
    print()

    payload = {
        "trustworthy": trustworthy,
        "drift_fraction": drift,
        "cpu_count": os.cpu_count(),
        "torch_default_threads": default_threads,
        "calls_per_config": CALLS_PER_CONFIG,
        "warmup_calls": WARMUP_CALLS,
        "results": [asdict(r) for r in results],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"raw results -> {args.output}")

    torch.set_num_threads(default_threads)
    return 0 if trustworthy else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
