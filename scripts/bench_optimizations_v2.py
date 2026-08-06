"""Corrected optimization benchmark with proper warmup and drift detection.

Fixes methodology flaws in bench_optimizations.py:
  1. WARMUP: each configuration gets discarded warmup runs before timing, so
     cold-start cost (lazy graph construction, cache population, page faults)
     is not attributed to whichever config happened to run first.
  2. A/B/A ORDERING: baseline is measured again at the end. If the two baseline
     measurements disagree by more than a threshold, the machine drifted
     (thermal throttling, background load) and the whole run is untrustworthy.
  3. MEDIAN not MEAN: a single OS scheduling hiccup adds hundreds of ms to one
     trial; the mean carries it into the reported number, the median does not.
  4. MORE TRIALS: 15 instead of 10, since we are now discarding warmup.

Usage:
    python scripts/bench_optimizations_v2.py --output bench_opt_v2.json
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

WARMUP_RUNS = 3
TIMED_RUNS = 15
# If the closing baseline differs from the opening baseline by more than this
# fraction, the machine was not in a steady state and results are suspect.
DRIFT_TOLERANCE = 0.25


@dataclass
class Measurement:
    name: str
    config: dict[str, Any]
    median_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    p95_ms: float
    n_runs: int
    notes: str = ""


def measure(name: str, fn: Callable, config: dict[str, Any], notes: str = "") -> Measurement:
    """Warm up, then time. Returns median-centred stats."""
    for _ in range(WARMUP_RUNS):
        fn()

    times_ms: list[float] = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - start) * 1000.0)

    times_ms.sort()
    p95_idx = max(0, int(len(times_ms) * 0.95) - 1)

    return Measurement(
        name=name,
        config=config,
        median_ms=statistics.median(times_ms),
        mean_ms=statistics.fmean(times_ms),
        stdev_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        min_ms=times_ms[0],
        p95_ms=times_ms[p95_idx],
        n_runs=len(times_ms),
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="bench_opt_v2.json")
    args = parser.parse_args()

    import torch

    from backend.app.rag.embeddings import _SentenceTransformerBackend
    from backend.app.rag.reranker import CrossEncoderReranker

    print("=" * 72)
    print("OPTIMIZATION BENCHMARK v2 (warmup-corrected)")
    print("=" * 72)
    print(f"  cpu_count            : {__import__('os').cpu_count()}")
    print(f"  torch intra-op       : {torch.get_num_threads()}")
    print(f"  torch inter-op       : {torch.get_num_interop_threads()}")
    print(f"  warmup runs          : {WARMUP_RUNS} (discarded)")
    print(f"  timed runs           : {TIMED_RUNS}")
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
    results: list[Measurement] = []

    # ---- embedder ---------------------------------------------------------
    print("--- embedder: 1 query + 4 variants (5 texts) ---")

    torch.set_num_threads(default_threads)
    m = measure(
        "embed_baseline_open",
        lambda: embedder.embed_texts(variants),
        {"torch_threads": default_threads},
        "opening baseline",
    )
    results.append(m)
    print(f"  threads={default_threads} (baseline)  median {m.median_ms:7.1f}ms  p95 {m.p95_ms:7.1f}ms  stdev {m.stdev_ms:6.1f}")
    embed_baseline_open = m.median_ms

    for n in (1, 2, 4):
        torch.set_num_threads(n)
        m = measure(
            f"embed_threads_{n}",
            lambda: embedder.embed_texts(variants),
            {"torch_threads": n},
        )
        results.append(m)
        speedup = embed_baseline_open / m.median_ms if m.median_ms else 0.0
        print(f"  threads={n}             median {m.median_ms:7.1f}ms  p95 {m.p95_ms:7.1f}ms  stdev {m.stdev_ms:6.1f}   {speedup:.2f}x")

    # closing baseline — drift check
    torch.set_num_threads(default_threads)
    m = measure(
        "embed_baseline_close",
        lambda: embedder.embed_texts(variants),
        {"torch_threads": default_threads},
        "closing baseline (drift check)",
    )
    results.append(m)
    embed_baseline_close = m.median_ms
    print(f"  threads={default_threads} (re-check) median {m.median_ms:7.1f}ms  <- drift check")
    print()

    # ---- cross-encoder ----------------------------------------------------
    print("--- cross-encoder: 16 passages, 1 query form ---")

    torch.set_num_threads(default_threads)
    m = measure(
        "rerank_baseline_open",
        lambda: reranker.score(query, passages),
        {"torch_threads": default_threads, "n_passages": 16},
        "opening baseline",
    )
    results.append(m)
    print(f"  threads={default_threads} (baseline)  median {m.median_ms:7.1f}ms  p95 {m.p95_ms:7.1f}ms  stdev {m.stdev_ms:6.1f}")
    rerank_baseline_open = m.median_ms

    for n in (1, 2, 4):
        torch.set_num_threads(n)
        m = measure(
            f"rerank_threads_{n}",
            lambda: reranker.score(query, passages),
            {"torch_threads": n, "n_passages": 16},
        )
        results.append(m)
        speedup = rerank_baseline_open / m.median_ms if m.median_ms else 0.0
        print(f"  threads={n}             median {m.median_ms:7.1f}ms  p95 {m.p95_ms:7.1f}ms  stdev {m.stdev_ms:6.1f}   {speedup:.2f}x")

    # shortlist sweep at the best thread count found (measure at threads=1)
    print()
    print("--- cross-encoder: shortlist sweep at threads=1 ---")
    torch.set_num_threads(1)
    for size in (4, 8, 12, 16):
        subset = passages[:size]
        m = measure(
            f"rerank_shortlist_{size}",
            lambda s=subset: reranker.score(query, s),
            {"torch_threads": 1, "n_passages": size},
        )
        results.append(m)
        print(f"  shortlist={size:2d}          median {m.median_ms:7.1f}ms  p95 {m.p95_ms:7.1f}ms")

    # query-forms cost: genuinely 2 sequential scoring passes
    print()
    print("--- cross-encoder: query-forms cost at threads=1, shortlist=16 ---")
    torch.set_num_threads(1)

    def two_forms() -> None:
        reranker.score(query, passages)
        reranker.score("LMS password reset steps", passages)

    m = measure(
        "rerank_forms_1",
        lambda: reranker.score(query, passages),
        {"torch_threads": 1, "query_forms": 1, "n_passages": 16},
    )
    results.append(m)
    forms_1 = m.median_ms
    print(f"  query_forms=1         median {m.median_ms:7.1f}ms")

    m = measure(
        "rerank_forms_2",
        two_forms,
        {"torch_threads": 1, "query_forms": 2, "n_passages": 16},
        "two genuine scoring passes",
    )
    results.append(m)
    print(f"  query_forms=2         median {m.median_ms:7.1f}ms   (+{m.median_ms - forms_1:.0f}ms)")

    # closing baseline — drift check
    print()
    torch.set_num_threads(default_threads)
    m = measure(
        "rerank_baseline_close",
        lambda: reranker.score(query, passages),
        {"torch_threads": default_threads, "n_passages": 16},
        "closing baseline (drift check)",
    )
    results.append(m)
    rerank_baseline_close = m.median_ms
    print(f"  threads={default_threads} (re-check) median {m.median_ms:7.1f}ms  <- drift check")
    print()

    # ---- drift verdict ----------------------------------------------------
    torch.set_num_threads(default_threads)

    print("=" * 72)
    print("DRIFT CHECK")
    print("=" * 72)
    trustworthy = True
    for label, open_ms, close_ms in (
        ("embedder", embed_baseline_open, embed_baseline_close),
        ("cross-encoder", rerank_baseline_open, rerank_baseline_close),
    ):
        drift = abs(close_ms - open_ms) / open_ms if open_ms else 0.0
        ok = drift <= DRIFT_TOLERANCE
        trustworthy = trustworthy and ok
        verdict = "OK" if ok else "DRIFTED — results unreliable"
        print(f"  {label:14s} open {open_ms:7.1f}ms  close {close_ms:7.1f}ms  drift {drift*100:5.1f}%  {verdict}")
    print()

    if not trustworthy:
        print("  WARNING: baseline moved more than "
              f"{DRIFT_TOLERANCE*100:.0f}% between the opening and closing")
        print("  measurement. Close background work and re-run before trusting")
        print("  any speedup number in this report.")
        print()

    payload = {
        "trustworthy": trustworthy,
        "cpu_count": __import__("os").cpu_count(),
        "torch_default_threads": default_threads,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "measurements": [asdict(r) for r in results],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"raw results -> {args.output}")

    return 0 if trustworthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
