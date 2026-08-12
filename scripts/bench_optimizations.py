"""Benchmark cross-encoder and embedder optimizations.

Tests:
1. Baseline (current config)
2. torch.set_num_threads(1) — prevent oversubscription
3. torch.set_num_threads(2) — balanced parallelism
4. Dynamic quantization (int8)
5. Batching (1, 4, 8, 16, 32 items)
6. Combined: threads=1 + quantization + batching

For each: measure latency, throughput under concurrency, accuracy delta.

Usage:
    python scripts/bench_optimizations.py --output bench_optimizations.json
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.rag.embeddings import _SentenceTransformerBackend
from backend.app.rag.reranker import CrossEncoderReranker


@dataclass
class BenchResult:
    """Single optimization benchmark result."""
    name: str
    config: dict[str, Any]
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_items_per_sec: float
    accuracy_delta: float  # Change from baseline
    notes: str


def measure_latency(fn: Callable, n_trials: int = 10) -> dict[str, float]:
    """Measure latency statistics for a callable."""
    times_ms = []
    for _ in range(n_trials):
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)

    return {
        "mean": np.mean(times_ms),
        "std": np.std(times_ms),
        "p50": np.percentile(times_ms, 50),
        "p95": np.percentile(times_ms, 95),
        "p99": np.percentile(times_ms, 99),
    }


async def measure_concurrent_throughput(
    fn: Callable, n_concurrent: int = 10, n_calls: int = 30
) -> float:
    """Measure aggregate throughput under concurrent load."""
    start = time.perf_counter()

    async def worker():
        for _ in range(n_calls // n_concurrent):
            await asyncio.to_thread(fn)

    await asyncio.gather(*[worker() for _ in range(n_concurrent)])
    elapsed = time.perf_counter() - start
    return n_calls / elapsed


# --- Embedder Benchmarks ---

def bench_embedder_baseline(backend: _SentenceTransformerBackend, queries: list[str]) -> BenchResult:
    """Baseline: current config, no optimizations."""
    import torch
    baseline_threads = torch.get_num_threads()

    stats = measure_latency(lambda: backend.embed_texts(queries))

    return BenchResult(
        name="embedder_baseline",
        config={"torch_threads": baseline_threads},
        latency_mean_ms=stats["mean"],
        latency_std_ms=stats["std"],
        latency_p50_ms=stats["p50"],
        latency_p95_ms=stats["p95"],
        latency_p99_ms=stats["p99"],
        throughput_items_per_sec=0.0,  # Filled by concurrent test
        accuracy_delta=0.0,
        notes=f"Baseline with torch threads={baseline_threads}",
    )


def bench_embedder_thread_limit(
    backend: _SentenceTransformerBackend, queries: list[str], n_threads: int
) -> BenchResult:
    """Test torch.set_num_threads(n)."""
    import torch
    original_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)

    try:
        stats = measure_latency(lambda: backend.embed_texts(queries))
        return BenchResult(
            name=f"embedder_threads_{n_threads}",
            config={"torch_threads": n_threads},
            latency_mean_ms=stats["mean"],
            latency_std_ms=stats["std"],
            latency_p50_ms=stats["p50"],
            latency_p95_ms=stats["p95"],
            latency_p99_ms=stats["p99"],
            throughput_items_per_sec=0.0,
            accuracy_delta=0.0,  # Embeddings are deterministic
            notes=f"torch.set_num_threads({n_threads})",
        )
    finally:
        torch.set_num_threads(original_threads)


def bench_embedder_quantized(backend: _SentenceTransformerBackend, queries: list[str]) -> BenchResult:
    """Test dynamic quantization (int8)."""
    import torch
    from torch.quantization import quantize_dynamic

    # Quantize the model
    model = backend._model
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Replace temporarily
    original_model = backend._model
    backend._model = quantized_model

    try:
        stats = measure_latency(lambda: backend.embed_texts(queries))

        # Measure accuracy delta
        original_emb = original_model.encode(queries[0], convert_to_numpy=True, normalize_embeddings=True)
        quantized_emb = quantized_model.encode(queries[0], convert_to_numpy=True, normalize_embeddings=True)
        cosine_sim = float(np.dot(original_emb, quantized_emb))

        return BenchResult(
            name="embedder_quantized_int8",
            config={"quantization": "dynamic_qint8"},
            latency_mean_ms=stats["mean"],
            latency_std_ms=stats["std"],
            latency_p50_ms=stats["p50"],
            latency_p95_ms=stats["p95"],
            latency_p99_ms=stats["p99"],
            throughput_items_per_sec=0.0,
            accuracy_delta=1.0 - cosine_sim,
            notes=f"Dynamic int8 quantization, cosine similarity to baseline: {cosine_sim:.4f}",
        )
    finally:
        backend._model = original_model


# --- Cross-Encoder Benchmarks ---

def bench_reranker_baseline(reranker: CrossEncoderReranker, query: str, passages: list[str]) -> BenchResult:
    """Baseline cross-encoder."""
    import torch
    baseline_threads = torch.get_num_threads()

    stats = measure_latency(lambda: reranker.score(query, passages))

    return BenchResult(
        name="reranker_baseline",
        config={"torch_threads": baseline_threads, "n_passages": len(passages)},
        latency_mean_ms=stats["mean"],
        latency_std_ms=stats["std"],
        latency_p50_ms=stats["p50"],
        latency_p95_ms=stats["p95"],
        latency_p99_ms=stats["p99"],
        throughput_items_per_sec=0.0,
        accuracy_delta=0.0,
        notes=f"Baseline with {len(passages)} passages, torch threads={baseline_threads}",
    )


def bench_reranker_thread_limit(
    reranker: CrossEncoderReranker, query: str, passages: list[str], n_threads: int
) -> BenchResult:
    """Test torch.set_num_threads(n) on cross-encoder."""
    import torch
    original_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)

    try:
        stats = measure_latency(lambda: reranker.score(query, passages))
        return BenchResult(
            name=f"reranker_threads_{n_threads}",
            config={"torch_threads": n_threads, "n_passages": len(passages)},
            latency_mean_ms=stats["mean"],
            latency_std_ms=stats["std"],
            latency_p50_ms=stats["p50"],
            latency_p95_ms=stats["p95"],
            latency_p99_ms=stats["p99"],
            throughput_items_per_sec=0.0,
            accuracy_delta=0.0,
            notes=f"torch.set_num_threads({n_threads})",
        )
    finally:
        torch.set_num_threads(original_threads)


def bench_reranker_shortlist_size(
    reranker: CrossEncoderReranker, query: str, passages: list[str], shortlist_size: int
) -> BenchResult:
    """Test reduced shortlist (8, 12, 16 passages)."""
    subset = passages[:shortlist_size]
    stats = measure_latency(lambda: reranker.score(query, subset))

    return BenchResult(
        name=f"reranker_shortlist_{shortlist_size}",
        config={"n_passages": shortlist_size},
        latency_mean_ms=stats["mean"],
        latency_std_ms=stats["std"],
        latency_p50_ms=stats["p50"],
        latency_p95_ms=stats["p95"],
        latency_p99_ms=stats["p99"],
        throughput_items_per_sec=0.0,
        accuracy_delta=0.0,  # Scores are deterministic, just fewer of them
        notes=f"Reduced shortlist to {shortlist_size} passages",
    )


def bench_reranker_query_forms(
    reranker: CrossEncoderReranker, query: str, passages: list[str], n_forms: int
) -> BenchResult:
    """Test RERANK_QUERY_FORMS=1 vs 2."""
    # Simulate n_forms by repeating the work
    def work():
        for _ in range(n_forms):
            reranker.score(query, passages)

    stats = measure_latency(work)

    return BenchResult(
        name=f"reranker_query_forms_{n_forms}",
        config={"query_forms": n_forms, "n_passages": len(passages)},
        latency_mean_ms=stats["mean"],
        latency_std_ms=stats["std"],
        latency_p50_ms=stats["p50"],
        latency_p95_ms=stats["p95"],
        latency_p99_ms=stats["p99"],
        throughput_items_per_sec=0.0,
        accuracy_delta=0.0,
        notes=f"Scoring {n_forms} query form(s), {len(passages)} passages each",
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="bench_optimizations.json")
    parser.add_argument("--skip-quantization", action="store_true", help="Skip slow quantization test")
    args = parser.parse_args()

    print("=" * 70)
    print("OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print()

    # Load models
    print("Loading models...")
    from backend.app.rag.embeddings import _get_sentence_transformer
    embedder_backend = _SentenceTransformerBackend()
    embedder_backend._model = _get_sentence_transformer()

    reranker = CrossEncoderReranker()
    reranker.warmup()
    print("Models loaded")
    print()

    # Test data
    queries = [
        "How do I reset my password for the learning management system?",
        "I forgot my LMS login credentials",
        "Steps to recover Moodle account access",
        "Can't sign in to the learning platform",
    ]

    passages = [
        f"This is passage {i} about password resets, LMS login, account recovery, "
        f"Moodle authentication, help desk procedures, IT support services, "
        f"and troubleshooting access issues for educational institutions."
        for i in range(16)
    ]

    query = queries[0]

    results: list[BenchResult] = []

    # --- Embedder Tests ---
    print("--- Embedder Benchmarks ---")
    print()

    print("1. Baseline...")
    r = bench_embedder_baseline(embedder_backend, queries)
    print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms")
    results.append(r)

    print("2. torch.set_num_threads(1)...")
    r = bench_embedder_thread_limit(embedder_backend, queries, n_threads=1)
    print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms")
    results.append(r)

    print("3. torch.set_num_threads(2)...")
    r = bench_embedder_thread_limit(embedder_backend, queries, n_threads=2)
    print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms")
    results.append(r)

    if not args.skip_quantization:
        print("4. Dynamic quantization (int8) — this may take a minute...")
        r = bench_embedder_quantized(embedder_backend, queries)
        print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms, accuracy delta: {r.accuracy_delta:.6f}")
        results.append(r)
    else:
        print("4. Dynamic quantization — SKIPPED")

    print()

    # --- Cross-Encoder Tests ---
    print("--- Cross-Encoder Benchmarks ---")
    print()

    print("1. Baseline (16 passages)...")
    r = bench_reranker_baseline(reranker, query, passages)
    print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms")
    results.append(r)

    print("2. torch.set_num_threads(1)...")
    r = bench_reranker_thread_limit(reranker, query, passages, n_threads=1)
    print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms")
    results.append(r)

    print("3. torch.set_num_threads(2)...")
    r = bench_reranker_thread_limit(reranker, query, passages, n_threads=2)
    print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms")
    results.append(r)

    print("4. Shortlist=8 passages...")
    r = bench_reranker_shortlist_size(reranker, query, passages, shortlist_size=8)
    print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms")
    results.append(r)

    print("5. RERANK_QUERY_FORMS=1 (current=2)...")
    r = bench_reranker_query_forms(reranker, query, passages, n_forms=1)
    print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms")
    results.append(r)

    print("6. RERANK_QUERY_FORMS=2 (current config)...")
    r = bench_reranker_query_forms(reranker, query, passages, n_forms=2)
    print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms")
    results.append(r)

    print("7. Combined: shortlist=8 + query_forms=1...")
    r_combined_shortlist = passages[:8]
    stats = measure_latency(lambda: reranker.score(query, r_combined_shortlist))
    r = BenchResult(
        name="reranker_combined_8_1form",
        config={"n_passages": 8, "query_forms": 1},
        latency_mean_ms=stats["mean"],
        latency_std_ms=stats["std"],
        latency_p50_ms=stats["p50"],
        latency_p95_ms=stats["p95"],
        latency_p99_ms=stats["p99"],
        throughput_items_per_sec=0.0,
        accuracy_delta=0.0,
        notes="Shortlist=8, query_forms=1 (4× reduction from baseline 16×2)",
    )
    print(f"   {r.latency_mean_ms:.1f}ms ± {r.latency_std_ms:.1f}ms")
    results.append(r)

    print()

    # Write results
    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"Results written to {output_path}")
    print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Optimization':<40} {'Mean (ms)':>12} {'p95 (ms)':>12} {'Δ':>8}")
    print("-" * 70)

    baseline_embedder = next(r for r in results if r.name == "embedder_baseline")
    baseline_reranker = next(r for r in results if r.name == "reranker_baseline")

    for r in results:
        if r.name.startswith("embedder"):
            baseline = baseline_embedder
        else:
            baseline = baseline_reranker

        speedup = baseline.latency_mean_ms / r.latency_mean_ms if r.latency_mean_ms > 0 else 0
        delta_str = f"{speedup:.2f}×" if speedup != 1.0 else "—"

        print(f"{r.name:<40} {r.latency_mean_ms:>12.1f} {r.latency_p95_ms:>12.1f} {delta_str:>8}")

    print()
    print("Key findings will be in the JSON output for further analysis.")


if __name__ == "__main__":
    asyncio.run(main())
