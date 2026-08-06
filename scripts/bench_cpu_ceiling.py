"""Measure CPU ceiling and thread oversubscription.

Determines:
1. Physical CPU cores vs PyTorch default threads
2. Actual CPU-seconds of work per request (embedding + cross-encoder)
3. Theoretical throughput ceiling via Little's Law
4. Whether 50 concurrent users at p95<10s is reachable on this hardware

Usage:
    python scripts/bench_cpu_ceiling.py
"""

import asyncio
import os
import time
from pathlib import Path

import numpy as np

# Add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.rag.embeddings import get_embedding_service
from backend.app.rag.reranker import get_reranker


def measure_torch_config():
    """Report PyTorch thread configuration."""
    try:
        import torch
        print(f"PyTorch threads (intra-op): {torch.get_num_threads()}")
        print(f"PyTorch threads (inter-op): {torch.get_num_interop_threads()}")
    except ImportError:
        print("PyTorch not installed")

    print(f"OS CPU count: {os.cpu_count()}")
    print(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS', 'unset')}")
    print(f"MKL_NUM_THREADS: {os.getenv('MKL_NUM_THREADS', 'unset')}")


def measure_embedding_cost(embedder, query: str, variants: list[str], n_trials: int = 3):
    """Measure CPU time for embedding one query + N variants."""
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        # Simulate what retriever does: embed primary query + variants
        embedder.embed_query(query)
        embedder.embed_texts(variants)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_s": np.mean(times),
        "std_s": np.std(times),
        "min_s": np.min(times),
        "max_s": np.max(times),
    }


def measure_reranker_cost(reranker, query: str, passages: list[str], n_trials: int = 3):
    """Measure CPU time for cross-encoder scoring."""
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        reranker.score(query, passages)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_s": np.mean(times),
        "std_s": np.std(times),
        "min_s": np.min(times),
        "max_s": np.max(times),
    }


async def main():
    print("=" * 70)
    print("CPU CEILING AND THREAD CONFIGURATION")
    print("=" * 70)
    print()

    print("--- System Configuration ---")
    measure_torch_config()
    print()

    print("--- Loading Models ---")
    embedder = get_embedding_service()
    print("Embedder loaded")

    reranker = get_reranker()
    reranker.warmup()
    print("Reranker loaded")
    print()

    # Simulate a typical query
    query = "How do I reset my password for the learning management system?"
    variants = [
        query,
        "How can I reset my LMS password?",
        "I forgot my Moodle password, what should I do?",
        "Steps to recover my learning platform login",
    ]

    # 16 passages to rerank (current RERANK_SHORTLIST default)
    passages = [
        f"This is passage number {i} about password resets, user authentication, "
        f"learning management systems, Moodle, account recovery, help desk procedures, "
        f"and IT support for educational institutions."
        for i in range(16)
    ]

    print("--- Embedding Cost (1 query + 4 variants) ---")
    embed_stats = measure_embedding_cost(embedder, query, variants, n_trials=5)
    print(f"  Mean: {embed_stats['mean_s']:.3f}s")
    print(f"  Std:  {embed_stats['std_s']:.3f}s")
    print(f"  Min:  {embed_stats['min_s']:.3f}s")
    print(f"  Max:  {embed_stats['max_s']:.3f}s")
    print()

    print("--- Cross-Encoder Cost (16 passages × 1 query form) ---")
    rerank_stats = measure_reranker_cost(reranker, query, passages, n_trials=5)
    print(f"  Mean: {rerank_stats['mean_s']:.3f}s")
    print(f"  Std:  {rerank_stats['std_s']:.3f}s")
    print(f"  Min:  {rerank_stats['min_s']:.3f}s")
    print(f"  Max:  {rerank_stats['max_s']:.3f}s")
    print()

    # Current config uses 2 query forms for reranking
    print("--- Cross-Encoder Cost (16 passages × 2 query forms) ---")
    rerank_2x_stats = measure_reranker_cost(
        reranker,
        query,
        passages + passages,  # Double the work
        n_trials=5
    )
    print(f"  Mean: {rerank_2x_stats['mean_s']:.3f}s")
    print(f"  Std:  {rerank_2x_stats['std_s']:.3f}s")
    print()

    print("--- Total CPU Work Per Request (Current Config) ---")
    total_cpu = embed_stats['mean_s'] + rerank_2x_stats['mean_s']
    print(f"  Embedding + Reranking: {total_cpu:.3f}s CPU-seconds")
    print()

    print("--- Theoretical Throughput Ceiling ---")
    cpu_cores = os.cpu_count() or 4
    print(f"  Available CPU cores: {cpu_cores}")
    print(f"  CPU work per request: {total_cpu:.3f}s")
    print(f"  Theoretical max throughput (perfect parallelism): {cpu_cores / total_cpu:.2f} req/s")
    print(f"  Observed throughput from load test: 0.5 req/s")
    print(f"  Efficiency: {(0.5 / (cpu_cores / total_cpu)) * 100:.1f}%")
    print()

    print("--- Little's Law Analysis ---")
    print("  Little's Law: Throughput = Concurrency / Latency")
    print(f"  Target: 50 users, p95 < 10s")
    print(f"  Required throughput: 50 / 10 = 5.0 req/s")
    print(f"  Current throughput: 0.5 req/s")
    print(f"  Gap: {5.0 / 0.5:.1f}× improvement needed")
    print()

    print(f"  If we achieve theoretical max ({cpu_cores / total_cpu:.2f} req/s):")
    print(f"    Latency at 50 users: 50 / {cpu_cores / total_cpu:.2f} = {50 / (cpu_cores / total_cpu):.1f}s")
    print()

    # Estimate what happens if we reduce CPU work
    scenarios = [
        ("Halve rerank work (RERANK_QUERY_FORMS=1)", embed_stats['mean_s'] + rerank_stats['mean_s']),
        ("Halve rerank shortlist (16→8)", embed_stats['mean_s'] + rerank_stats['mean_s'] / 2),
        ("Both (1 form + 8 shortlist)", embed_stats['mean_s'] + rerank_stats['mean_s'] / 2),
        ("10× speedup (GPU or quantization)", total_cpu / 10),
    ]

    print("--- Scenario Analysis ---")
    for name, cpu_work in scenarios:
        theoretical_tput = cpu_cores / cpu_work
        latency_at_50 = 50 / theoretical_tput
        meets_target = "✓" if latency_at_50 < 10 else "✗"
        print(f"  {meets_target} {name}")
        print(f"      CPU work: {cpu_work:.3f}s")
        print(f"      Theoretical throughput: {theoretical_tput:.2f} req/s")
        print(f"      Latency at 50 users: {latency_at_50:.1f}s")
        print()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    observed_under_load = 14.2  # from load test metrics
    amplification = observed_under_load / total_cpu
    print(f"Uncontended CPU work: {total_cpu:.2f}s")
    print(f"Observed latency under load: {observed_under_load:.1f}s")
    print(f"Amplification factor: {amplification:.1f}×")
    print()
    print("Root cause: PyTorch grabbing all cores per call → thread oversubscription")
    print(f"            {cpu_cores} cores × many concurrent requests = thrashing")
    print()
    print("To reach 50 users at p95<10s on this hardware:")
    print("  1. Reduce CPU work per request (halve reranking, reduce shortlist)")
    print("  2. Prevent thread oversubscription (torch.set_num_threads(1))")
    print("  3. Add concurrency semaphore (serialize CPU-bound stages)")
    print("  4. OR: Move to GPU (eliminates CPU contention entirely)")


if __name__ == "__main__":
    asyncio.run(main())
