"""Per-stage latency breakdown for the chat pipeline, in-process.

Measures the stages the RAG request actually has, at a given concurrency, and
prints a stage table plus each stage's share of total latency.

Why in-process rather than through HTTP: the reported ~46s warm request has to
be attributed to a *stage*, and only the retrieval trace and the LLM call stats
carry that attribution. A black-box HTTP timing can only show the total.

Two properties this harness has on purpose:

* **A failed LLM call is not a fast success.** ``generate_answer`` returns its
  error text as the answer instead of raising, so a provider outage otherwise
  reads as a complete request with a suspiciously low LLM time. Answers are
  checked for the error marker and reported as failures.
* **Concurrency is real concurrency.** Requests are launched into one event
  loop with a task group, so CPU contention between concurrent embeddings and
  cross-encoder passes is exercised rather than averaged away.

    python scripts/diag_latency.py --concurrency 1
    python scripts/diag_latency.py --concurrency 1,5,10 --queries 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings  # noqa: E402
from backend.app.rag.llm import get_llm_service  # noqa: E402
from backend.app.rag.retriever import get_retriever  # noqa: E402
from backend.app.services.rag_service import RAGService  # noqa: E402
from backend.app.utils.timing import collect_timers  # noqa: E402

QUERIES = [
    "MFA",
    "how do I use MFA",
    "how do I set up Microsoft Authenticator?",
    "I forgot my MFA",
    "how do I log into Moodle?",
    "student email",
]

# generate_answer returns provider failures as answer text; see module docstring.
ERROR_MARKERS = ("could not generate an answer", "check the server logs")

# The pipeline is two-level, and mixing the levels double-counts: the trace's
# `search` wraps vector + bm25 + rerank, and RAGService's `retrieval` stage
# wraps all of those again. Only TOP_LEVEL sums to the request total; RETRIEVAL
# stages are a breakdown *inside* it.
TOP_LEVEL = [
    "session_history",   # DB scope: session row + history read + user message
    "embedding",         # SentenceTransformers, local CPU
    "retrieval",         # Chroma + BM25 + cross-encoder (see RETRIEVAL below)
    "hydration",         # DB scope: titles/urls/captions
    "context_build",     # prompt assembly + confidence
    "llm",               # the provider call
    "persist",           # DB scope: answer + analytics
    "other",             # response formatting and anything unattributed
]

RETRIEVAL = [
    "query_processing",  # normalization, synonyms, multi-query expansion
    "vector",            # ChromaDB, all variants
    "bm25",              # lexical index
    "rerank",            # cross-encoder, local CPU
]


def failed(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in ERROR_MARKERS)


async def one_request(query: str) -> dict[str, Any]:
    """Run a single chat request, returning its stage breakdown in ms."""
    service = RAGService()
    sink: list = []
    llm_stats: dict[str, Any] = {}

    retriever = service.retriever
    real_retrieve = retriever.retrieve
    real_generate = service.llm_service.generate_answer

    async def traced_retrieve(*a: Any, **kw: Any):
        kw["trace_sink"] = sink
        return await real_retrieve(*a, **kw)

    async def traced_generate(*a: Any, **kw: Any):
        kw["stats"] = llm_stats
        return await real_generate(*a, **kw)

    # Patched on the *instance* the service holds, not on the class: the
    # retriever and LLM service are process-wide singletons, so a class-level
    # patch would leak into every other concurrent request in this run.
    service.retriever = _Proxy(retriever, retrieve=traced_retrieve)
    service.llm_service = _Proxy(service.llm_service, generate_answer=traced_generate)

    started = time.perf_counter()
    with collect_timers() as timers:
        response = await service.chat(query)
    total_ms = (time.perf_counter() - started) * 1000.0

    # Top level: RAGService's own stages, which partition the request.
    top: dict[str, float] = dict(timers[0].stages) if timers else {}
    top["other"] = max(0.0, total_ms - sum(top.values()))

    # Inside `retrieval`: the retriever's trace. Kept separate so the two
    # levels are never summed together.
    detail: dict[str, float] = {}
    if sink:
        detail = {
            k: float(v)
            for k, v in sink[0].timings_ms.items()
            if k in RETRIEVAL
        }

    return {
        "query": query,
        "total_ms": total_ms,
        "stages": top,
        "retrieval_detail": detail,
        "confidence": response.confidence,
        "n_sources": len(response.sources),
        "answer_chars": len(response.answer),
        "llm_ok": llm_stats.get("ok", None),
        "prompt_chars": llm_stats.get("prompt_chars"),
        "output_tokens": llm_stats.get("output_tokens"),
        "llm_call_ms": llm_stats.get("llm_call_ms"),
        "failed": failed(response.answer),
    }


class _Proxy:
    """Delegate everything to ``target`` except the named overrides."""

    def __init__(self, target: Any, **overrides: Any) -> None:
        self.__dict__["_target"] = target
        self.__dict__["_overrides"] = overrides

    def __getattr__(self, name: str) -> Any:
        overrides = self.__dict__["_overrides"]
        if name in overrides:
            return overrides[name]
        return getattr(self.__dict__["_target"], name)


async def run_level(concurrency: int, queries: list[str]) -> dict[str, Any]:
    """Run ``concurrency`` requests simultaneously and aggregate their stages.

    Every request is launched before any is awaited, so the level really does
    hold N requests in flight — the point is to expose contention, which a
    sequential loop would hide.
    """
    picks = [queries[i % len(queries)] for i in range(concurrency)]

    started = time.perf_counter()
    results = await asyncio.gather(
        *(one_request(q) for q in picks), return_exceptions=True
    )
    wall_ms = (time.perf_counter() - started) * 1000.0

    errors = [r for r in results if isinstance(r, BaseException)]
    ok = [r for r in results if not isinstance(r, BaseException)]
    provider_failures = [r for r in ok if r["failed"]]

    stage_means: dict[str, float] = {}
    detail_means: dict[str, float] = {}
    if ok:
        for name in {n for r in ok for n in r["stages"]}:
            stage_means[name] = statistics.mean(
                r["stages"].get(name, 0.0) for r in ok
            )
        for name in {n for r in ok for n in r["retrieval_detail"]}:
            detail_means[name] = statistics.mean(
                r["retrieval_detail"].get(name, 0.0) for r in ok
            )

    totals = sorted(r["total_ms"] for r in ok)
    return {
        "concurrency": concurrency,
        "n": len(ok),
        "raised": [type(e).__name__ for e in errors],
        "provider_failures": len(provider_failures),
        "wall_ms": wall_ms,
        "throughput_rps": (len(ok) / (wall_ms / 1000.0)) if wall_ms else 0.0,
        "total_ms_mean": statistics.mean(totals) if totals else 0.0,
        "total_ms_p50": statistics.median(totals) if totals else 0.0,
        "total_ms_max": max(totals) if totals else 0.0,
        "stages_mean_ms": stage_means,
        "retrieval_detail_mean_ms": detail_means,
        "requests": ok,
    }


def print_level(level: dict[str, Any]) -> None:
    c = level["concurrency"]
    print(f"\n{'=' * 74}")
    print(f"CONCURRENCY {c}   n={level['n']}   wall={level['wall_ms'] / 1000:.1f}s   "
          f"throughput={level['throughput_rps']:.2f} req/s")
    if level["raised"]:
        print(f"  RAISED: {level['raised']}")
    if level["provider_failures"]:
        print(f"  PROVIDER FAILURES: {level['provider_failures']}/{level['n']} "
              f"(answers contain the LLM error text)")
    print(f"{'=' * 74}")
    print(f"  total  mean={level['total_ms_mean'] / 1000:6.2f}s  "
          f"p50={level['total_ms_p50'] / 1000:6.2f}s  "
          f"max={level['total_ms_max'] / 1000:6.2f}s")

    stages = level["stages_mean_ms"]
    total = level["total_ms_mean"] or 1.0
    print(f"\n  {'stage':<18} {'mean ms':>10} {'% of total':>12}")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 12}")
    for name in [s for s in TOP_LEVEL if s in stages]:
        ms = stages[name]
        print(f"  {name:<18} {ms:>10.0f} {100 * ms / total:>11.1f}%")
    leftover = [s for s in sorted(stages) if s not in TOP_LEVEL]
    for name in leftover:
        ms = stages[name]
        print(f"  {name + ' (?)':<18} {ms:>10.0f} {100 * ms / total:>11.1f}%")

    detail = level["retrieval_detail_mean_ms"]
    if detail:
        retrieval_ms = stages.get("retrieval", 0.0) or 1.0
        print(f"\n  within retrieval   {'mean ms':>10} {'% of retr.':>12}")
        print(f"  {'-' * 18} {'-' * 10} {'-' * 12}")
        for name in [s for s in RETRIEVAL if s in detail]:
            ms = detail[name]
            print(f"  {name:<18} {ms:>10.0f} {100 * ms / retrieval_ms:>11.1f}%")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concurrency", default="1",
        help="comma-separated levels, e.g. 1,5,10 (run in order)",
    )
    parser.add_argument("--queries", type=int, default=len(QUERIES))
    parser.add_argument("--json", default="", help="write raw results here")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="clear the retrieval cache between levels so retrieval is measured, "
             "not the cache",
    )
    args = parser.parse_args()

    settings = get_settings()
    llm = get_llm_service()
    print("configuration under test")
    print("-" * 74)
    for key, value in llm.describe().items():
        print(f"  {key:24} {value}")
    print(f"  {'retrieval_cache_ttl':24} {settings.retrieval_cache_ttl}")
    print(f"  {'rerank_shortlist':24} {settings.rerank_shortlist}")
    print(f"  {'multi_query_variants':24} {settings.multi_query_variants}")

    queries = QUERIES[: args.queries]
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    # Warm-up: the first request in a process pays for lazy model loads
    # (SentenceTransformers, the cross-encoder) and a cold HTTP connection.
    # Those are startup costs, not per-request latency, and the user's figure
    # is explicitly a *warm* request.
    print("\nwarming up (model load + first connection) ...")
    warm = await one_request(queries[0])
    print(f"  cold request: {warm['total_ms'] / 1000:.1f}s "
          f"(llm {warm['stages'].get('llm_call', 0) / 1000:.1f}s)")

    out = []
    for level in levels:
        if args.no_cache:
            get_retriever().clear_cache()
        result = await run_level(level, queries)
        print_level(result)
        out.append(result)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                [{k: v for k, v in r.items() if k != "requests"} for r in out],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    return 2 if any(r["provider_failures"] or r["raised"] for r in out) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
