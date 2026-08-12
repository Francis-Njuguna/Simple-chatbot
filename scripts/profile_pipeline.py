"""True per-stage cost breakdown of the retrieval pipeline.

WHY
---
retriever.retrieve() only instruments three stages (vector_ms, bm25_ms,
rerank_ms). Baseline median retrieval is ~2.0s while rerank accounts for
roughly 0.7-1.0s of it, so ~half the wall time is in stages nothing measures.
Optimising rerank harder is pointless if the unmeasured half dominates.

cProfile cannot answer this: every expensive stage runs inside an
anyio.to_thread worker, and cProfile only instruments the calling thread. So
this monkeypatches the real call boundaries with thread-safe accumulators,
which captures worker-thread time correctly.

Stages are attributed by WALL time inside the call. Because some stages run
concurrently (text and image retrieval overlap), the per-stage sum can exceed
end-to-end wall time; that is expected and is itself informative — it tells you
how much overlap you are already getting.

Usage:
    ./.venv/Scripts/python.exe -u scripts/profile_pipeline.py
    ./.venv/Scripts/python.exe -u scripts/profile_pipeline.py --queries 12
"""

import argparse
import asyncio
import logging
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent.parent))

_lock = threading.Lock()
_totals: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)


def _record(stage: str, ms: float) -> None:
    with _lock:
        _totals[stage] += ms
        _counts[stage] += 1


def _wrap(obj, attr: str, stage: str):
    """Wrap a callable attribute so its wall time is accumulated under `stage`."""
    original = getattr(obj, attr)

    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return original(*a, **kw)
        finally:
            _record(stage, (time.perf_counter() - t0) * 1000.0)

    setattr(obj, attr, wrapper)
    return original


def _wrap_async(obj, attr: str, stage: str):
    original = getattr(obj, attr)

    async def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return await original(*a, **kw)
        finally:
            _record(stage, (time.perf_counter() - t0) * 1000.0)

    setattr(obj, attr, wrapper)
    return original


def _setup_logging() -> None:
    logging.basicConfig(level=logging.ERROR, stream=sys.stdout, force=True)
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "urllib3", "openai",
                  "backend.app.rag.retriever", "backend.app.rag.lexical"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=16)
    args = parser.parse_args()

    _setup_logging()

    from scripts.eval_set import EVAL_QUERIES

    from backend.app.rag.embeddings import get_embedding_service

    await get_embedding_service().embed_query_async("warmup")

    from backend.app.rag.lexical import get_lexical_index
    from backend.app.rag.reranker import get_reranker

    get_lexical_index().rebuild()
    get_reranker().warmup()
    get_reranker().score("warmup", ["warmup passage"])

    from backend.app.database.session import async_session_factory
    from backend.app.rag.retriever import get_retriever

    retriever = get_retriever()

    # ---- instrument the real boundaries ---------------------------------
    import backend.app.database.chroma as chroma_mod
    import backend.app.rag.query_processing as qp_mod
    import backend.app.rag.retriever as retr_mod

    emb = get_embedding_service()
    _wrap(emb, "embed_query", "embed")
    _wrap(emb, "embed_texts", "embed")
    _wrap(emb._backend, "embed_query", "embed")
    _wrap(emb._backend, "embed_texts", "embed")

    _wrap(chroma_mod, "query_text_collection", "chroma_vector_search")
    _wrap(chroma_mod, "query_image_collection", "chroma_image_search")
    if hasattr(chroma_mod, "fetch_text_chunks_by_id"):
        _wrap(chroma_mod, "fetch_text_chunks_by_id", "chroma_fetch_by_id")
    # retriever imported these by name, so patch its module globals too
    for name, stage in (
        ("query_text_collection", "chroma_vector_search"),
        ("query_image_collection", "chroma_image_search"),
        ("fetch_text_chunks_by_id", "chroma_fetch_by_id"),
    ):
        if hasattr(retr_mod, name):
            _wrap(retr_mod, name, stage)

    _wrap(get_lexical_index(), "search", "bm25_search")
    _wrap(get_reranker(), "score", "rerank")
    _wrap(qp_mod, "process_query", "query_preprocess")
    if hasattr(retr_mod, "process_query"):
        _wrap(retr_mod, "process_query", "query_preprocess")
    if hasattr(retr_mod, "rewrite_query"):
        _wrap(retr_mod, "rewrite_query", "query_rewrite")

    # metadata hydration from Postgres
    from backend.app.services import metadata_service
    for attr in dir(metadata_service):
        if attr.startswith("get_") or attr.startswith("fetch_"):
            fn = getattr(metadata_service, attr)
            if callable(fn) and asyncio.iscoroutinefunction(fn):
                _wrap_async(metadata_service, attr, "pg_hydrate")

    queries = [q for q, _, kind in EVAL_QUERIES if kind != "offtopic"][:args.queries]

    print("=" * 74)
    print("PIPELINE STAGE PROFILE")
    print("=" * 74)
    print(f"  queries: {len(queries)}")
    print(f"  config : shortlist={retriever.settings.rerank_shortlist} "
          f"forms={retriever.settings.rerank_query_forms} "
          f"pool={retriever.settings.retrieval_candidate_pool} "
          f"variants={retriever.settings.multi_query_variants}")
    print()

    # warm one query so first-call costs are not attributed
    async with async_session_factory() as db:
        await retriever.retrieve(queries[0], db=db)
    with _lock:
        _totals.clear()
        _counts.clear()

    wall: list[float] = []
    for q in queries:
        async with async_session_factory() as db:
            t0 = time.perf_counter()
            await retriever.retrieve(q, db=db)
            wall.append((time.perf_counter() - t0) * 1000.0)

    total_wall = sum(wall)
    print(f"{'stage':<24} {'total ms':>10} {'per query':>11} {'calls/q':>9} {'% wall':>8}")
    print("-" * 74)
    for stage, total in sorted(_totals.items(), key=lambda kv: -kv[1]):
        per_q = total / len(queries)
        calls_q = _counts[stage] / len(queries)
        pct = 100.0 * total / total_wall if total_wall else 0.0
        print(f"{stage:<24} {total:>10.0f} {per_q:>11.1f} {calls_q:>9.1f} {pct:>7.1f}%")
    print("-" * 74)
    accounted = sum(_totals.values())
    print(f"{'SUM OF STAGES':<24} {accounted:>10.0f} {accounted/len(queries):>11.1f}")
    print(f"{'END-TO-END WALL':<24} {total_wall:>10.0f} {total_wall/len(queries):>11.1f}")
    print()
    print(f"  wall per query: median {statistics.median(wall):.0f}ms  "
          f"mean {statistics.fmean(wall):.0f}ms  max {max(wall):.0f}ms")
    ratio = accounted / total_wall if total_wall else 0.0
    print(f"  stage-sum / wall = {ratio:.2f}  "
          f"({'>1 means stages overlap (good concurrency)' if ratio > 1.05 else 'stages are largely sequential'})")
    print()

    top = max(_totals.items(), key=lambda kv: kv[1]) if _totals else None
    if top:
        print("=" * 74)
        print(f"DOMINANT STAGE: {top[0]}  "
              f"({top[1]/len(queries):.0f}ms/query, {100*top[1]/total_wall:.0f}% of wall)")
        print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
