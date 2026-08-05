"""Concurrency checks for the shared retrieval singletons (Objective 12).

Three failure modes, all of which only appear under real parallelism:

1. **Thundering-herd rebuild** — the BM25 index is built lazily on the query
   path, so N simultaneous cold queries each ran a full Chroma scan plus BM25
   construction. Verified by counting how many times `rebuild` executes when
   many threads hit a cold index at once.
2. **Torn read** — `rebuild` used to mutate `_corpus_ids` and `_bm25`
   separately, so a concurrent `search` could score against the new corpus and
   map the result onto the old id list: wrong chunk ids, or IndexError when the
   corpus shrank. Verified by rebuilding continuously while searching.
3. **Cache races** — the retrieval cache is an OrderedDict mutated from every
   request. Verified by hammering get/put from many threads and checking the
   hit/miss counters and the size bound both survive.

Run:  ./.venv/Scripts/python.exe -u scripts/_test_concurrency.py
"""

import os
import sys
import threading
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, ".")

import logging  # noqa: E402

logging.basicConfig(level=logging.ERROR, force=True)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1 + 2: lexical index
# ---------------------------------------------------------------------------

def test_lexical() -> None:
    print("\nLexicalIndex")
    from backend.app.rag import lexical as lx

    index = lx.LexicalIndex()

    # -- 1. thundering herd -------------------------------------------------
    builds = {"n": 0}
    builds_lock = threading.Lock()
    real_rebuild = index.rebuild

    def counting_rebuild() -> None:
        with builds_lock:
            builds["n"] += 1
        real_rebuild()

    index.rebuild = counting_rebuild  # type: ignore[method-assign]

    start = threading.Barrier(16)
    errors: list[BaseException] = []

    def cold_query() -> None:
        try:
            start.wait()
            index.search("moodle login")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=cold_query) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check("16 cold queries raise nothing", not errors, repr(errors[:2]) if errors else "")
    check(
        "cold index built exactly once, not once per thread",
        builds["n"] == 1,
        f"rebuild ran {builds['n']}x",
    )
    index.rebuild = real_rebuild  # type: ignore[method-assign]

    # -- 2. torn read -------------------------------------------------------
    valid_ids = set(index._corpus_ids)
    stop = threading.Event()
    read_errors: list[BaseException] = []
    bad_ids: list[str] = []
    n_reads = {"n": 0}

    def rebuilder() -> None:
        while not stop.is_set():
            index.rebuild()

    def searcher() -> None:
        while not stop.is_set():
            try:
                for cid, _score in index.search("moodle login password exam"):
                    if cid not in valid_ids:
                        bad_ids.append(cid)
                n_reads["n"] += 1
            except BaseException as exc:  # noqa: BLE001
                read_errors.append(exc)
                return

    workers = [threading.Thread(target=rebuilder) for _ in range(2)]
    workers += [threading.Thread(target=searcher) for _ in range(6)]
    for t in workers:
        t.start()
    time.sleep(3.0)
    stop.set()
    for t in workers:
        t.join(timeout=30)

    check(
        "searches during concurrent rebuild raise nothing",
        not read_errors,
        f"{len(read_errors)} error(s): {read_errors[0]!r}" if read_errors else
        f"{n_reads['n']} searches interleaved with rebuilds",
    )
    check(
        "no chunk id from a torn corpus snapshot",
        not bad_ids,
        f"{len(bad_ids)} bad id(s)" if bad_ids else "",
    )


# ---------------------------------------------------------------------------
# 3: retrieval cache
# ---------------------------------------------------------------------------

def test_cache() -> None:
    print("\nRetrieval cache")
    from backend.app.rag.retriever import get_retriever

    retriever = get_retriever()
    retriever.clear_cache()
    capacity = retriever.settings.retrieval_cache_size

    errors: list[BaseException] = []
    start = threading.Barrier(12)

    def hammer(worker: int) -> None:
        try:
            start.wait()
            for i in range(400):
                key = (f"q{(worker * 7 + i) % 50}", "general")
                if retriever._cache_get(key) is None:
                    retriever._cache_put(key, [], [])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(w,)) for w in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = retriever.cache_stats()
    check("concurrent cache access raises nothing", not errors,
          repr(errors[:2]) if errors else "")
    check(
        "cache respects its capacity bound",
        stats["size"] <= capacity,
        f"size={stats['size']} capacity={capacity}",
    )
    check(
        "hit/miss counters account for every lookup",
        stats["hits"] + stats["misses"] == 12 * 400,
        f"hits={stats['hits']} misses={stats['misses']} "
        f"total={stats['hits'] + stats['misses']} expected={12 * 400}",
    )
    check(
        "warm cache actually hits (50 distinct keys, 4800 lookups)",
        stats["hit_rate"] > 0.8,
        f"hit_rate={stats['hit_rate']:.3f}",
    )
    retriever.clear_cache()


# ---------------------------------------------------------------------------
# 4: metrics registry
# ---------------------------------------------------------------------------

def test_metrics() -> None:
    print("\nMetricsRegistry")
    from backend.app.utils.metrics import MetricsRegistry

    registry = MetricsRegistry()
    errors: list[BaseException] = []
    start = threading.Barrier(10)

    def hammer() -> None:
        try:
            start.wait()
            for _ in range(500):
                registry.counter("test_total")
                registry.observe("test_ms", 12.5)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = registry.snapshot()
    check("concurrent metric recording raises nothing", not errors,
          repr(errors[:2]) if errors else "")
    check(
        "counter lost no increments under contention",
        snap["counters"].get("test_total") == 5000,
        f"got {snap['counters'].get('test_total')} expected 5000",
    )
    check(
        "histogram lost no observations under contention",
        snap["histograms"].get("test_ms", {}).get("count") == 5000,
        f"got {snap['histograms'].get('test_ms', {}).get('count')} expected 5000",
    )


# ---------------------------------------------------------------------------
# 5: concurrent end-to-end retrieval
# ---------------------------------------------------------------------------

async def test_parallel_retrieval() -> None:
    print("\nParallel retrieval (end-to-end)")
    import asyncio

    from backend.app.database.session import async_session_factory
    from backend.app.rag.retriever import get_retriever

    retriever = get_retriever()
    retriever.clear_cache()

    queries = [
        "How do I reset my student portal password?",
        "What is SMOWL proctoring?",
        "How do I log in to the LMS?",
        "How do I set up Microsoft Authenticator?",
        "How do I access my student email?",
        "moddle login",
        "2FA setup",
        "exam registration",
    ]

    async def one(query: str) -> tuple[str, list[str]]:
        async with async_session_factory() as db:
            chunks, _images, _processed = await retriever.retrieve(query, db=db)
        return query, [c.article_id for c in chunks]

    # Serial pass first: this is the ground truth each parallel run must match.
    retriever.clear_cache()
    serial: dict[str, list[str]] = {}
    t0 = time.perf_counter()
    for query in queries:
        q, ids = await one(query)
        serial[q] = ids
    serial_ms = (time.perf_counter() - t0) * 1000

    # Parallel pass, cold cache, 3 copies of each query in flight at once.
    retriever.clear_cache()
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[one(q) for q in queries * 3], return_exceptions=True
    )
    parallel_ms = (time.perf_counter() - t0) * 1000

    exceptions = [r for r in results if isinstance(r, BaseException)]
    check("24 concurrent retrievals raise nothing", not exceptions,
          repr(exceptions[:2]) if exceptions else "")

    mismatches = [
        (q, ids, serial[q])
        for q, ids in results
        if not isinstance((q, ids), BaseException) and ids != serial[q]
    ] if not exceptions else []
    check(
        "parallel results identical to serial results",
        not mismatches,
        f"{len(mismatches)} differ: {mismatches[0]}" if mismatches else
        f"{len(results)} retrievals agree",
    )
    print(f"       serial {len(queries)} queries: {serial_ms:.0f}ms  |  "
          f"parallel {len(results)} queries: {parallel_ms:.0f}ms  "
          f"({parallel_ms / len(results):.0f}ms/query)")


def main() -> None:
    import asyncio

    # Warm the heavy singletons first — cold-start cost is not what this
    # measures, and the embedding/Chroma import order matters.
    from backend.app.rag.embeddings import get_embedding_service

    asyncio.run(get_embedding_service().embed_query_async("warmup"))
    from backend.app.rag.reranker import get_reranker

    get_reranker().warmup()
    get_reranker().score("warmup", ["warmup passage"])

    test_lexical()
    test_cache()
    test_metrics()
    asyncio.run(test_parallel_retrieval())

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        sys.exit(1)
    print("all concurrency checks passed")


if __name__ == "__main__":
    main()
