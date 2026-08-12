"""End-to-end latency benchmark: retrieval + LLM + total.

Measures what the user actually waits for, against the <10s objective. Runs the
real RAGService against the real Postgres + Chroma + LLM, so the numbers include
network latency to the model provider.

Import ordering: the embedding model is loaded before anything touches Chroma —
see the docstring in scripts/reingest.py for why that matters on Windows.

Usage:
    ./.venv/Scripts/python.exe -u scripts/bench_e2e.py
    ./.venv/Scripts/python.exe -u scripts/bench_e2e.py --no-llm   # retrieval only
"""

import asyncio
import logging
import statistics
import sys
import time

sys.path.insert(0, ".")

QUERIES = [
    "How do I reset my student portal password?",
    "What is SMOWL proctoring?",
    "smwol camera not working",                      # typo + real issue
    "How do I set up Microsoft Authenticator?",
    "I cannot access my assignments",                # vague, no exact tokens
    "How do I log in to the LMS?",
    "moddle assignments",                            # typo
    "How do I register for supplementary exams?",
    "What is the capital of France?",                # off-topic → must decline
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _pct(values: list[float], p: float) -> float:
    """Percentile via nearest-rank; fine for the small n we run here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[idx]


async def main() -> None:
    run_llm = "--no-llm" not in sys.argv
    verbose = "--verbose" in sys.argv
    _setup_logging(verbose)

    # --- Warmup: load every heavy singleton, embedding model FIRST ----------
    warm_start = time.perf_counter()

    from backend.app.rag.embeddings import get_embedding_service

    await get_embedding_service().embed_query_async("warmup")

    from backend.app.database.chroma import count_by_source_type
    from backend.app.rag.lexical import get_lexical_index
    from backend.app.rag.reranker import get_reranker

    counts = count_by_source_type()
    get_lexical_index().rebuild()
    get_reranker().warmup()
    get_reranker().score("warmup", ["warmup passage"])

    from backend.app.rag.retriever import get_retriever

    retriever = get_retriever()
    warm_ms = (time.perf_counter() - warm_start) * 1000

    print(f"collection : {counts}")
    print(f"llm        : {'ON' if run_llm else 'OFF (retrieval only)'}")
    print(f"warmup     : {warm_ms / 1000:.2f}s")
    print()

    from backend.app.database.session import async_session_factory

    if run_llm:
        from backend.app.services.rag_service import RAGService

    retrieval_ms: list[float] = []
    llm_ms: list[float] = []
    total_ms: list[float] = []
    rows: list[tuple[str, int, float, float, float, str]] = []

    for query in QUERIES:
        # A fresh session per query, as a real request would have. The retrieval
        # cache is bypassed by using distinct queries, so these are cold numbers.
        async with async_session_factory() as db:
            t0 = time.perf_counter()
            chunks, images, _processed = await retriever.retrieve(query, db=db)
            t_retrieval = (time.perf_counter() - t0) * 1000

            t_llm = 0.0
            answer = ""
            if run_llm:
                from backend.app.rag.llm import get_llm_service

                llm = get_llm_service()
                context = retriever.format_context(chunks)
                image_context = retriever.format_images(images)
                t1 = time.perf_counter()
                answer = await llm.generate_answer(
                    question=query,
                    context=context,
                    history="No prior conversation.",
                    images=image_context,
                )
                t_llm = (time.perf_counter() - t1) * 1000

        t_total = t_retrieval + t_llm
        retrieval_ms.append(t_retrieval)
        if run_llm:
            llm_ms.append(t_llm)
        total_ms.append(t_total)

        declined = any(
            marker in answer.lower()
            for marker in ("isn't covered", "is not covered", "outside", "not able to")
        )
        verdict = "DECLINED" if declined else ("answered" if answer else "-")
        rows.append((query, len(chunks), t_retrieval, t_llm, t_total, verdict))

        print(
            f"{t_total / 1000:6.2f}s total | ret {t_retrieval:7.0f}ms | "
            f"llm {t_llm:7.0f}ms | {len(chunks)} chunks, {len(images)} imgs | "
            f"{verdict:8s} | {query[:44]}"
        )

    print()
    print("=" * 78)
    print(f"{'stage':<12} {'mean':>9} {'p50':>9} {'p95':>9} {'max':>9}")
    for name, series in (
        ("retrieval", retrieval_ms),
        ("llm", llm_ms),
        ("total", total_ms),
    ):
        if not series:
            continue
        print(
            f"{name:<12} {statistics.mean(series):8.0f}ms "
            f"{_pct(series, 50):8.0f}ms {_pct(series, 95):8.0f}ms "
            f"{max(series):8.0f}ms"
        )

    print("=" * 78)
    worst = max(total_ms) / 1000
    over = [q for q, _, _, _, t, _ in rows if t > 10_000]
    print(f"target <10s : worst case {worst:.2f}s → {'PASS' if not over else 'FAIL'}")
    if over:
        for q in over:
            print(f"   over budget: {q}")

    # An off-topic query must decline; a covered one must not. Report it so a
    # latency win that silently broke answer quality cannot pass unnoticed.
    print()
    print("answer sanity:")
    for query, n_chunks, _, _, _, verdict in rows:
        print(f"   {verdict:8s} {n_chunks} chunk(s)  {query}")


if __name__ == "__main__":
    asyncio.run(main())
