"""Is retrieve_text's cost a one-time model load or per-query? Time warm calls."""

import sys
import time

sys.path.insert(0, ".")

import anyio  # noqa: E402


async def main():
    from backend.app.rag.retriever import get_retriever
    from backend.app.rag.reranker import get_reranker

    r = get_retriever()

    # Force both models to load BEFORE any timing.
    t = time.perf_counter()
    await r.embedding_service.embed_query_async("warmup")
    print(f"embedding model load+warm: {time.perf_counter() - t:.2f}s", flush=True)

    t = time.perf_counter()
    rr = get_reranker()
    rr.score("warmup query", ["warmup passage"])
    print(f"reranker model load+warm : {time.perf_counter() - t:.2f}s", flush=True)

    # Now everything is warm. These numbers are the real per-query latency.
    print("\n--- warm per-query latency ---", flush=True)
    for q in ["LMS login", "moddle login", "smowl cam", "Student email", "MFA"]:
        t = time.perf_counter()
        chunks = await r.retrieve_text(q)
        print(f"  {q!r:20s} {time.perf_counter() - t:6.2f}s  {len(chunks)} chunks", flush=True)

    # And how much of it is the reranker alone?
    print("\n--- reranker cost by batch size ---", flush=True)
    passage = "To log into Moodle, open the LMS portal and enter your credentials. " * 4
    for n in (1, 4, 8, 16, 32):
        t = time.perf_counter()
        rr.score("How do I log into Moodle?", [passage] * n)
        print(f"  {n:3d} passages: {time.perf_counter() - t:6.2f}s", flush=True)


anyio.run(main)
