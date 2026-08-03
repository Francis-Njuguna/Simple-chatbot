"""Benchmark retrieval against the repaired unified collection."""

import asyncio
import sys
import time

sys.path.insert(0, ".")

from backend.app.database.chroma import count_by_source_type  # noqa: E402
from backend.app.database.session import async_session_factory  # noqa: E402
from backend.app.rag.retriever import get_retriever  # noqa: E402

QUERIES = [
    "SMOWL camera not working",
    "reset student portal password",
    "what is smwol proctoring",
    "how to instal athenticator app",
    "How do I submit an assignment in Moodle?",
    "I cannot access my assignments",
    "LMS",
]


async def main() -> None:
    print("collection:", count_by_source_type())
    retriever = get_retriever()

    t0 = time.perf_counter()
    async with async_session_factory() as db:
        # Warm the model so the first query does not absorb the load cost.
        await retriever.retrieve("warmup", db=db)
        print(f"warmup (incl. model load): {time.perf_counter() - t0:.2f}s\n")

        for query in QUERIES:
            start = time.perf_counter()
            async with async_session_factory() as qdb:
                chunks, images = await retriever.retrieve(query, db=qdb)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"Q: {query!r}  [{elapsed:.0f}ms]  chunks={len(chunks)} images={len(images)}")
            if not chunks:
                print("   !! NO RESULTS")
            for chunk in chunks[:3]:
                text = (chunk.text or "")[:85].replace("\n", " ")
                print(f"   {chunk.score:.3f}  art={chunk.article_id}  {chunk.title!r}")
                print(f"          {text!r}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
