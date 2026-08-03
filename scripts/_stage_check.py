"""Does loading the sentence-transformer inside asyncio + an open asyncpg
session trigger the access violation?"""

import asyncio
import sys

sys.path.insert(0, ".")


async def main() -> None:
    from backend.app.database.chroma import count_by_source_type
    from backend.app.database.session import async_session_factory

    print("1 chroma ->", count_by_source_type(), flush=True)

    print("2 open async DB session", flush=True)
    async with async_session_factory() as db:
        from sqlalchemy import text

        await db.execute(text("SELECT 1"))
        print("3 DB query ok", flush=True)

        print("4 load sentence-transformer inside session", flush=True)
        from backend.app.rag.embeddings import EmbeddingService

        svc = EmbeddingService()
        svc.embed_texts(["hello world"])
        print("5 SURVIVED embedding inside session", flush=True)

    print("6 ALL OK", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
