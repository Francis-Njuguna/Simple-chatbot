"""CLI script to run knowledge base ingestion."""

import asyncio

from backend.app.database.session import async_session_factory, init_db
from backend.app.ingest.pipeline import IngestionPipeline
from backend.app.rag.embeddings import EmbeddingService
from backend.app.utils.logging import setup_logging


async def main() -> None:
    setup_logging()
    await init_db()
    async with async_session_factory() as session:
        pipeline = IngestionPipeline(session, EmbeddingService())
        result = await pipeline.run(force=False, include_images=True)
        await session.commit()
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
