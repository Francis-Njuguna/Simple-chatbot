"""CLI script to run knowledge base ingestion."""

import asyncio
import sys
from pathlib import Path

# Allow running from the project root without the package being installed.
# Inside Docker this is redundant (pip install -e . handles it) but keeps
# the script working for local runs too — same pattern as verify_db_credentials.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
