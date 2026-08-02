"""CLI script for the optional LLM summary enrichment pass.

Ingestion already writes an extractive summary for every article, so this is
never required — it upgrades those summaries using the configured LLM. Safe to
re-run; by default it only touches articles whose summary is missing or older
than the article content.

    python scripts/enrich_summaries.py                 # only stale articles
    python scripts/enrich_summaries.py --limit 3       # trial run
    python scripts/enrich_summaries.py --force         # redo everything
    python scripts/enrich_summaries.py --article-id 42
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from backend.app.database.session import async_session_factory
from backend.app.services.enrichment_service import enrich_summaries
from backend.app.utils.logging import setup_logging


async def main(force: bool, limit: int | None, article_ids: list[str] | None) -> None:
    setup_logging()
    async with async_session_factory() as session:
        stats = await enrich_summaries(
            session, force=force, limit=limit, article_ids=article_ids
        )
        print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LLM article summaries (optional enrichment)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-summarise every article, even ones already up to date.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N articles — useful for a costed trial run.",
    )
    parser.add_argument(
        "--article-id",
        action="append",
        dest="article_ids",
        help="Restrict to this article id (repeatable). Ignores staleness.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.force, args.limit, args.article_ids))
