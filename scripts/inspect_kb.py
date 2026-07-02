"""
inspect_kb.py
-------------
Inspect what is currently stored in the knowledge base.

Shows:
  1. ChromaDB — text chunk count and sample chunks
  2. ChromaDB — image embedding count and sample images
  3. Raw article files on disk (data/raw/)
  4. PostgreSQL — document metadata table summary

Run from the project root:
    python scripts/inspect_kb.py                  # summary only
    python scripts/inspect_kb.py --samples 5      # show 5 sample chunks
    python scripts/inspect_kb.py --full           # full detail on every article
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings
from backend.app.database.chroma import get_image_collection, get_text_collection

_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _header(title: str) -> None:
    width = 60
    print(f"\n{_BOLD}{_CYAN}{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}{_RESET}")

def _ok(label: str, value: object) -> None:
    print(f"  {_GREEN}•{_RESET} {_BOLD}{label}:{_RESET} {value}")


# ---------------------------------------------------------------------------
# 1. ChromaDB — Text chunks
# ---------------------------------------------------------------------------

def inspect_text_chunks(num_samples: int = 3, full: bool = False) -> None:
    _header("ChromaDB — Text Chunks")
    col = get_text_collection()
    count = col.count()
    _ok("Total chunks", count)

    if count == 0:
        print("  ⚠  No text chunks found. Run ingestion first:")
        print("       python scripts/ingest.py")
        return

    # Pull sample items
    n = count if full else min(num_samples, count)
    result = col.get(limit=n, include=["documents", "metadatas"])

    # Summarise unique articles and categories from ALL metadata
    all_meta = col.get(include=["metadatas"])["metadatas"] or []
    articles  = {m.get("article_id", "?") for m in all_meta}
    categories = {m.get("category", "?") for m in all_meta}

    _ok("Unique articles", len(articles))
    _ok("Unique categories", len(categories))
    print()
    print(f"  {_BOLD}Categories:{_RESET}")
    for cat in sorted(categories):
        print(f"    - {cat}")

    print()
    print(f"  {_BOLD}Sample chunks ({n}):{_RESET}")
    for i, (doc, meta) in enumerate(
        zip(result["documents"], result["metadatas"]), start=1
    ):
        print(f"\n  [{i}] Article : {meta.get('article_id')} — {meta.get('title', 'N/A')}")
        print(f"      Category: {meta.get('category', 'N/A')}")
        print(f"      URL     : {meta.get('url', 'N/A')}")
        snippet = doc[:300].replace("\n", " ")
        print(f"      Text    : {snippet}{'...' if len(doc) > 300 else ''}")


# ---------------------------------------------------------------------------
# 2. ChromaDB — Image embeddings
# ---------------------------------------------------------------------------

def inspect_images(num_samples: int = 3, full: bool = False) -> None:
    _header("ChromaDB — Image Embeddings")
    col = get_image_collection()
    count = col.count()
    _ok("Total images", count)

    if count == 0:
        print("  ⚠  No image embeddings found.")
        return

    n = count if full else min(num_samples, count)
    result = col.get(limit=n, include=["documents", "metadatas"])

    print(f"\n  {_BOLD}Sample images ({n}):{_RESET}")
    for i, (doc, meta) in enumerate(
        zip(result["documents"], result["metadatas"]), start=1
    ):
        print(f"\n  [{i}] File    : {meta.get('filename', 'N/A')}")
        print(f"      Article : {meta.get('article_id', 'N/A')}")
        print(f"      Category: {meta.get('category', 'N/A')}")
        print(f"      Caption : {meta.get('caption', 'N/A')}")
        print(f"      Alt text: {meta.get('alt_text', 'N/A')}")


# ---------------------------------------------------------------------------
# 3. Raw JSON files on disk
# ---------------------------------------------------------------------------

def inspect_raw_files(full: bool = False) -> None:
    _header("Raw Article Files on Disk")
    settings = get_settings()
    raw_dir = Path(settings.raw_data_dir)

    files = sorted(raw_dir.glob("article_*.json"))
    _ok("Raw files found", len(files))

    if not files:
        print("  ⚠  No raw files found. Run ingestion first:")
        print("       python scripts/ingest.py")
        return

    items_to_show = files if full else files[:5]
    print(f"\n  {_BOLD}Articles ({len(items_to_show)} shown):{_RESET}")
    for f in items_to_show:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            title    = data.get("title", "N/A")
            category = data.get("category", "N/A")
            url      = data.get("url", "N/A")
            chars    = len(data.get("text", ""))
            print(f"\n  • {f.name}")
            print(f"    Title   : {title}")
            print(f"    Category: {category}")
            print(f"    URL     : {url}")
            print(f"    Length  : {chars:,} characters")
        except Exception as e:
            print(f"  ✘ Could not read {f.name}: {e}")

    if not full and len(files) > 5:
        print(f"\n  ... and {len(files) - 5} more. Use --full to see all.")


# ---------------------------------------------------------------------------
# 4. PostgreSQL document_metadata table
# ---------------------------------------------------------------------------

def inspect_postgres() -> None:
    _header("PostgreSQL — document_metadata table")
    try:
        import asyncio
        import asyncpg

        settings = get_settings()

        async def _query() -> None:
            conn = await asyncpg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
                timeout=10,
            )
            rows = await conn.fetch(
                """
                SELECT article_id, title, category, url, chunk_count, ingested_at
                FROM document_metadata
                ORDER BY ingested_at DESC
                """
            )
            img_rows = await conn.fetch("SELECT COUNT(*) AS cnt FROM image_metadata")
            await conn.close()
            return rows, img_rows

        rows, img_rows = asyncio.run(_query())
        _ok("Documents in DB", len(rows))
        _ok("Images in DB", img_rows[0]["cnt"])

        if rows:
            total_chunks = sum(r["chunk_count"] for r in rows)
            _ok("Total chunks recorded", total_chunks)
            cats = {r["category"] for r in rows}
            _ok("Categories", ", ".join(sorted(cats)))

            print(f"\n  {_BOLD}All articles:{_RESET}")
            for r in rows:
                print(
                    f"\n  • [{r['article_id']}] {r['title']}"
                    f"\n    Category : {r['category']}"
                    f"\n    Chunks   : {r['chunk_count']}"
                    f"\n    Ingested : {r['ingested_at'].strftime('%Y-%m-%d %H:%M') if r['ingested_at'] else 'N/A'}"
                    f"\n    URL      : {r['url']}"
                )
    except ImportError:
        print("  ⚠  asyncpg not installed — skipping PostgreSQL check.")
        print("       pip install asyncpg")
    except Exception as e:
        print(f"  ⚠  Could not connect to PostgreSQL: {e}")
        print("       Is the database running?  docker compose up postgres -d")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect what is stored in the Amref Help Desk knowledge base."
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Number of sample chunks/images to display (default: 3)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Show every article and chunk (can be long)"
    )
    parser.add_argument(
        "--no-postgres", action="store_true",
        help="Skip the PostgreSQL check (useful if DB is not running)"
    )
    args = parser.parse_args()

    print(f"\n{_BOLD}{'=' * 60}")
    print("  Amref Help Desk — Knowledge Base Inspector")
    print(f"{'=' * 60}{_RESET}")

    inspect_raw_files(full=args.full)
    inspect_text_chunks(num_samples=args.samples, full=args.full)
    inspect_images(num_samples=args.samples, full=args.full)

    if not args.no_postgres:
        inspect_postgres()

    print(f"\n{_BOLD}{'=' * 60}{_RESET}\n")


if __name__ == "__main__":
    main()
