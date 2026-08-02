"""Metadata hydration — PostgreSQL is the source of truth.

Chroma stores only the identifiers and filter keys needed to *retrieve*
(``article_id``, ``chunk_id``, ``category``, ``source_type``…). Everything a
caller needs to *display* — title, url, summary, caption, filename,
static_path — is fetched here, by id, from PostgreSQL.

Caching
-------
Article/image metadata changes only at ingestion time, but a chat request needs
it on the hot path. A short-TTL in-process cache keeps the common case free
while still picking up a re-ingest within ``METADATA_CACHE_TTL`` seconds. The
cache is keyed by id and invalidated wholesale by the ingestion pipeline via
:func:`invalidate_metadata_cache`, so a re-ingest is visible immediately in the
process that ran it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import get_settings
from backend.app.database.models import DocumentMetadata, ImageMetadata
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ArticleMeta:
    """Display metadata for one knowledge-base article."""

    article_id: str
    title: str
    url: str
    category: Optional[str] = None
    summary: Optional[str] = None


@dataclass(frozen=True)
class ImageMeta:
    """Display metadata for one knowledge-base image."""

    image_id: str
    filename: str
    filepath: str
    static_path: Optional[str] = None
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    article_id: Optional[str] = None
    category: Optional[str] = None


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_article_cache: dict[str, tuple[float, ArticleMeta]] = {}
_image_cache: dict[str, tuple[float, ImageMeta]] = {}


def _ttl() -> int:
    return get_settings().metadata_cache_ttl


def _cache_get(cache: dict, key: str):
    entry = cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if expires_at < time.monotonic():
        # Expired — drop it so the dict cannot grow without bound.
        cache.pop(key, None)
        return None
    return value


def invalidate_metadata_cache() -> None:
    """Clear both caches — called by the ingestion pipeline after a write."""
    with _lock:
        _article_cache.clear()
        _image_cache.clear()
    logger.info("Metadata cache invalidated.")


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


async def get_articles(db: AsyncSession, article_ids: Iterable[str]) -> dict[str, ArticleMeta]:
    """Return ``{article_id: ArticleMeta}`` for the given ids.

    Cache hits are served without touching the database; only the misses go out
    in a single ``WHERE article_id IN (...)`` query.
    """
    wanted = [aid for aid in dict.fromkeys(article_ids) if aid]
    if not wanted:
        return {}

    resolved: dict[str, ArticleMeta] = {}
    missing: list[str] = []

    with _lock:
        for article_id in wanted:
            cached = _cache_get(_article_cache, article_id)
            if cached is not None:
                resolved[article_id] = cached
            else:
                missing.append(article_id)

    if missing:
        result = await db.execute(
            select(DocumentMetadata).where(DocumentMetadata.article_id.in_(missing))
        )
        rows = result.scalars().all()
        expires_at = time.monotonic() + _ttl()
        with _lock:
            for row in rows:
                meta = ArticleMeta(
                    article_id=row.article_id,
                    title=row.title,
                    url=row.url,
                    category=row.category,
                    summary=row.best_summary,
                )
                resolved[row.article_id] = meta
                _article_cache[row.article_id] = (expires_at, meta)

        found = {row.article_id for row in rows}
        for article_id in missing:
            if article_id not in found:
                # A vector whose Postgres row is gone (partial re-ingest, manual
                # delete). Retrieval still works; the caller falls back to
                # whatever Chroma carries.
                logger.warning("No Postgres metadata for article_id=%s", article_id)

    return resolved


async def get_images(db: AsyncSession, image_ids: Iterable[str]) -> dict[str, ImageMeta]:
    """Return ``{image_id: ImageMeta}`` for the given ids."""
    wanted = [iid for iid in dict.fromkeys(image_ids) if iid]
    if not wanted:
        return {}

    resolved: dict[str, ImageMeta] = {}
    missing: list[str] = []

    with _lock:
        for image_id in wanted:
            cached = _cache_get(_image_cache, image_id)
            if cached is not None:
                resolved[image_id] = cached
            else:
                missing.append(image_id)

    if missing:
        result = await db.execute(
            select(ImageMetadata).where(ImageMetadata.image_id.in_(missing))
        )
        rows = result.scalars().all()
        expires_at = time.monotonic() + _ttl()
        with _lock:
            for row in rows:
                meta = ImageMeta(
                    image_id=row.image_id,
                    filename=row.filename,
                    filepath=row.filepath,
                    static_path=row.static_path,
                    caption=row.caption,
                    alt_text=row.alt_text,
                    article_id=row.article_id,
                    category=row.category,
                )
                resolved[row.image_id] = meta
                _image_cache[row.image_id] = (expires_at, meta)

        found = {row.image_id for row in rows}
        for image_id in missing:
            if image_id not in found:
                logger.warning("No Postgres metadata for image_id=%s", image_id)

    return resolved
