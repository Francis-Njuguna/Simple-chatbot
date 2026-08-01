"""Optional background enrichment — LLM-generated article summaries.

Ingestion is deliberately independent of this module. Every article gets an
*extractive* summary inline (see ``ingest/summarizer.py``), which needs no model
endpoint and cannot fail. This pass upgrades those summaries to abstractive ones
by calling the configured LLM, and is safe to run, skip, re-run or interrupt:

* it only touches ``document_metadata.llm_summary`` / ``llm_summary_at``;
  ``summary`` (extractive) is never modified, so a bad enrichment run can be
  reverted by clearing ``llm_summary``;
* :attr:`DocumentMetadata.best_summary` prefers ``llm_summary`` and falls back
  to ``summary``, so retrieval keeps working whether or not this ever runs;
* work is claimed per-article and committed per-article, so an interrupted run
  keeps everything it finished;
* by default it skips articles whose summary is already newer than their last
  content update, making re-runs cheap.

Usage::

    from backend.app.services.enrichment_service import enrich_summaries
    stats = await enrich_summaries(db)                 # only what's stale
    stats = await enrich_summaries(db, force=True)     # redo everything
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import get_settings
from backend.app.database.models import DocumentMetadata
from backend.app.prompts.templates import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_TEMPLATE,
)
from backend.app.rag.llm import LLMService, get_llm_service
from backend.app.services.metadata_service import invalidate_metadata_cache
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Cap the text handed to the model. Help desk articles are short, and this keeps
# a pathological page from blowing the context window or the token bill.
_MAX_INPUT_CHARS = 12000

# An LLM that returns a couple of words has failed at the task even though the
# call succeeded; don't overwrite a decent extractive summary with it.
_MIN_SUMMARY_CHARS = 60


@dataclass
class EnrichmentStats:
    """Outcome of one enrichment run."""

    considered: int = 0
    enriched: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "considered": self.considered,
            "enriched": self.enriched,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": self.errors[:10],
        }


def _strip_preamble(text: str) -> str:
    """Drop the wrapper models like to add around a requested summary."""
    cleaned = text.strip()
    # Occasional fenced output despite the prompt asking for plain text.
    if cleaned.startswith("```"):
        lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    for prefix in ("Summary:", "Overview:", "Here is the summary:", "Here's the summary:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    # Collapse to a single paragraph — this is injected into a prompt, not shown.
    return " ".join(cleaned.split())


async def _select_candidates(
    db: AsyncSession,
    *,
    force: bool,
    limit: Optional[int],
    article_ids: Optional[list[str]],
) -> list[DocumentMetadata]:
    stmt = select(DocumentMetadata).order_by(DocumentMetadata.article_id)

    if article_ids:
        stmt = stmt.where(DocumentMetadata.article_id.in_(article_ids))
    elif not force:
        # Stale = never enriched, or enriched before the article last changed.
        stmt = stmt.where(
            or_(
                DocumentMetadata.llm_summary.is_(None),
                DocumentMetadata.llm_summary_at.is_(None),
                DocumentMetadata.llm_summary_at < DocumentMetadata.updated_at,
            )
        )

    if limit is not None:
        stmt = stmt.limit(limit)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def enrich_summaries(
    db: AsyncSession,
    *,
    force: bool = False,
    limit: Optional[int] = None,
    article_ids: Optional[list[str]] = None,
    llm_service: LLMService | None = None,
) -> dict[str, object]:
    """Generate abstractive summaries for articles that need one.

    Args:
        db: Session to read articles from and write summaries back to.
        force: Re-summarise every article, even if already up to date.
        limit: Process at most this many articles (useful for a trial run).
        article_ids: Restrict to these articles, ignoring staleness.
        llm_service: Injectable for tests; defaults to the shared singleton.

    Returns:
        A dict of counters — see :meth:`EnrichmentStats.as_dict`.

    A per-article failure is logged and counted, not raised: one unavailable
    article must not abandon the rest of the batch.
    """
    settings = get_settings()
    llm = llm_service or get_llm_service()
    stats = EnrichmentStats()

    articles = await _select_candidates(
        db, force=force, limit=limit, article_ids=article_ids
    )
    stats.considered = len(articles)
    if not articles:
        logger.info("Summary enrichment: nothing to do.")
        return stats.as_dict()

    logger.info(
        "Summary enrichment: %d article(s) to process (force=%s, provider=%s)",
        len(articles),
        force,
        settings.llm_provider,
    )

    for article in articles:
        text = (article.raw_content or "").strip()
        if not text:
            # Nothing to summarise; the extractive summary (if any) still stands.
            stats.skipped += 1
            logger.warning(
                "Skipping article %s — no raw_content stored.", article.article_id
            )
            continue

        prompt = SUMMARY_USER_TEMPLATE.format(
            title=article.title or "Untitled",
            category=article.category or "General",
            text=text[:_MAX_INPUT_CHARS],
        )

        try:
            raw = await llm.complete(SUMMARY_SYSTEM_PROMPT, prompt)
        except Exception as exc:  # noqa: BLE001 — one article must not kill the run
            stats.failed += 1
            message = f"{article.article_id}: {type(exc).__name__}: {exc}"
            stats.errors.append(message)
            logger.exception("Summary enrichment failed for article %s", article.article_id)
            continue

        summary = _strip_preamble(raw)
        if len(summary) < _MIN_SUMMARY_CHARS:
            stats.skipped += 1
            logger.warning(
                "Discarding too-short summary (%d chars) for article %s",
                len(summary),
                article.article_id,
            )
            continue

        article.llm_summary = summary
        # Writing the row fires ``updated_at``'s onupdate=_utcnow, which lands a
        # few microseconds AFTER this timestamp — enough for the staleness check
        # (llm_summary_at < updated_at) to flag the article again on the very
        # next run. Assigning updated_at explicitly suppresses the onupdate hook,
        # so both columns share one instant and the run converges.
        stamp = datetime.now(timezone.utc)
        article.llm_summary_at = stamp
        article.updated_at = stamp
        # Commit per article: an interrupted run keeps everything it finished.
        await db.commit()
        stats.enriched += 1

    if stats.enriched:
        # Hydration serves ArticleMeta.summary from best_summary, so the cached
        # entries are now stale.
        invalidate_metadata_cache()

    logger.info(
        "Summary enrichment complete — enriched=%d skipped=%d failed=%d",
        stats.enriched,
        stats.skipped,
        stats.failed,
    )
    return stats.as_dict()
