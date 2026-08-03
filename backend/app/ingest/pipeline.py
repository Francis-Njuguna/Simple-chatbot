"""Main ingestion orchestrator.

Storage split
-------------
* **PostgreSQL** — source of truth for all metadata (title, url, category,
  summary, captions, file paths, chunk counts).
* **ChromaDB** — one collection of vectors carrying only the keys needed to
  retrieve: ``source_type``, ``article_id``/``image_id``, ``chunk_index`` and
  ``category`` (kept for server-side filtering). Display fields are deliberately
  NOT written here; see ``_chroma_text_metadata`` / ``_chroma_image_metadata``.

Article summaries are extractive and computed inline (no LLM call), so ingestion
stays fast and works without a model endpoint. The optional abstractive summary
is a separate background pass — see ``services/enrichment_service.py``.
"""

import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import get_settings
from backend.app.database.chroma import (
    clear_collections,
    count_by_source_type,
    delete_stale_article_chunks,
    upsert_image_embeddings,
    upsert_text_chunks,
)
from backend.app.database.models import DocumentMetadata, ImageMetadata
from backend.app.ingest.chunker import TextChunker
from backend.app.ingest.crawler import KnowledgeBaseCrawler
from backend.app.ingest.image_processor import ImageProcessor
from backend.app.ingest.summarizer import summarize_extractive
from backend.app.rag.embeddings import EmbeddingService
from backend.app.rag.lexical import invalidate_lexical_index, invalidate_lexicon
from backend.app.services.metadata_service import invalidate_metadata_cache
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Metadata keys Chroma is allowed to carry for a text chunk. Everything else
# lives in PostgreSQL and is hydrated after retrieval.
#   article_id  → join key back to Postgres
#   chunk_index → citation ordering
#   category    → server-side `where` filtering + MMR scoping
_CHROMA_TEXT_KEYS = ("article_id", "chunk_index", "category")


class IngestionPipeline:
    """Orchestrates crawling, chunking, embedding, and storage."""

    def __init__(
        self,
        db: AsyncSession,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.db = db
        self.settings = get_settings()
        self.crawler = KnowledgeBaseCrawler()
        self.chunker = TextChunker()
        self.image_processor = ImageProcessor()
        self.embedding_service = embedding_service or EmbeddingService()

    async def run(self, force: bool = False, include_images: bool = True) -> dict[str, Any]:
        # Crawl BEFORE clearing anything. The old order cleared the collection
        # first, so a crawl that then failed — network, TLS, a site change —
        # left the KB permanently empty with no way back short of a working
        # crawl. Destroy the live index only once replacement content is in hand.
        articles = await self.crawler.crawl_all()

        if force:
            if not articles:
                # Refusing here is the whole point: an empty crawl must not be
                # allowed to wipe a working knowledge base.
                logger.error(
                    "Crawl returned 0 articles — REFUSING to clear the existing "
                    "collection. The knowledge base is unchanged."
                )
                return {
                    "status": "error",
                    "articles_processed": 0,
                    "chunks_created": 0,
                    "images_processed": 0,
                    "message": (
                        "Crawl returned no articles; existing knowledge base left "
                        "intact. Check crawler connectivity/TLS before retrying."
                    ),
                }

            # A *partial* crawl is the subtler danger. crawl_all() swallows
            # per-article errors and returns whatever succeeded, so one TLS or
            # network blip can return 3 of 20 articles — enough to pass the
            # zero-check above, and clearing on that silently discards the other
            # 17. Compare against what Postgres already knows and refuse a
            # suspicious collapse.
            known = (
                await self.db.execute(select(func.count()).select_from(DocumentMetadata))
            ).scalar_one()
            if known and len(articles) < known * self.settings.reingest_min_coverage:
                logger.error(
                    "Crawl returned only %d article(s) but Postgres knows %d "
                    "(< %.0f%% coverage) — REFUSING to clear. Set "
                    "REINGEST_MIN_COVERAGE=0 to override once the drop is "
                    "confirmed to be a genuine upstream deletion.",
                    len(articles),
                    known,
                    self.settings.reingest_min_coverage * 100,
                )
                return {
                    "status": "error",
                    "articles_processed": len(articles),
                    "chunks_created": 0,
                    "images_processed": 0,
                    "message": (
                        f"Crawl returned {len(articles)} of {known} known articles; "
                        f"existing knowledge base left intact. Investigate the "
                        f"crawler before re-running, or set REINGEST_MIN_COVERAGE=0 "
                        f"if the articles were genuinely removed upstream."
                    ),
                }

            clear_collections()
            logger.info(
                "ChromaDB collections cleared for full re-ingest (%d articles crawled)",
                len(articles),
            )

        raw_dir = Path(self.settings.raw_data_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        all_chunks: list[dict[str, Any]] = []
        all_images: list[dict[str, Any]] = []

        for article in articles:
            raw_path = raw_dir / f"article_{article.article_id}.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "article_id": article.article_id,
                        "title": article.title,
                        "category": article.category,
                        "url": article.url,
                        "text": article.text,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            chunks = self.chunker.chunk_article(
                article_id=article.article_id,
                title=article.title,
                category=article.category,
                url=article.url,
                text=article.text,
            )
            all_chunks.extend(chunks)

            # Extractive, no LLM call — ingestion must not depend on a model
            # endpoint being reachable.
            summary = summarize_extractive(
                article.text,
                max_sentences=self.settings.summary_max_sentences,
                max_chars=self.settings.summary_max_chars,
                title=article.title,
            )

            await self._upsert_document_metadata(article, len(chunks), summary)

            if include_images:
                for img in article.images:
                    downloaded = await self.image_processor.download_image(
                        url=img["url"],
                        article_id=article.article_id,
                        alt_text=img.get("alt_text", ""),
                        category=article.category,
                        caption=img.get("caption", ""),
                        article_title=article.title,
                        # Surrounding step text — one semantic embedding per
                        # image is built from caption + alt text + this.
                        context=img.get("context", ""),
                    )
                    if downloaded:
                        all_images.append(downloaded)

        local_images = self.image_processor.scan_local_images()
        existing_ids = {img["image_id"] for img in all_images}
        for img in local_images:
            if img["image_id"] not in existing_ids:
                all_images.append(img)

        chunks_created = await self._store_chunks(all_chunks)

        # Incremental runs only: a force run already cleared the collection, so
        # there is nothing stale left to find and the extra scans are pure cost.
        stale_removed = 0
        if not force:
            stale_removed = self._prune_stale_chunks(all_chunks)

        images_processed = await self._store_images(all_images)

        await self.db.commit()

        self._invalidate_derived_caches()

        vector_counts = count_by_source_type()
        logger.info(
            "Ingestion complete — vectors in Chroma: text=%s image=%s",
            vector_counts.get("text"),
            vector_counts.get("image"),
        )

        return {
            "status": "success",
            "articles_processed": len(articles),
            "chunks_created": chunks_created,
            "images_processed": images_processed,
            "stale_chunks_removed": stale_removed,
            "vectors_text": vector_counts.get("text"),
            "vectors_image": vector_counts.get("image"),
            "message": (
                f"Ingested {len(articles)} articles with {chunks_created} chunks "
                f"and {images_processed} images."
                + (f" Pruned {stale_removed} stale chunk(s)." if stale_removed else "")
            ),
        }

    @staticmethod
    def _invalidate_derived_caches() -> None:
        """Drop every cache derived from the corpus we just rewrote.

        Four independent caches go stale the moment ingestion commits, and each
        one fails in a different, quiet way if it is left behind:

        * **metadata cache** — read-through title/url/caption lookups, would
          keep serving the pre-crawl titles (e.g. the old 'Article Details').
        * **BM25 index** — built from a full scan of the Chroma text corpus. A
          stale index scores against deleted chunk ids, so lexical hits point at
          rows that no longer exist and silently drop out of the fused ranking.
        * **lexicon** — the vocabulary that backs fuzzy query correction. New
          articles introduce new terms; without a rebuild they can never be
          matched as corrections.
        * **retrieval cache** — TTL'd query→results map. Would keep serving
          pre-ingest answers for up to ``retrieval_cache_ttl`` seconds.

        Best-effort by design: ingestion has already committed by this point, so
        a cache-flush failure must be logged, never raised. The worst case is a
        stale read that expires on its own.
        """
        invalidate_metadata_cache()

        try:
            invalidate_lexical_index()
            invalidate_lexicon()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to invalidate lexical index/lexicon: %s", exc)

        try:
            # Imported lazily: retriever pulls in embeddings/reranker at module
            # scope, and ingestion should not depend on that import graph.
            from backend.app.rag.retriever import get_retriever

            get_retriever().clear_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to clear the retrieval cache: %s", exc)

        logger.info("Derived caches invalidated (metadata, BM25, lexicon, retrieval).")

    @staticmethod
    def _prune_stale_chunks(chunks: list[dict[str, Any]]) -> int:
        """Remove positional chunk ids this run did not rewrite, per article.

        Only articles present in ``chunks`` are touched. An article the crawl
        missed this time is left completely alone — a transient fetch failure
        must not silently delete good content.
        """
        by_article: dict[str, list[str]] = {}
        for chunk in chunks:
            article_id = chunk["metadata"].get("article_id")
            if article_id:
                by_article.setdefault(str(article_id), []).append(chunk["chunk_id"])

        total = 0
        for article_id, keep_ids in by_article.items():
            try:
                total += delete_stale_article_chunks(article_id, keep_ids)
            except Exception as exc:  # noqa: BLE001
                # A failed prune leaves stale chunks, which is bad but not as bad
                # as aborting an ingest that has already written good content.
                logger.warning(
                    "Stale-chunk prune failed for article %s: %s", article_id, exc
                )
        if total:
            logger.info("Pruned %d stale chunk(s) across %d article(s)", total, len(by_article))
        return total

    async def _upsert_document_metadata(
        self, article: Any, chunk_count: int, summary: str
    ) -> None:
        result = await self.db.execute(
            select(DocumentMetadata).where(DocumentMetadata.article_id == article.article_id)
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.title = article.title
            existing.category = article.category
            existing.url = article.url
            existing.chunk_count = chunk_count
            existing.raw_content = article.text[:50000]
            # Only overwrite with a non-empty summary: a crawl that returned
            # thin text should not wipe a good summary from a previous run.
            if summary:
                existing.summary = summary
        else:
            self.db.add(
                DocumentMetadata(
                    article_id=article.article_id,
                    title=article.title,
                    category=article.category,
                    url=article.url,
                    chunk_count=chunk_count,
                    raw_content=article.text[:50000],
                    summary=summary or None,
                )
            )

    @staticmethod
    def _chroma_text_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Project chunk metadata down to the keys Chroma needs.

        Title and url are dropped here — PostgreSQL owns them, and the
        retriever hydrates them after filtering/MMR.
        """
        projected = {
            key: metadata[key] for key in _CHROMA_TEXT_KEYS if metadata.get(key) is not None
        }
        # Chroma rejects None; normalise the filter key to a real string.
        projected["category"] = projected.get("category") or "General"
        return projected

    async def _store_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0

        batch_size = 32
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embedding_service.embed_texts(texts)
            ids = [c["chunk_id"] for c in batch]
            metadatas = [self._chroma_text_metadata(c["metadata"]) for c in batch]
            upsert_text_chunks(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            total += len(batch)
        logger.info("Stored %d text chunks in ChromaDB", total)
        return total

    @staticmethod
    def _dedupe_images_by_id(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep one entry per unique image_id.

        image_id is derived from the image URL, so a shared asset (e.g. the
        amiuhelp.png logo that appears on nearly every article) yields the same
        id from many articles. Without deduping, the batched lists handed to
        ChromaDB's upsert() contain duplicate ids within a single call, which
        raises chromadb.errors.DuplicateIDError. Dedupe once, up front, so both
        the Chroma upsert and the per-image Postgres writes see unique ids.
        """
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for img in images:
            image_id = img["image_id"]
            if image_id in seen:
                continue
            seen.add(image_id)
            unique.append(img)

        dropped = len(images) - len(unique)
        if dropped:
            logger.info(
                "Deduplicated images: %d duplicate id(s) collapsed (%d unique of %d collected)",
                dropped,
                len(unique),
                len(images),
            )
        return unique

    async def _store_images(self, images: list[dict[str, Any]]) -> int:
        images = self._dedupe_images_by_id(images)
        if not images:
            return 0

        batch_size = 32
        total = 0
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            texts = [img["embed_text"] for img in batch]
            embeddings = self.embedding_service.embed_texts(texts)
            ids = [img["image_id"] for img in batch]
            # Minimal metadata: the join key, plus category for `where` filtering.
            # Caption/filename/paths live in PostgreSQL and are hydrated after
            # retrieval. The caption still shapes the *vector* via embed_text.
            metadatas = [
                {
                    "article_id": img.get("article_id") or "",
                    "category": img.get("category") or "General",
                }
                for img in batch
            ]
            upsert_image_embeddings(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            for img in batch:
                result = await self.db.execute(
                    select(ImageMetadata).where(ImageMetadata.image_id == img["image_id"])
                )
                existing = result.scalar_one_or_none()
                if existing:
                    existing.filename = img["filename"]
                    existing.filepath = img["filepath"]
                    existing.static_path = img.get("static_path")
                    existing.caption = img.get("caption")
                    existing.alt_text = img.get("alt_text")
                    existing.article_id = img.get("article_id")
                    existing.category = img.get("category")
                    existing.keywords = img.get("keywords")
                    existing.source_url = img.get("source_url")
                else:
                    self.db.add(
                        ImageMetadata(
                            image_id=img["image_id"],
                            filename=img["filename"],
                            filepath=img["filepath"],
                            static_path=img.get("static_path"),
                            caption=img.get("caption"),
                            alt_text=img.get("alt_text"),
                            article_id=img.get("article_id"),
                            category=img.get("category"),
                            keywords=img.get("keywords"),
                            source_url=img.get("source_url"),
                        )
                    )
            total += len(batch)

        logger.info("Stored %d image embeddings in ChromaDB", total)
        return total
