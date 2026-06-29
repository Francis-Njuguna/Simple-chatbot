"""Main ingestion orchestrator."""

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import get_settings
from backend.app.database.chroma import clear_collections, upsert_image_embeddings, upsert_text_chunks
from backend.app.database.models import DocumentMetadata, ImageMetadata
from backend.app.ingest.chunker import TextChunker
from backend.app.ingest.crawler import KnowledgeBaseCrawler
from backend.app.ingest.image_processor import ImageProcessor
from backend.app.rag.embeddings import EmbeddingService
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


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
        if force:
            clear_collections()
            logger.info("ChromaDB collections cleared for full re-ingest")

        articles = await self.crawler.crawl_all()
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

            await self._upsert_document_metadata(article, len(chunks))

            if include_images:
                for img in article.images:
                    downloaded = await self.image_processor.download_image(
                        url=img["url"],
                        article_id=article.article_id,
                        alt_text=img.get("alt_text", ""),
                        category=article.category,
                    )
                    if downloaded:
                        all_images.append(downloaded)

        local_images = self.image_processor.scan_local_images()
        existing_ids = {img["image_id"] for img in all_images}
        for img in local_images:
            if img["image_id"] not in existing_ids:
                all_images.append(img)

        chunks_created = await self._store_chunks(all_chunks)
        images_processed = await self._store_images(all_images)

        await self.db.commit()

        return {
            "status": "success",
            "articles_processed": len(articles),
            "chunks_created": chunks_created,
            "images_processed": images_processed,
            "message": f"Ingested {len(articles)} articles with {chunks_created} chunks and {images_processed} images.",
        }

    async def _upsert_document_metadata(self, article: Any, chunk_count: int) -> None:
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
        else:
            self.db.add(
                DocumentMetadata(
                    article_id=article.article_id,
                    title=article.title,
                    category=article.category,
                    url=article.url,
                    chunk_count=chunk_count,
                    raw_content=article.text[:50000],
                )
            )

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
            metadatas = [c["metadata"] for c in batch]
            upsert_text_chunks(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            total += len(batch)
        logger.info("Stored %d text chunks in ChromaDB", total)
        return total

    async def _store_images(self, images: list[dict[str, Any]]) -> int:
        if not images:
            return 0

        batch_size = 32
        total = 0
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            texts = [img["embed_text"] for img in batch]
            embeddings = self.embedding_service.embed_texts(texts)
            ids = [img["image_id"] for img in batch]
            metadatas = [
                {
                    "filename": img["filename"],
                    "filepath": img["filepath"],
                    "static_path": img["static_path"],
                    "caption": img.get("caption", ""),
                    "alt_text": img.get("alt_text", ""),
                    "article_id": img.get("article_id") or "",
                    "category": img.get("category") or "",
                    "keywords": img.get("keywords", ""),
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
