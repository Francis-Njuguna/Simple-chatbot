"""Chat history service."""

import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.models import ChatMessage, DocumentMetadata, ImageMetadata, Session
from backend.app.models.schemas import (
    ArticleSummary,
    ArticlesResponse,
    HistoryMessage,
    HistoryResponse,
    ImageMetadataResponse,
    ImagesListResponse,
)


class HistoryService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_history(self, session_id: str) -> HistoryResponse:
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == uuid.UUID(session_id))
            .order_by(ChatMessage.created_at)
        )
        messages = result.scalars().all()
        return HistoryResponse(
            session_id=session_id,
            messages=[
                HistoryMessage(
                    id=str(m.id),
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at,
                    metadata=m.metadata_,
                )
                for m in messages
            ],
        )

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        result = await self.db.execute(
            select(Session).order_by(Session.updated_at.desc()).limit(limit)
        )
        sessions = result.scalars().all()
        return [
            {
                "session_id": str(s.id),
                "title": s.title or "Chat session",
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in sessions
        ]

    async def list_articles(
        self,
        category: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> ArticlesResponse:
        query = select(DocumentMetadata)
        if category:
            query = query.where(DocumentMetadata.category == category)
        count_result = await self.db.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar() or 0

        result = await self.db.execute(
            query.order_by(DocumentMetadata.title).offset(skip).limit(limit)
        )
        docs = result.scalars().all()
        return ArticlesResponse(
            articles=[
                ArticleSummary(
                    article_id=d.article_id,
                    title=d.title,
                    category=d.category,
                    url=d.url,
                    chunk_count=d.chunk_count,
                )
                for d in docs
            ],
            total=total,
        )

    async def list_images(
        self,
        article_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> ImagesListResponse:
        query = select(ImageMetadata)
        if article_id:
            query = query.where(ImageMetadata.article_id == article_id)
        count_result = await self.db.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar() or 0

        result = await self.db.execute(query.offset(skip).limit(limit))
        images = result.scalars().all()
        return ImagesListResponse(
            images=[
                ImageMetadataResponse(
                    image_id=img.image_id,
                    filename=img.filename,
                    filepath=img.filepath,
                    caption=img.caption,
                    alt_text=img.alt_text,
                    article_id=img.article_id,
                    category=img.category,
                    keywords=img.keywords,
                )
                for img in images
            ],
            total=total,
        )

    async def list_categories(self) -> list[str]:
        result = await self.db.execute(
            select(DocumentMetadata.category)
            .where(DocumentMetadata.category.isnot(None))
            .distinct()
            .order_by(DocumentMetadata.category)
        )
        return [row[0] for row in result.all() if row[0]]
