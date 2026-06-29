"""RAG orchestration service."""

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.models import AnalyticsLog, ChatMessage, Session
from backend.app.models.schemas import (
    ChatResponse,
    ImageResult,
    SourceCitation,
)
from backend.app.rag.llm import LLMService
from backend.app.rag.retriever import HybridRetriever
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_TITLE_LEN = 60


class RAGService:
    """End-to-end RAG pipeline: retrieve, generate, persist."""

    def __init__(
        self,
        db: AsyncSession,
        retriever: HybridRetriever | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.db = db
        self.retriever = retriever or HybridRetriever()
        self.llm_service = llm_service or LLMService()

    async def _get_or_create_session(
        self, session_id: Optional[str], first_message: str
    ) -> Session:
        if session_id:
            result = await self.db.execute(
                select(Session).where(Session.id == uuid.UUID(session_id))
            )
            session = result.scalar_one_or_none()
            if session:
                return session

        # Derive a readable title from the first message
        title = first_message.strip()
        if len(title) > _MAX_TITLE_LEN:
            title = title[:_MAX_TITLE_LEN].rsplit(" ", 1)[0] + "…"

        session = Session(title=title)
        self.db.add(session)
        await self.db.flush()
        return session

    async def _get_history_text(self, session_id: uuid.UUID, limit: int = 6) -> str:
        result = await self.db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        messages = list(reversed(result.scalars().all()))
        if not messages:
            return "No prior conversation."
        lines = [f"{m.role}: {m.content[:500]}" for m in messages]
        return "\n".join(lines)

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> ChatResponse:
        session = await self._get_or_create_session(session_id, message)
        history = await self._get_history_text(session.id)

        user_msg = ChatMessage(session_id=session.id, role="user", content=message)
        self.db.add(user_msg)
        await self.db.flush()

        chunks = await self.retriever.retrieve_text(message, category=category)
        images = await self.retriever.retrieve_images(message, category=category)
        context = self.retriever.format_context(chunks)
        confidence = self.retriever.compute_confidence(chunks)

        answer = await self.llm_service.generate_answer(
            question=message,
            context=context,
            history=history,
        )

        sources = [
            SourceCitation(
                article_id=c.article_id,
                title=c.title,
                url=c.url,
                category=c.category,
                chunk_index=c.chunk_index,
                score=c.score,
            )
            for c in chunks
        ]

        image_results = [
            ImageResult(
                image_id=img.image_id,
                filename=img.filename,
                filepath=img.static_path or img.filepath,
                caption=img.caption,
                alt_text=img.alt_text,
                article_id=img.article_id,
                category=img.category,
                score=img.score,
            )
            for img in images
        ]

        metadata = {
            "sources": [s.model_dump() for s in sources],
            "images": [i.model_dump() for i in image_results],
            "confidence": confidence,
        }

        assistant_msg = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer,
            metadata_=metadata,
        )
        self.db.add(assistant_msg)

        self.db.add(
            AnalyticsLog(
                event_type="chat_query",
                session_id=session.id,
                payload={"message": message[:200], "confidence": confidence},
            )
        )
        await self.db.flush()

        return ChatResponse(
            answer=answer,
            images=image_results,
            sources=sources,
            confidence=confidence,
            session_id=str(session.id),
            message_id=str(assistant_msg.id),
        )
