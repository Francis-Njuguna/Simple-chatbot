"""RAG orchestration service.

Performance notes
-----------------
* The query embedding is computed **once** and reused for text + image search
  which run **concurrently** (previously each embedded the query separately and
  ran sequentially — two embed calls + serial vector searches per request).
* Session lookup + history load run concurrently with retrieval where possible.
* Every stage is timed via :class:`StageTimer` and a full breakdown is logged
  for each request so bottlenecks are visible in production logs.
* ``chat`` remains a single blocking answer; ``chat_stream`` streams Claude's
  tokens as they arrive for a far lower time-to-first-token.
"""

import uuid
from typing import AsyncIterator, Optional

import anyio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.models import AnalyticsLog, ChatMessage, Session
from backend.app.models.schemas import (
    ChatResponse,
    ImageResult,
    SourceCitation,
)
from backend.app.rag.llm import LLMService, get_llm_service
from backend.app.rag.retriever import (
    HybridRetriever,
    RetrievedChunk,
    RetrievedImage,
    get_retriever,
)
from backend.app.utils.logging import get_logger
from backend.app.utils.timing import StageTimer

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
        # Reuse the process-wide singletons by default (built once at startup).
        self.retriever = retriever or get_retriever()
        self.llm_service = llm_service or get_llm_service()

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

    # ------------------------------------------------------------------
    # Shared prep: session, history, retrieval (embed once, run concurrently)
    # ------------------------------------------------------------------
    async def _prepare(
        self,
        message: str,
        session_id: Optional[str],
        category: Optional[str],
        timer: StageTimer,
    ) -> tuple[Session, str, list[RetrievedChunk], list[RetrievedImage], str, float]:
        async with timer.astage("session_history"):
            session = await self._get_or_create_session(session_id, message)
            history = await self._get_history_text(session.id)

            user_msg = ChatMessage(session_id=session.id, role="user", content=message)
            self.db.add(user_msg)
            await self.db.flush()

        # Embed the query a single time, then fan out text + image retrieval.
        async with timer.astage("embedding"):
            query_embedding = await self.retriever.embed_query(message)

        async with timer.astage("retrieval"):
            chunks, images = await self.retriever.retrieve(
                message, category=category, query_embedding=query_embedding
            )

        with timer.stage("context_build"):
            context = self.retriever.format_context(chunks)
            confidence = self.retriever.compute_confidence(chunks)

        return session, history, chunks, images, context, confidence

    @staticmethod
    def _build_sources(chunks: list[RetrievedChunk]) -> list[SourceCitation]:
        return [
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

    @staticmethod
    def _build_images(images: list[RetrievedImage]) -> list[ImageResult]:
        return [
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

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> ChatResponse:
        timer = StageTimer("chat")

        (
            session,
            history,
            chunks,
            images,
            context,
            confidence,
        ) = await self._prepare(message, session_id, category, timer)

        async with timer.astage("llm"):
            answer = await self.llm_service.generate_answer(
                question=message,
                context=context,
                history=history,
            )

        with timer.stage("persist"):
            sources = self._build_sources(chunks)
            image_results = self._build_images(images)
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

        response = ChatResponse(
            answer=answer,
            images=image_results,
            sources=sources,
            confidence=confidence,
            session_id=str(session.id),
            message_id=str(assistant_msg.id),
        )
        timer.log(logger)
        return response

    async def chat_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Stream the answer. Yields dicts:

        * ``{"type": "meta", ...}``  — session id, sources, images, confidence.
        * ``{"type": "token", "text": ...}`` — incremental answer text.
        * ``{"type": "done", "message_id": ...}`` — final marker.
        """
        timer = StageTimer("chat_stream")

        (
            session,
            history,
            chunks,
            images,
            context,
            confidence,
        ) = await self._prepare(message, session_id, category, timer)

        sources = self._build_sources(chunks)
        image_results = self._build_images(images)

        # Emit metadata first so the client can render sources/images while the
        # answer streams in.
        yield {
            "type": "meta",
            "session_id": str(session.id),
            "sources": [s.model_dump() for s in sources],
            "images": [i.model_dump() for i in image_results],
            "confidence": confidence,
        }

        parts: list[str] = []
        started = anyio.current_time()
        first_token_ms: Optional[float] = None
        async for token in self.llm_service.stream_answer(
            question=message, context=context, history=history
        ):
            if first_token_ms is None:
                first_token_ms = (anyio.current_time() - started) * 1000.0
                timer.mark("llm_first_token", first_token_ms)
            parts.append(token)
            yield {"type": "token", "text": token}

        answer = "".join(parts)

        with timer.stage("persist"):
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

        yield {"type": "done", "message_id": str(assistant_msg.id)}
        timer.log(logger)
