"""RAG orchestration service.

Connection lifecycle
--------------------
This service does NOT hold a database session. It opens short ``db_scope``
blocks around database work and closes them before anything slow runs.

    acquire → session + history + user message → RELEASE
              embedding, Chroma, BM25, rerank        (no connection held)
    acquire → hydrate titles/urls/captions   → RELEASE
              context build, NVIDIA NIM generation   (no connection held)
    acquire → persist answer + analytics     → RELEASE

The previous design took a request-scoped ``AsyncSession`` from FastAPI's
dependency injection. Because a ``yield`` dependency is only unwound *after* the
response is sent, one pooled connection stayed checked out for the entire
request — including the multi-second LLM call. That capped the server at
pool_size + max_overflow concurrent requests (15) no matter how idle the CPU
was, and request 16 blocked for ``pool_timeout`` and then failed. Connection
hold time is now the DB work itself (single-digit ms) rather than the request.

Performance notes
-----------------
* The query embedding is computed **once** and reused for text + image search
  which run **concurrently**.
* Every stage is timed via :class:`StageTimer` and a full breakdown is logged
  for each request so bottlenecks are visible in production logs.
* ``chat`` remains a single blocking answer; ``chat_stream`` streams the
  LLM's tokens as they arrive for a far lower time-to-first-token.
"""

import uuid
from typing import AsyncIterator, Optional

import anyio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.models import AnalyticsLog, ChatMessage, Session
from backend.app.database.session import db_scope
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
        retriever: HybridRetriever | None = None,
        llm_service: LLMService | None = None,
        session_scope=db_scope,
    ) -> None:
        # Reuse the process-wide singletons by default (built once at startup).
        self.retriever = retriever or get_retriever()
        self.llm_service = llm_service or get_llm_service()
        # Injectable so a test can supply a scope bound to its own transaction.
        self._scope = session_scope

    async def _get_or_create_session(
        self, db: AsyncSession, session_id: Optional[str], first_message: str
    ) -> uuid.UUID:
        """Return the session id, creating the row if needed.

        Returns a plain UUID rather than the ORM instance: the caller uses it
        after this session has closed, and a detached instance is a lazy-load
        waiting to happen.
        """
        if session_id:
            result = await db.execute(
                select(Session.id).where(Session.id == uuid.UUID(session_id))
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                return existing

        # Derive a readable title from the first message
        title = first_message.strip()
        if len(title) > _MAX_TITLE_LEN:
            title = title[:_MAX_TITLE_LEN].rsplit(" ", 1)[0] + "…"

        session = Session(title=title)
        db.add(session)
        await db.flush()
        return session.id

    async def _get_history_text(
        self, db: AsyncSession, session_id: uuid.UUID, limit: int = 6
    ) -> str:
        result = await db.execute(
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
    ) -> tuple[uuid.UUID, str, list[RetrievedChunk], list[RetrievedImage], str, str, float]:
        """Everything up to the LLM: session, history, retrieval.

        ``_get_or_create_session`` and ``_get_history_text`` are the ONLY DB
        calls before the NIM call. They happen inside one short ``db_scope``
        block, which commits and releases the connection before embedding /
        retrieval start — see the module docstring for the lifecycle.
        """
        async with timer.astage("session_history"):
            async with self._scope("chat_prepare") as db:
                session_id = await self._get_or_create_session(db, session_id, message)
                history = await self._get_history_text(db, session_id)
                db.add(ChatMessage(session_id=session_id, role="user", content=message))
                await db.flush()

        # Embed the query a single time, then fan out text + image retrieval.
        async with timer.astage("embedding"):
            query_embedding = await self.retriever.embed_query(message)

        async with timer.astage("retrieval"):
            # No session is held during search. Chroma, BM25 and the
            # cross-encoder never touch PostgreSQL, so no connection is
            # occupied for the ~2s this takes.
            chunks, images, processed = await self.retriever.retrieve(
                message,
                category=category,
                query_embedding=query_embedding,
            )

        # Hydration is the one part of retrieval that reads PostgreSQL, so it
        # gets its own short scope rather than keeping one open across search.
        async with timer.astage("hydration"):
            async with self._scope("chat_hydrate") as db:
                chunks, images = await self.retriever.hydrate_results(db, chunks, images)

        with timer.stage("context_build"):
            context = self.retriever.format_context(chunks)
            image_context = self.retriever.format_images(images)
            # `processed` comes straight out of retrieval so the threshold can
            # adapt to whether preprocessing understood the query (entity/intent
            # detected, typo corrected, synonym expanded).
            confidence = self.retriever.compute_confidence(chunks, processed)

        return session_id, history, chunks, images, context, image_context, confidence

    @staticmethod
    def _build_sources(chunks: list[RetrievedChunk]) -> list[SourceCitation]:
        """One citation per *article*, not per chunk.

        Retrieval works in chunks, and several chunks of the same article
        routinely survive reranking — so mapping chunks straight to citations
        rendered the same title and URL four or five times in the widget's
        "Sources & References" list. The user is being pointed at documents, so
        the article is the right unit.

        Chunks arrive in relevance order; ``dict`` preserves insertion order, so
        the best-scoring chunk of each article is the one kept and the overall
        ordering is unchanged.
        """
        best: dict[str, SourceCitation] = {}
        for c in chunks:
            if c.article_id in best:
                continue
            best[c.article_id] = SourceCitation(
                article_id=c.article_id,
                title=c.title,
                url=c.url,
                category=c.category,
                chunk_index=c.chunk_index,
                score=c.score,
            )
        return list(best.values())

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
            session_id_uuid,
            history,
            chunks,
            images,
            context,
            image_context,
            confidence,
        ) = await self._prepare(message, session_id, category, timer)

        # NO connection is held here. This is the whole point of the refactor:
        # the LLM call is the longest stage of the request by an order of
        # magnitude, and it must not occupy a pooled connection.
        async with timer.astage("llm"):
            answer = await self.llm_service.generate_answer(
                question=message,
                context=context,
                history=history,
                images=image_context,
            )

        sources = self._build_sources(chunks)
        image_results = self._build_images(images)
        metadata = {
            "sources": [s.model_dump() for s in sources],
            "images": [i.model_dump() for i in image_results],
            "confidence": confidence,
        }

        async with timer.astage("persist"):
            async with self._scope("chat_persist") as db:
                assistant_msg = ChatMessage(
                    session_id=session_id_uuid,
                    role="assistant",
                    content=answer,
                    metadata_=metadata,
                )
                db.add(assistant_msg)
                db.add(
                    AnalyticsLog(
                        event_type="chat_query",
                        session_id=session_id_uuid,
                        payload={"message": message[:200], "confidence": confidence},
                    )
                )
                await db.flush()
                # Read the id before the session closes: expire_on_commit is
                # False, but the attribute must still be loaded while attached.
                message_id = str(assistant_msg.id)

        response = ChatResponse(
            answer=answer,
            images=image_results,
            sources=sources,
            confidence=confidence,
            session_id=str(session_id_uuid),
            message_id=message_id,
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
            session_id_uuid,
            history,
            chunks,
            images,
            context,
            image_context,
            confidence,
        ) = await self._prepare(message, session_id, category, timer)

        sources = self._build_sources(chunks)
        image_results = self._build_images(images)

        # Emit metadata first so the client can render sources/images while the
        # answer streams in.
        yield {
            "type": "meta",
            "session_id": str(session_id_uuid),
            "sources": [s.model_dump() for s in sources],
            "images": [i.model_dump() for i in image_results],
            "confidence": confidence,
        }

        # No connection is held across the token stream. Streaming is the worst
        # case for the old design — the connection stayed checked out not just
        # for generation but until the client finished consuming the response.
        parts: list[str] = []
        started = anyio.current_time()
        first_token_ms: Optional[float] = None
        async for token in self.llm_service.stream_answer(
            question=message, context=context, history=history, images=image_context
        ):
            if first_token_ms is None:
                first_token_ms = (anyio.current_time() - started) * 1000.0
                timer.mark("llm_first_token", first_token_ms)
            parts.append(token)
            yield {"type": "token", "text": token}

        answer = "".join(parts)

        async with timer.astage("persist"):
            metadata = {
                "sources": [s.model_dump() for s in sources],
                "images": [i.model_dump() for i in image_results],
                "confidence": confidence,
            }
            async with self._scope("chat_stream_persist") as db:
                assistant_msg = ChatMessage(
                    session_id=session_id_uuid,
                    role="assistant",
                    content=answer,
                    metadata_=metadata,
                )
                db.add(assistant_msg)
                db.add(
                    AnalyticsLog(
                        event_type="chat_query",
                        session_id=session_id_uuid,
                        payload={"message": message[:200], "confidence": confidence},
                    )
                )
                await db.flush()
                message_id = str(assistant_msg.id)

        yield {"type": "done", "message_id": message_id}
        timer.log(logger)
