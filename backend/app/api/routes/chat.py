"""Chat API routes."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from backend.app.api.dependencies import get_rag_service
from backend.app.core.limiter import limiter  # ← was: from backend.app.main import limiter
from backend.app.models.schemas import ChatRequest, ChatResponse
from backend.app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatResponse:
    """Submit a chat message and receive a RAG-powered answer.

    The ``request`` parameter is required by ``slowapi`` so it can read the
    client IP address for rate-limiting; FastAPI injects it automatically.
    """
    return await rag_service.chat(
        message=body.message,
        session_id=body.session_id,
        category=body.category,
    )


@router.post("/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> StreamingResponse:
    """Stream a RAG answer as newline-delimited JSON (NDJSON) events.

    Each line is a JSON object with a ``type`` field:
      * ``meta``  — session id, sources, images, confidence (sent first).
      * ``token`` — an incremental piece of the answer text.
      * ``done``  — final marker with the persisted ``message_id``.

    Streaming dramatically reduces perceived latency: the client renders the
    first tokens in ~1–2s instead of waiting for Claude to finish the full
    answer.
    """

    async def event_generator():
        async for event in rag_service.chat_stream(
            message=body.message,
            session_id=body.session_id,
            category=body.category,
        ):
            yield json.dumps(event) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
