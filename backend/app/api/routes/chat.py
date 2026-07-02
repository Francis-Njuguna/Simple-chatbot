"""Chat API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

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
