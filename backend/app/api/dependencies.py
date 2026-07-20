"""FastAPI dependency injection.

Performance note
----------------
``RAGService`` is intentionally lightweight — it only binds a request-scoped DB
session.  Its heavy collaborators (the embedding model / HTTP pools and the LLM
chat client) are *process-wide singletons* obtained via ``get_embedding_service``
and ``get_llm_service``.  Previously ``RAGService(db)`` implicitly constructed a
fresh ``HybridRetriever()`` and ``LLMService()`` on every request, which rebuilt
the Anthropic HTTP connection pool per call.  We now inject the shared singletons
so nothing expensive is reconstructed on the hot path.
"""

import uuid
from collections.abc import AsyncGenerator
from typing import Annotated, Optional

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.session import get_db_session
from backend.app.rag.llm import get_llm_service
from backend.app.rag.retriever import get_retriever
from backend.app.services.auth_service import AuthService
from backend.app.services.feedback_service import FeedbackService
from backend.app.services.history_service import HistoryService
from backend.app.services.rag_service import RAGService
from backend.app.utils.exceptions import AuthenticationError
from backend.app.utils.security import decode_access_token


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db_session():
        yield session


DbSession = Annotated[AsyncSession, Depends(get_session)]


def get_rag_service(db: DbSession) -> RAGService:
    # Reuse the process-wide retriever + LLM singletons (built once at startup).
    return RAGService(db, retriever=get_retriever(), llm_service=get_llm_service())


def get_history_service(db: DbSession) -> HistoryService:
    return HistoryService(db)


def get_feedback_service(db: DbSession) -> FeedbackService:
    return FeedbackService(db)


def get_auth_service(db: DbSession) -> AuthService:
    return AuthService(db)


async def get_optional_user_id(
    authorization: Annotated[Optional[str], Header()] = None,
    db: DbSession = None,
) -> Optional[uuid.UUID]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_access_token(token)
        user_id = uuid.UUID(payload["sub"])
        auth_service = AuthService(db)
        user = await auth_service.get_user(user_id)
        if user and user.is_active:
            return user_id
    except (AuthenticationError, ValueError, KeyError):
        return None
    return None


async def require_user_id(
    authorization: Annotated[Optional[str], Header()] = None,
    db: DbSession = None,
) -> uuid.UUID:
    user_id = await get_optional_user_id(authorization=authorization, db=db)
    if user_id is None:
        raise AuthenticationError("Authentication required")
    return user_id
