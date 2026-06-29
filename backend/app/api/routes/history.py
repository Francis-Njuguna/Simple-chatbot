"""History and catalog API routes."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query

from backend.app.api.dependencies import get_history_service
from backend.app.models.schemas import (
    ArticlesResponse,
    HistoryResponse,
    ImagesListResponse,
)
from backend.app.services.history_service import HistoryService

router = APIRouter(tags=["catalog"])


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(
    session_id: str,
    history_service: Annotated[HistoryService, Depends(get_history_service)],
) -> HistoryResponse:
    return await history_service.get_history(session_id)


@router.get("/history", response_model=list)
async def list_sessions(
    history_service: Annotated[HistoryService, Depends(get_history_service)],
    limit: int = Query(default=20, le=100),
) -> list:
    return await history_service.list_sessions(limit=limit)


@router.get("/articles", response_model=ArticlesResponse)
async def list_articles(
    history_service: Annotated[HistoryService, Depends(get_history_service)],
    category: Optional[str] = None,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, le=200),
) -> ArticlesResponse:
    return await history_service.list_articles(category=category, skip=skip, limit=limit)


@router.get("/categories", response_model=list[str])
async def list_categories(
    history_service: Annotated[HistoryService, Depends(get_history_service)],
) -> list[str]:
    return await history_service.list_categories()


@router.get("/images", response_model=ImagesListResponse)
async def list_images(
    history_service: Annotated[HistoryService, Depends(get_history_service)],
    article_id: Optional[str] = None,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, le=200),
) -> ImagesListResponse:
    return await history_service.list_images(article_id=article_id, skip=skip, limit=limit)
