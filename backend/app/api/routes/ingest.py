"""Ingestion API routes."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import DbSession, require_user_id
from backend.app.ingest.pipeline import IngestionPipeline
from backend.app.models.schemas import IngestRequest, IngestResponse
from backend.app.rag.embeddings import EmbeddingService

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def trigger_ingest(
    request: IngestRequest,
    db: DbSession,
    _: Annotated[uuid.UUID, Depends(require_user_id)],
) -> IngestResponse:
    pipeline = IngestionPipeline(db, EmbeddingService())
    result = await pipeline.run(force=request.force, include_images=request.include_images)
    return IngestResponse(**result)
