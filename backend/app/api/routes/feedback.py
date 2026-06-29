"""Feedback API routes."""

import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_feedback_service, get_optional_user_id
from backend.app.models.schemas import FeedbackRequest, FeedbackResponse
from backend.app.services.feedback_service import FeedbackService

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    feedback_service: Annotated[FeedbackService, Depends(get_feedback_service)],
    user_id: Annotated[Optional[uuid.UUID], Depends(get_optional_user_id)] = None,
) -> FeedbackResponse:
    return await feedback_service.submit_feedback(request, user_id=user_id)
