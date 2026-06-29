"""Feedback service."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.models import AnalyticsLog, Feedback
from backend.app.models.schemas import FeedbackRequest, FeedbackResponse
from backend.app.utils.exceptions import NotFoundError
from sqlalchemy import select

from backend.app.database.models import ChatMessage


class FeedbackService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def submit_feedback(
        self,
        request: FeedbackRequest,
        user_id: uuid.UUID | None = None,
    ) -> FeedbackResponse:
        result = await self.db.execute(
            select(ChatMessage).where(ChatMessage.id == uuid.UUID(request.message_id))
        )
        message = result.scalar_one_or_none()
        if not message:
            raise NotFoundError("Message not found")

        feedback = Feedback(
            message_id=message.id,
            user_id=user_id,
            rating=request.rating,
            comment=request.comment,
        )
        self.db.add(feedback)
        self.db.add(
            AnalyticsLog(
                event_type="feedback",
                user_id=user_id,
                session_id=message.session_id,
                payload={"rating": request.rating},
            )
        )
        await self.db.flush()

        return FeedbackResponse(
            id=str(feedback.id),
            message_id=str(feedback.message_id),
            rating=feedback.rating,
            comment=feedback.comment,
            created_at=feedback.created_at,
        )
