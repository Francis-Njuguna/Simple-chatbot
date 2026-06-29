"""Unit tests for Pydantic schemas."""

from backend.app.models.schemas import ChatRequest, FeedbackRequest


def test_chat_request_valid() -> None:
    req = ChatRequest(message="How do I login to LMS?")
    assert req.message == "How do I login to LMS?"
    assert req.session_id is None


def test_feedback_request_rating_bounds() -> None:
    req = FeedbackRequest(message_id="abc-123", rating=5)
    assert req.rating == 5
