"""Pydantic request/response schemas."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None
    category: Optional[str] = None


class ImageResult(BaseModel):
    image_id: str
    filename: str
    filepath: str
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    article_id: Optional[str] = None
    category: Optional[str] = None
    score: float = 0.0


class SourceCitation(BaseModel):
    article_id: str
    title: str
    url: str
    category: Optional[str] = None
    chunk_index: Optional[int] = None
    score: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    images: list[ImageResult] = []
    sources: list[SourceCitation] = []
    confidence: float = 0.0
    session_id: str
    message_id: str


class FeedbackRequest(BaseModel):
    message_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    message_id: str
    rating: int
    comment: Optional[str] = None
    created_at: datetime


class IngestRequest(BaseModel):
    force: bool = False
    include_images: bool = True


class IngestResponse(BaseModel):
    status: str
    articles_processed: int
    chunks_created: int
    images_processed: int
    message: str


class HistoryMessage(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime
    metadata: Optional[dict[str, Any]] = None


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class ArticleSummary(BaseModel):
    article_id: str
    title: str
    category: Optional[str] = None
    url: str
    chunk_count: int = 0


class ArticlesResponse(BaseModel):
    articles: list[ArticleSummary]
    total: int


class ImageMetadataResponse(BaseModel):
    image_id: str
    filename: str
    filepath: str
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    article_id: Optional[str] = None
    category: Optional[str] = None
    keywords: Optional[str] = None


class ImagesListResponse(BaseModel):
    images: list[ImageMetadataResponse]
    total: int
