"""Unit tests for text chunking."""

from backend.app.ingest.chunker import TextChunker


def test_chunk_article_splits_text() -> None:
    chunker = TextChunker()
    long_text = "This is a test sentence. " * 200
    chunks = chunker.chunk_article(
        article_id="1",
        title="Test Article",
        category="LMS",
        url="https://example.com/article=1",
        text=long_text,
    )
    assert len(chunks) > 1
    assert chunks[0]["metadata"]["article_id"] == "1"
    assert chunks[0]["metadata"]["title"] == "Test Article"
    assert "chunk_0" in chunks[0]["chunk_id"]


def test_chunk_article_empty_text() -> None:
    chunker = TextChunker()
    chunks = chunker.chunk_article("1", "Title", None, "http://x", "")
    assert chunks == []
