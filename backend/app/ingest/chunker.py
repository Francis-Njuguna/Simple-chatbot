"""Text chunking for ingestion."""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.config import get_settings


class TextChunker:
    """Splits article text into retrieval-sized chunks."""

    def __init__(self) -> None:
        settings = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_article(
        self,
        article_id: str,
        title: str,
        category: str | None,
        url: str,
        text: str,
    ) -> list[dict[str, Any]]:
        if not text.strip():
            return []

        chunks = self._splitter.split_text(text)
        result: list[dict[str, Any]] = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{article_id}_chunk_{idx}"
            result.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "article_id": article_id,
                        "title": title,
                        "category": category or "General",
                        "url": url,
                        "chunk_index": idx,
                    },
                }
            )
        return result
