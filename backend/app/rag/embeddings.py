"""Embedding service — Ollama (default: nomic-embed-text) | sentence-transformers fallback."""

from functools import lru_cache
from typing import Sequence

import httpx
import numpy as np

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Ollama embeddings backend  (PRIMARY — nomic-embed-text)
# ---------------------------------------------------------------------------

class _OllamaEmbeddingBackend:
    """Calls Ollama's /api/embeddings endpoint.

    Default model : nomic-embed-text  (768-dim, fast, free, locally hosted)
    Install       : ollama pull nomic-embed-text
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_embedding_model
        logger.info(
            "Ollama embedding backend ready — model: %s  url: %s",
            self._model,
            self._base_url,
        )

    def _embed_one(self, text: str) -> list[float]:
        """POST to Ollama /api/embeddings and return an L2-normalised vector."""
        url = f"{self._base_url}/api/embeddings"
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json={"model": self._model, "prompt": text})
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding", [])
                if not embedding:
                    raise ValueError(
                        f"Ollama returned empty embedding for text: {text[:80]!r}"
                    )
                # L2-normalise so cosine similarity == dot product
                arr = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                return arr.tolist()
        except httpx.HTTPError as exc:
            logger.error("Ollama embedding request failed: %s", exc)
            raise

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._embed_one(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)


# ---------------------------------------------------------------------------
# sentence-transformers backend  (FALLBACK — local HuggingFace model)
# ---------------------------------------------------------------------------

@lru_cache
def _get_sentence_transformer():
    """Load and cache the local SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer  # lazy import

    settings = get_settings()
    logger.info("Loading sentence-transformer model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model, device=settings.embedding_device)


class _SentenceTransformerBackend:
    def __init__(self) -> None:
        self._model = _get_sentence_transformer()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


# ---------------------------------------------------------------------------
# Public EmbeddingService — provider chosen at construction time
# ---------------------------------------------------------------------------

class EmbeddingService:
    """Generates text embeddings using the configured provider.

    EMBEDDING_PROVIDER=ollama              → nomic-embed-text via Ollama (default)
    EMBEDDING_PROVIDER=sentence-transformers → local HuggingFace model (fallback)
    """

    def __init__(self) -> None:
        settings = get_settings()
        provider = settings.embedding_provider
        if provider == "sentence-transformers":
            self._backend: _OllamaEmbeddingBackend | _SentenceTransformerBackend = (
                _SentenceTransformerBackend()
            )
        else:
            # "ollama" is the default
            self._backend = _OllamaEmbeddingBackend()
        logger.info("EmbeddingService initialised — provider: %s", provider)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self._backend.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._backend.embed_query(query)

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity (vectors are L2-normalised, so this equals dot product)."""
        return float(np.dot(a, b))
