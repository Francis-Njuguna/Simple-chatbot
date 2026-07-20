"""Embedding service — Ollama (default: nomic-embed-text) | sentence-transformers fallback.

Performance notes
-----------------
* A single shared ``httpx.AsyncClient`` (with keep-alive connection pooling) is
  reused across the whole process instead of opening a fresh TCP/TLS connection
  on every embed call.
* ``embed_query_async`` / ``embed_texts_async`` are true coroutines so the
  FastAPI event loop is never blocked on network I/O.
* The synchronous CPU-bound sentence-transformers path is off-loaded to a
  thread via ``anyio.to_thread`` when awaited, so it also never blocks the loop.
* The ``EmbeddingService`` object itself is cheap and cached at module level
  (see ``get_embedding_service``) so we never rebuild the backend per request.
"""

from functools import lru_cache
from typing import Sequence

import anyio
import httpx
import numpy as np

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared async HTTP client (keep-alive connection pooling for Ollama)
# ---------------------------------------------------------------------------

_async_http_client: httpx.AsyncClient | None = None


def _get_async_http_client() -> httpx.AsyncClient:
    global _async_http_client
    if _async_http_client is None:
        _async_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _async_http_client


async def close_async_http_client() -> None:
    """Close the shared client on application shutdown."""
    global _async_http_client
    if _async_http_client is not None:
        await _async_http_client.aclose()
        _async_http_client = None


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
        # Reusable sync client for ingestion / non-async call sites.
        self._sync_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        logger.info(
            "Ollama embedding backend ready — model: %s  url: %s",
            self._model,
            self._base_url,
        )

    @staticmethod
    def _normalise(embedding: list[float]) -> list[float]:
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    # ---- sync (ingestion) -------------------------------------------------
    def _embed_one(self, text: str) -> list[float]:
        url = f"{self._base_url}/api/embeddings"
        try:
            response = self._sync_client.post(
                url, json={"model": self._model, "prompt": text}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            if not embedding:
                raise ValueError(
                    f"Ollama returned empty embedding for text: {text[:80]!r}"
                )
            return self._normalise(embedding)
        except httpx.HTTPError as exc:
            logger.error("Ollama embedding request failed: %s", exc)
            raise

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._embed_one(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)

    # ---- async (request path) --------------------------------------------
    async def _embed_one_async(self, text: str) -> list[float]:
        url = f"{self._base_url}/api/embeddings"
        client = _get_async_http_client()
        try:
            response = await client.post(
                url, json={"model": self._model, "prompt": text}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            if not embedding:
                raise ValueError(
                    f"Ollama returned empty embedding for text: {text[:80]!r}"
                )
            return self._normalise(embedding)
        except httpx.HTTPError as exc:
            logger.error("Ollama async embedding request failed: %s", exc)
            raise

    async def embed_query_async(self, query: str) -> list[float]:
        return await self._embed_one_async(query)

    async def embed_texts_async(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [await self._embed_one_async(t) for t in texts]


# ---------------------------------------------------------------------------
# sentence-transformers backend  (FALLBACK — local HuggingFace model)
# ---------------------------------------------------------------------------

@lru_cache
def _get_sentence_transformer():
    """Load and cache the local SentenceTransformer model (loaded once)."""
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

    async def embed_query_async(self, query: str) -> list[float]:
        # CPU-bound → run in a worker thread to avoid blocking the event loop.
        return await anyio.to_thread.run_sync(self.embed_query, query)

    async def embed_texts_async(self, texts: Sequence[str]) -> list[list[float]]:
        return await anyio.to_thread.run_sync(self.embed_texts, list(texts))


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

    # ---- sync API (ingestion / scripts) ----------------------------------
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self._backend.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._backend.embed_query(query)

    # ---- async API (request path) ----------------------------------------
    async def embed_query_async(self, query: str) -> list[float]:
        return await self._backend.embed_query_async(query)

    async def embed_texts_async(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._backend.embed_texts_async(texts)

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity (vectors are L2-normalised, so this equals dot product)."""
        return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Process-wide singleton — built ONCE, reused across every request.
# ---------------------------------------------------------------------------

@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
