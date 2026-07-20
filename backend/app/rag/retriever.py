"""Hybrid retrieval with MMR and reranking.

Performance notes
-----------------
* The query is embedded **once per request** and the resulting vector is shared
  between text retrieval, image retrieval, MMR and reranking.
* Candidate chunk embeddings are read straight from ChromaDB
  (``include_embeddings=True``) — we never re-embed the retrieved chunks over
  the network (previously ~15 sequential embed calls per query).
* MMR is fully vectorised with NumPy (matrix ops) instead of an O(n²) Python
  loop calling ``similarity`` repeatedly.
* Synchronous ChromaDB calls are off-loaded to a worker thread so they don't
  block the FastAPI event loop.
* ``get_retriever`` returns a process-wide singleton so the retriever (and its
  embedding backend) is built once, not per request.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import anyio
import numpy as np

from backend.app.config import get_settings
from backend.app.database.chroma import query_image_collection, query_text_collection
from backend.app.prompts.templates import CONTEXT_CHUNK_TEMPLATE
from backend.app.rag.embeddings import EmbeddingService, get_embedding_service
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    article_id: str
    title: str
    url: str
    category: Optional[str]
    chunk_index: int
    score: float


@dataclass
class RetrievedImage:
    image_id: str
    filename: str
    filepath: str
    static_path: str
    caption: Optional[str]
    alt_text: Optional[str]
    article_id: Optional[str]
    category: Optional[str]
    score: float


class HybridRetriever:
    """Vector search with metadata filtering, MMR, and reranking."""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self.settings = get_settings()
        self.embedding_service = embedding_service or get_embedding_service()

    def _distance_to_score(self, distance: float) -> float:
        return max(0.0, min(1.0, 1.0 - distance))

    def _mmr_select_vectorised(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        k: int,
        lambda_param: float,
    ) -> list[int]:
        """Vectorised MMR — returns selected candidate indices in order.

        Vectors are assumed L2-normalised, so dot product == cosine similarity.
        """
        n = candidate_embeddings.shape[0]
        if n == 0:
            return []
        if n <= k:
            return list(range(n))

        # Precompute relevance (query vs each candidate) once.
        relevance = candidate_embeddings @ query_embedding  # shape (n,)
        # Pairwise candidate similarity matrix (n x n) — computed once.
        pairwise = candidate_embeddings @ candidate_embeddings.T

        selected: list[int] = []
        remaining = list(range(n))

        # First pick = most relevant.
        first = int(np.argmax(relevance))
        selected.append(first)
        remaining.remove(first)

        while len(selected) < k and remaining:
            rem = np.array(remaining)
            # Max similarity of each remaining candidate to any selected one.
            max_sim = pairwise[np.ix_(rem, selected)].max(axis=1)
            mmr = lambda_param * relevance[rem] - (1.0 - lambda_param) * max_sim
            best = int(rem[int(np.argmax(mmr))])
            selected.append(best)
            remaining.remove(best)

        return selected

    async def embed_query(self, query: str) -> list[float]:
        """Embed the query a single time (shared across text + image search)."""
        return await self.embedding_service.embed_query_async(query)

    async def retrieve_text(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k_retrieval
        if query_embedding is None:
            query_embedding = await self.embedding_service.embed_query_async(query)

        where_filter: dict[str, Any] | None = None
        if category:
            where_filter = {"category": category}

        fetch_k = top_k * 3
        results = await anyio.to_thread.run_sync(
            lambda: query_text_collection(
                query_embedding=query_embedding,
                n_results=fetch_k,
                where=where_filter,
                include_embeddings=True,
            )
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        embeddings = results.get("embeddings", [[]])
        embeddings = embeddings[0] if embeddings else []

        if not ids:
            return []

        texts = [documents[i] or "" for i in range(len(ids))]

        candidates: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] or {}
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=texts[i],
                    article_id=meta.get("article_id", ""),
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                    category=meta.get("category"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=self._distance_to_score(distances[i]),
                )
            )

        # Reuse the stored embeddings from Chroma — no re-embedding.
        q_vec = np.asarray(query_embedding, dtype=np.float32)
        cand_matrix = np.asarray(embeddings, dtype=np.float32)

        # MMR select on the candidate pool.
        mmr_indices = self._mmr_select_vectorised(
            query_embedding=q_vec,
            candidate_embeddings=cand_matrix,
            k=fetch_k,
            lambda_param=self.settings.mmr_diversity,
        )

        # Rerank the MMR-selected chunks by relevance to the query, keep top_n.
        rel_scores = cand_matrix[mmr_indices] @ q_vec
        order = np.argsort(rel_scores)[::-1]
        rerank_top_n = self.settings.rerank_top_n

        reranked: list[RetrievedChunk] = []
        seen_texts: set[str] = set()
        for rank_pos in order[:rerank_top_n]:
            cand_idx = mmr_indices[int(rank_pos)]
            chunk = candidates[cand_idx]
            # Drop exact-duplicate chunk bodies so we never spend prompt tokens
            # (and Claude latency) on repeated context.
            dedup_key = chunk.text.strip()
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)
            chunk.score = float(rel_scores[int(rank_pos)])
            reranked.append(chunk)

        logger.info("Retrieved %d text chunks for query", len(reranked[:top_k]))
        return reranked[:top_k]

    async def retrieve_images(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> list[RetrievedImage]:
        top_k = top_k or self.settings.top_k_images
        if query_embedding is None:
            query_embedding = await self.embedding_service.embed_query_async(query)

        where_filter: dict[str, Any] | None = None
        if category:
            where_filter = {"category": category}

        results = await anyio.to_thread.run_sync(
            lambda: query_image_collection(
                query_embedding=query_embedding,
                n_results=top_k * 2,
                where=where_filter,
            )
        )

        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        images: list[RetrievedImage] = []
        for i, image_id in enumerate(ids):
            meta = metadatas[i] or {}
            score = self._distance_to_score(distances[i])
            if score < 0.3:
                continue
            images.append(
                RetrievedImage(
                    image_id=image_id,
                    filename=meta.get("filename", ""),
                    filepath=meta.get("filepath", ""),
                    static_path=meta.get("static_path", ""),
                    caption=meta.get("caption"),
                    alt_text=meta.get("alt_text"),
                    article_id=meta.get("article_id") or None,
                    category=meta.get("category") or None,
                    score=score,
                )
            )

        images.sort(key=lambda x: x.score, reverse=True)
        return images[:top_k]

    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> tuple[list[RetrievedChunk], list[RetrievedImage]]:
        """Embed the query ONCE and run text + image retrieval concurrently."""
        if query_embedding is None:
            query_embedding = await self.embedding_service.embed_query_async(query)

        chunks: list[RetrievedChunk] = []
        images: list[RetrievedImage] = []

        async with anyio.create_task_group() as tg:
            async def _text() -> None:
                nonlocal chunks
                chunks = await self.retrieve_text(
                    query, category=category, query_embedding=query_embedding
                )

            async def _images() -> None:
                nonlocal images
                images = await self.retrieve_images(
                    query, category=category, query_embedding=query_embedding
                )

            tg.start_soon(_text)
            tg.start_soon(_images)

        return chunks, images

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context found."
        return "\n".join(
            CONTEXT_CHUNK_TEMPLATE.format(
                title=c.title,
                category=c.category or "General",
                url=c.url,
                text=c.text,
            )
            for c in chunks
        )

    def compute_confidence(self, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        scores = [c.score for c in chunks]
        return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Process-wide singleton — built ONCE, reused across every request.
# ---------------------------------------------------------------------------

@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
