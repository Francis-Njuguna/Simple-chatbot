"""Hybrid retrieval with MMR and reranking.

Metadata split
--------------
Retrieval reads only what Chroma carries (ids, ``category``, ``chunk_index``)
and returns chunks/images whose display fields are blank. ``hydrate_chunks`` /
``hydrate_images`` then fill title, url, summary, caption and paths from
**PostgreSQL**, which is the source of truth. Filtering, MMR and reranking all
happen before that, inside Chroma / NumPy, so hydration touches only the handful
of records that survived.

For a vector whose Postgres row is missing (partial re-ingest, or a store
written before the metadata split), hydration falls back to whatever the Chroma
record still carries — retrieval degrades to the old behaviour rather than
returning blank citations.

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
* Hydration is a single ``IN (...)`` query per kind, behind a TTL cache.
* ``get_retriever`` returns a process-wide singleton so the retriever (and its
  embedding backend) is built once, not per request.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import anyio
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import get_settings
from backend.app.database.chroma import query_image_collection, query_text_collection
from backend.app.prompts.templates import (
    CONTEXT_CHUNK_TEMPLATE,
    EMPTY_CONTEXT_NOTE,
    IMAGE_CHUNK_TEMPLATE,
    IMAGE_CONTEXT_NOTE,
    NO_IMAGES_NOTE,
)
from backend.app.rag.embeddings import EmbeddingService, get_embedding_service
from backend.app.services import metadata_service
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    article_id: str
    category: Optional[str]
    chunk_index: int
    score: float
    # Hydrated from PostgreSQL after retrieval (see ``hydrate_chunks``).
    title: str = ""
    url: str = ""
    summary: Optional[str] = None
    # Raw Chroma metadata, kept only as a fallback when the Postgres row is
    # missing. Not used once hydration succeeds.
    _raw_meta: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class RetrievedImage:
    image_id: str
    article_id: Optional[str]
    category: Optional[str]
    score: float
    # Hydrated from PostgreSQL after retrieval (see ``hydrate_images``).
    filename: str = ""
    filepath: str = ""
    static_path: str = ""
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    _raw_meta: dict[str, Any] = field(default_factory=dict, repr=False)


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
                    category=meta.get("category"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=self._distance_to_score(distances[i]),
                    _raw_meta=meta,
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
            # (and higher LLM latency) on repeated context.
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
                    article_id=meta.get("article_id") or None,
                    category=meta.get("category") or None,
                    score=score,
                    _raw_meta=meta,
                )
            )

        images.sort(key=lambda x: x.score, reverse=True)
        return images[:top_k]

    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        db: Optional[AsyncSession] = None,
    ) -> tuple[list[RetrievedChunk], list[RetrievedImage]]:
        """Embed the query ONCE and run text + image retrieval concurrently.

        When ``db`` is supplied the survivors are hydrated from PostgreSQL
        before returning; without it the caller gets id-only records and can
        hydrate later (or not at all, e.g. in tests).
        """
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

        if db is not None:
            # Both hydrations are independent single-table reads, but they share
            # one AsyncSession — SQLAlchemy sessions are not concurrency-safe, so
            # these must run sequentially, not in a task group.
            await self.hydrate_chunks(db, chunks)
            await self.hydrate_images(db, images)

        return chunks, images

    # ------------------------------------------------------------------
    # PostgreSQL hydration — display metadata, source of truth
    # ------------------------------------------------------------------

    async def hydrate_chunks(self, db: AsyncSession, chunks: list[RetrievedChunk]) -> None:
        """Fill title/url/category/summary on ``chunks`` from PostgreSQL."""
        if not chunks:
            return

        articles = await metadata_service.get_articles(db, (c.article_id for c in chunks))

        for chunk in chunks:
            meta = articles.get(chunk.article_id)
            if meta is not None:
                chunk.title = meta.title
                chunk.url = meta.url
                chunk.summary = meta.summary
                # Chroma's category drives filtering; Postgres owns the value.
                chunk.category = meta.category or chunk.category
            else:
                # No Postgres row — fall back to whatever the vector carries so
                # a pre-migration store still yields usable citations.
                raw = chunk._raw_meta
                chunk.title = raw.get("title", "") or chunk.title
                chunk.url = raw.get("url", "") or chunk.url

    async def hydrate_images(self, db: AsyncSession, images: list[RetrievedImage]) -> None:
        """Fill filename/paths/caption on ``images`` from PostgreSQL."""
        if not images:
            return

        records = await metadata_service.get_images(db, (i.image_id for i in images))

        for image in images:
            meta = records.get(image.image_id)
            if meta is not None:
                image.filename = meta.filename
                image.filepath = meta.filepath
                image.static_path = meta.static_path or f"/static/images/{meta.filename}"
                image.caption = meta.caption
                image.alt_text = meta.alt_text
                image.article_id = meta.article_id or image.article_id
                image.category = meta.category or image.category
            else:
                raw = image._raw_meta
                image.filename = raw.get("filename", "") or image.filename
                image.filepath = raw.get("filepath", "") or image.filepath
                image.static_path = raw.get("static_path", "") or image.static_path
                image.caption = raw.get("caption") or image.caption
                image.alt_text = raw.get("alt_text") or image.alt_text

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return EMPTY_CONTEXT_NOTE

        # One article contributes several chunks; its summary is emitted once,
        # as framing above the excerpts, rather than repeated per chunk.
        summaries_emitted: set[str] = set()
        blocks: list[str] = []
        for chunk in chunks:
            summary = ""
            if chunk.summary and chunk.article_id not in summaries_emitted:
                summaries_emitted.add(chunk.article_id)
                summary = f"Article overview: {chunk.summary}\n"
            blocks.append(
                CONTEXT_CHUNK_TEMPLATE.format(
                    title=chunk.title or "Untitled article",
                    category=chunk.category or "General",
                    url=chunk.url,
                    summary=summary,
                    text=chunk.text,
                )
            )
        return "\n".join(blocks)

    def format_images(self, images: list[RetrievedImage]) -> str:
        """Describe the images the client will render, for the LLM prompt.

        The model never sees pixels — only captions — so it can point at a
        screenshot ("see the image below") without inventing one.
        """
        if not images:
            return NO_IMAGES_NOTE

        lines: list[str] = []
        for img in images:
            caption = (img.caption or img.alt_text or "").strip()
            # Ingest falls back to "Image from article N" when a page ships no
            # caption or alt text; that carries no meaning for the model, so
            # skip it rather than feed it a filler description to quote.
            if not caption or caption.lower().startswith("image from article"):
                caption = f"Screenshot from {img.category or 'the knowledge base'}"
            source = f" (from article {img.article_id})" if img.article_id else ""
            lines.append(IMAGE_CHUNK_TEMPLATE.format(caption=caption, source=source))

        return "\n".join([IMAGE_CONTEXT_NOTE, "", *lines])

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
