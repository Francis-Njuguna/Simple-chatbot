"""Hybrid retrieval with MMR and reranking."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from backend.app.config import get_settings
from backend.app.database.chroma import query_image_collection, query_text_collection
from backend.app.prompts.templates import CONTEXT_CHUNK_TEMPLATE
from backend.app.rag.embeddings import EmbeddingService
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
        self.embedding_service = embedding_service or EmbeddingService()

    def _distance_to_score(self, distance: float) -> float:
        return max(0.0, min(1.0, 1.0 - distance))

    def _mmr_select(
        self,
        query_embedding: list[float],
        candidates: list[RetrievedChunk],
        candidate_embeddings: list[list[float]],
        k: int,
        lambda_param: float,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if len(candidates) <= k:
            return candidates

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_indices) < k and remaining:
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                relevance = self.embedding_service.similarity(
                    query_embedding, candidate_embeddings[idx]
                )
                if not selected_indices:
                    mmr_score = relevance
                else:
                    max_sim = max(
                        self.embedding_service.similarity(
                            candidate_embeddings[idx], candidate_embeddings[s]
                        )
                        for s in selected_indices
                    )
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def _rerank(
        self,
        query: str,
        query_emb: list[float],
        chunks: list[RetrievedChunk],
        chunk_embs: list[list[float]],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """Re-score chunks against the query embedding and return top_n."""
        if not chunks:
            return []

        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk, emb in zip(chunks, chunk_embs, strict=True):
            score = self.embedding_service.similarity(query_emb, emb)
            scored.append((score, RetrievedChunk(**{**chunk.__dict__, "score": score})))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_n]]

    async def retrieve_text(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k_retrieval
        query_embedding = self.embedding_service.embed_query(query)

        where_filter: dict[str, Any] | None = None
        if category:
            where_filter = {"category": category}

        fetch_k = top_k * 3
        results = query_text_collection(
            query_embedding=query_embedding,
            n_results=fetch_k,
            where=where_filter,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not ids:
            return []

        # --- batch embed all candidate texts in one call (was: one call per chunk) ---
        texts = [documents[i] or "" for i in range(len(ids))]
        candidate_embeddings: list[list[float]] = self.embedding_service.embed_texts(texts)

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

        mmr_chunks = self._mmr_select(
            query_embedding=query_embedding,
            candidates=candidates,
            candidate_embeddings=candidate_embeddings,
            k=fetch_k,
            lambda_param=self.settings.mmr_diversity,
        )

        # Reuse the embeddings we already have for the MMR-selected chunks
        mmr_indices = [candidates.index(c) for c in mmr_chunks]
        mmr_embeddings = [candidate_embeddings[i] for i in mmr_indices]

        reranked = self._rerank(
            query=query,
            query_emb=query_embedding,
            chunks=mmr_chunks,
            chunk_embs=mmr_embeddings,
            top_n=self.settings.rerank_top_n,
        )

        logger.info("Retrieved %d text chunks for query", len(reranked))
        return reranked[:top_k]

    async def retrieve_images(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievedImage]:
        top_k = top_k or self.settings.top_k_images
        query_embedding = self.embedding_service.embed_query(query)

        where_filter: dict[str, Any] | None = None
        if category:
            where_filter = {"category": category}

        results = query_image_collection(
            query_embedding=query_embedding,
            n_results=top_k * 2,
            where=where_filter,
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
