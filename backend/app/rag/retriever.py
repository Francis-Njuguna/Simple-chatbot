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

import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Optional

import anyio
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import get_settings
from backend.app.database.chroma import (
    fetch_text_chunks_by_id,
    query_image_collection,
    query_text_collection,
)
from backend.app.rag.lexical import get_lexical_index, rewrite_query
from backend.app.rag.reranker import get_reranker
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
    # Cosine similarity to the query, in [0, 1]. This stays the *semantic*
    # score even when a cross-encoder reorders the results, because confidence
    # and the relevance floor are both calibrated against it.
    score: float
    # Hydrated from PostgreSQL after retrieval (see ``hydrate_chunks``).
    title: str = ""
    url: str = ""
    summary: Optional[str] = None
    # Fusion / rerank diagnostics. ``rerank_score`` is a cross-encoder logit
    # (unbounded, often negative) and is used for ORDERING only — never as a
    # confidence value or compared against a similarity threshold.
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    retrieval_source: str = "vector"
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
        # (query, category) → (stored_at, chunks, images). OrderedDict so the
        # oldest entry can be evicted once the cache is full.
        self._cache: "OrderedDict[tuple[str, str], tuple[float, list[RetrievedChunk], list[RetrievedImage]]]" = (
            OrderedDict()
        )

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

    def _rrf_fuse(
        self,
        vector_ranking: list[str],
        lexical_ranking: list[str],
    ) -> dict[str, float]:
        """Reciprocal Rank Fusion of two rankings → {chunk_id: fused_score}.

        RRF combines *positions*, not scores: a chunk's contribution from each
        engine is ``weight / (k + rank)``. That sidesteps the fact that cosine
        similarity (0-1) and BM25 (unbounded, corpus-dependent) are on wildly
        different scales and cannot be averaged directly. ``k`` damps the head
        of each list so one engine's top hit cannot dominate on its own.
        """
        k = self.settings.rrf_k

        # With only one ranking present, its weight must be 1.0 rather than the
        # hybrid split. Otherwise every vector-only score is uniformly scaled by
        # hybrid_vector_weight (0.6) — harmless for ordering, but it makes the
        # rrf_score in debug output 40% low and not comparable against a hybrid
        # query's scores, which is exactly when you are reading those numbers.
        if lexical_ranking:
            v_weight = self.settings.hybrid_vector_weight
            l_weight = self.settings.hybrid_bm25_weight
        else:
            v_weight, l_weight = 1.0, 0.0

        fused: dict[str, float] = {}
        for rank, chunk_id in enumerate(vector_ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + (v_weight / (k + rank + 1))
        for rank, chunk_id in enumerate(lexical_ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + (l_weight / (k + rank + 1))
        return fused

    async def retrieve_text(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval: vector + BM25 → RRF → MMR → cross-encoder.

        Each stage narrows a wider pool than the last stage needs, which is the
        point: MMR can only add diversity if it has spare candidates to choose
        between, and the cross-encoder can only fix ordering if the right chunk
        is somewhere in its shortlist.
        """
        settings = self.settings
        top_k = top_k or settings.top_k_retrieval
        debug = settings.retrieval_debug_active
        timings: dict[str, float] = {}

        if query_embedding is None:
            query_embedding = await self.embedding_service.embed_query_async(query)

        where_filter: dict[str, Any] | None = {"category": category} if category else None
        pool = max(settings.retrieval_candidate_pool, top_k * 3)

        # --- Stage 1: vector candidates ---------------------------------
        start = time.perf_counter()
        results = await anyio.to_thread.run_sync(
            lambda: query_text_collection(
                query_embedding=query_embedding,
                n_results=pool,
                where=where_filter,
                include_embeddings=True,
            )
        )
        timings["vector_ms"] = (time.perf_counter() - start) * 1000

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        embeddings_raw = results.get("embeddings", [])
        embeddings = embeddings_raw[0] if len(embeddings_raw) else []

        # id → (text, meta, embedding, cosine score)
        pool_by_id: dict[str, dict[str, Any]] = {}
        for i, chunk_id in enumerate(ids):
            pool_by_id[chunk_id] = {
                "text": documents[i] or "",
                "meta": metadatas[i] or {},
                "embedding": embeddings[i] if i < len(embeddings) else None,
                "score": self._distance_to_score(distances[i]),
                "source": "vector",
            }
        vector_ranking = list(ids)

        # --- Stage 2: lexical (BM25) candidates -------------------------
        lexical_ranking: list[str] = []
        if settings.hybrid_search_enabled:
            start = time.perf_counter()
            lexical_hits = await anyio.to_thread.run_sync(
                lambda: get_lexical_index().search(query, k=pool)
            )
            timings["bm25_ms"] = (time.perf_counter() - start) * 1000
            lexical_ranking = [chunk_id for chunk_id, _ in lexical_hits]

            # BM25 can surface chunks the vector query never returned. Pull
            # their text/embedding so they can compete on equal footing.
            missing = [cid for cid in lexical_ranking if cid not in pool_by_id]
            if missing:
                extra = await anyio.to_thread.run_sync(
                    lambda: fetch_text_chunks_by_id(missing, include_embeddings=True)
                )
                extra_ids = extra["ids"]
                extra_docs = extra["documents"]
                extra_meta = extra["metadatas"]
                extra_emb = extra["embeddings"]
                q_arr = np.asarray(query_embedding, dtype=np.float32)
                for i, chunk_id in enumerate(extra_ids):
                    vec = extra_emb[i] if i < len(extra_emb) else None
                    # Vectors are L2-normalised, so dot == cosine. Compute the
                    # score directly rather than leaving these at 0.
                    cosine = (
                        float(np.asarray(vec, dtype=np.float32) @ q_arr)
                        if vec is not None
                        else 0.0
                    )
                    pool_by_id[chunk_id] = {
                        "text": extra_docs[i] or "",
                        "meta": extra_meta[i] or {},
                        "embedding": vec,
                        "score": max(0.0, min(1.0, cosine)),
                        "source": "bm25",
                    }

        if not pool_by_id:
            if debug:
                logger.info("[retrieval] query=%r → no candidates at all", query)
            return []

        # --- Stage 3: fuse ----------------------------------------------
        if lexical_ranking:
            fused = self._rrf_fuse(vector_ranking, lexical_ranking)
        else:
            # Vector-only: rank position IS the order, so fuse against itself.
            fused = self._rrf_fuse(vector_ranking, [])

        fused_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        fused_ids = [cid for cid in fused_ids if cid in pool_by_id]

        # --- Stage 4: MMR on the fused pool -----------------------------
        # Only candidates with a usable embedding can take part.
        mmr_ids = [cid for cid in fused_ids if pool_by_id[cid]["embedding"] is not None]
        shortlist_n = min(settings.mmr_shortlist, len(mmr_ids))
        selected_ids: list[str]
        if mmr_ids and shortlist_n > 0:
            q_vec = np.asarray(query_embedding, dtype=np.float32)
            cand_matrix = np.asarray(
                [pool_by_id[cid]["embedding"] for cid in mmr_ids], dtype=np.float32
            )
            start = time.perf_counter()
            mmr_idx = self._mmr_select_vectorised(
                query_embedding=q_vec,
                candidate_embeddings=cand_matrix,
                # Was fetch_k (== pool size), which hit the n <= k early return
                # and made MMR a pass-through. A shortlist strictly smaller than
                # the pool is what makes the selection meaningful.
                k=shortlist_n,
                lambda_param=settings.mmr_diversity,
            )
            timings["mmr_ms"] = (time.perf_counter() - start) * 1000
            selected_ids = [mmr_ids[i] for i in mmr_idx]
        else:
            selected_ids = fused_ids[: settings.mmr_shortlist]

        # --- Stage 5: cross-encoder rerank ------------------------------
        reranker = get_reranker()
        start = time.perf_counter()
        rerank_scores = await anyio.to_thread.run_sync(
            lambda: reranker.score(query, [pool_by_id[cid]["text"] for cid in selected_ids])
        )
        timings["rerank_ms"] = (time.perf_counter() - start) * 1000

        if rerank_scores is not None:
            order = sorted(
                range(len(selected_ids)), key=lambda i: rerank_scores[i], reverse=True
            )
        else:
            # No cross-encoder — fall back to cosine, which is a real ordering
            # (the old code re-derived the cosine Chroma had already sorted by
            # and called that a rerank).
            order = sorted(
                range(len(selected_ids)),
                key=lambda i: pool_by_id[selected_ids[i]]["score"],
                reverse=True,
            )

        # --- Stage 6: materialise, dedupe, apply the relevance floors ----
        chunks: list[RetrievedChunk] = []
        seen_texts: set[str] = set()
        dropped_by_gate = 0
        dropped_by_cosine = 0
        # Every chunk BM25 vouched for — including ones the vector search also
        # returned. Membership here, not entry["source"], is the lexical signal:
        # Stage 2 only records source="bm25" for chunks the vector search MISSED,
        # so a strong BM25 hit that both engines found still reads as "vector".
        lexical_ids = set(lexical_ranking)
        for i in order:
            chunk_id = selected_ids[i]
            entry = pool_by_id[chunk_id]
            dedup_key = entry["text"].strip()
            if not dedup_key or dedup_key in seen_texts:
                continue
            # Cosine floor — but only for candidates the *vector* search found.
            # A BM25 hit earns its place by exact-token match, which is the whole
            # reason hybrid search exists: "SMOWL", "VAS" and error codes are
            # precisely the queries where cosine is weakest. Applying a cosine
            # floor to a lexical hit silently cancels that rescue, so the
            # cross-encoder gate below is what judges those instead.
            if chunk_id not in lexical_ids and entry["score"] < settings.min_relevance_score:
                dropped_by_cosine += 1
                continue
            # Absolute relevance gate on the cross-encoder score. Cosine cannot
            # make this call: text that merely shares vocabulary with the query
            # sits comfortably above the cosine floor. The cross-encoder reads
            # query and passage *together*, so its logit reflects whether the
            # passage actually answers the question.
            #
            # Measured on this KB: for a question the articles do document,
            # surviving chunks score roughly -5..+4; for one they do not
            # ("how do I submit an assignment in Moodle?"), every chunk scored
            # ≈ -11 while still passing the cosine floor at 0.34-0.41. Dropping
            # those is what stops the model being handed plausible-looking but
            # unrelated context and answering from it.
            if rerank_scores is not None and rerank_scores[i] < settings.rerank_min_score:
                dropped_by_gate += 1
                continue
            seen_texts.add(dedup_key)
            meta = entry["meta"]
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=entry["text"],
                    article_id=meta.get("article_id", ""),
                    category=meta.get("category"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=entry["score"],
                    rrf_score=fused.get(chunk_id, 0.0),
                    rerank_score=(
                        float(rerank_scores[i]) if rerank_scores is not None else None
                    ),
                    retrieval_source=entry["source"],
                    _raw_meta=meta,
                )
            )
            if len(chunks) >= top_k:
                break

        # An empty result after the gate dropped everything is a real signal, not
        # a bug: the KB has nothing that answers this question, and the prompt
        # turns that into a decline rather than an answer improvised from
        # unrelated context. Log it at INFO so it is visible without debug mode.
        if dropped_by_gate and not chunks:
            logger.info(
                "Retrieval returned no context for %r — all %d shortlisted chunk(s) "
                "scored below the cross-encoder relevance gate (%.1f). The knowledge "
                "base likely does not cover this question.",
                query,
                dropped_by_gate,
                settings.rerank_min_score,
            )

        if debug:
            self._log_retrieval_debug(query, pool_by_id, vector_ranking,
                                      lexical_ranking, chunks, timings,
                                      dropped_by_gate, dropped_by_cosine)
        else:
            logger.info(
                "Retrieved %d chunks (pool=%d vector=%d bm25=%d gated=%d "
                "below_cosine=%d) in %.0fms",
                len(chunks),
                len(pool_by_id),
                len(vector_ranking),
                len(lexical_ranking),
                dropped_by_gate,
                dropped_by_cosine,
                sum(timings.values()),
            )
        return chunks

    def _log_retrieval_debug(
        self,
        query: str,
        pool_by_id: dict[str, dict[str, Any]],
        vector_ranking: list[str],
        lexical_ranking: list[str],
        chunks: list["RetrievedChunk"],
        timings: dict[str, float],
        dropped_by_gate: int = 0,
        dropped_by_cosine: int = 0,
    ) -> None:
        """Dump the full retrieval trace.

        Only reachable when ``retrieval_debug_active`` is true, which is forced
        off in production — this prints raw user queries and chunk bodies.
        """
        lines = [
            "",
            "=" * 72,
            f"[retrieval] query   : {query!r}",
            f"[retrieval] pool    : {len(pool_by_id)} candidates "
            f"(vector={len(vector_ranking)}, bm25={len(lexical_ranking)})",
            "[retrieval] timings : "
            + ", ".join(f"{k}={v:.0f}ms" for k, v in timings.items()),
        ]
        if dropped_by_gate:
            lines.append(
                f"[retrieval] gated   : {dropped_by_gate} chunk(s) dropped below "
                f"rerank_min_score={self.settings.rerank_min_score:+.1f}"
            )
        if dropped_by_cosine:
            lines.append(
                f"[retrieval] cos-floor: {dropped_by_cosine} vector-only chunk(s) "
                f"dropped below min_relevance_score="
                f"{self.settings.min_relevance_score:.2f} (BM25 hits are exempt)"
            )
        if lexical_ranking:
            lines.append(f"[retrieval] bm25 top: {lexical_ranking[:5]}")
        lines.append(f"[retrieval] vector top: {vector_ranking[:5]}")
        lines.append(f"[retrieval] final   : {len(chunks)} chunk(s)")
        for rank, chunk in enumerate(chunks, 1):
            rerank = (
                f" rerank={chunk.rerank_score:+.3f}"
                if chunk.rerank_score is not None
                else " rerank=n/a"
            )
            preview = chunk.text[:120].replace("\n", " ")
            lines.append(
                f"   {rank}. cos={chunk.score:.3f} rrf={chunk.rrf_score:.4f}{rerank} "
                f"src={chunk.retrieval_source} art={chunk.article_id} "
                f"[{chunk.chunk_id}]"
            )
            lines.append(f"      {preview!r}")
        lines.append("=" * 72)
        logger.info("\n".join(lines))

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
            # Configurable floor (was a hardcoded 0.3). Screenshots are only
            # useful when they actually match the question — a weak image is
            # worse than none, because the prompt invites the model to point at it.
            if score < self.settings.min_image_score:
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

        The query is spell-corrected first (see ``rag.lexical.rewrite_query``)
        so a typo'd acronym still matches, and the result is served from a
        short-TTL cache when the same question repeats.

        When ``db`` is supplied the survivors are hydrated from PostgreSQL
        before returning; without it the caller gets id-only records and can
        hydrate later (or not at all, e.g. in tests).
        """
        if self.settings.query_rewrite_enabled:
            query = rewrite_query(query)

        cache_key = (query.strip().lower(), category or "")
        cached = self._cache_get(cache_key)
        if cached is not None:
            chunks, images = cached
            if db is not None:
                # Hydration mutates the records, so serve copies — otherwise a
                # later request would mutate the cached objects in place.
                chunks = [replace(c) for c in chunks]
                images = [replace(i) for i in images]
                await self.hydrate_chunks(db, chunks)
                await self.hydrate_images(db, images)
            return chunks, images

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

        # Cache the un-hydrated results: hydration is a cheap cached DB read,
        # and caching pre-hydration keeps Postgres the source of truth for
        # display fields even on a cache hit.
        self._cache_put(cache_key, chunks, images)

        if db is not None:
            # Both hydrations are independent single-table reads, but they share
            # one AsyncSession — SQLAlchemy sessions are not concurrency-safe, so
            # these must run sequentially, not in a task group.
            chunks = [replace(c) for c in chunks]
            images = [replace(i) for i in images]
            await self.hydrate_chunks(db, chunks)
            await self.hydrate_images(db, images)

        return chunks, images

    # ------------------------------------------------------------------
    # Retrieval cache — TTL, keyed by (normalised query, category)
    # ------------------------------------------------------------------

    def _cache_get(
        self, key: tuple[str, str]
    ) -> Optional[tuple[list[RetrievedChunk], list[RetrievedImage]]]:
        ttl = self.settings.retrieval_cache_ttl
        if ttl <= 0:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_at, chunks, images = entry
        if (time.monotonic() - stored_at) > ttl:
            self._cache.pop(key, None)
            return None
        # Refresh recency for the LRU eviction below.
        self._cache.move_to_end(key)
        logger.info("Retrieval cache HIT for %r", key[0])
        return chunks, images

    def _cache_put(
        self,
        key: tuple[str, str],
        chunks: list[RetrievedChunk],
        images: list[RetrievedImage],
    ) -> None:
        ttl = self.settings.retrieval_cache_ttl
        if ttl <= 0:
            return
        self._cache[key] = (time.monotonic(), chunks, images)
        self._cache.move_to_end(key)
        while len(self._cache) > self.settings.retrieval_cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Drop cached retrievals — called after a re-ingest changes the corpus."""
        self._cache.clear()

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
