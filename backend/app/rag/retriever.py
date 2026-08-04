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

import re
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
from backend.app.rag.query_processing import ProcessedQuery, process_query
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

# The "[Title]" header the chunker prepends to every chunk body. Stripped when
# merging adjacent chunks so the header appears once per block, not once per
# member (see HybridRetriever._group_adjacent).
_TITLE_HEADER_RE = re.compile(r"^\s*\[[^\]]{1,200}\]\s*\n?")


def _strip_title_header(text: str) -> str:
    return _TITLE_HEADER_RE.sub("", text, count=1)


# Test-only override for the ordering blend; None means "use the configured
# rerank_order_weight". scripts/_diag_ordering.py sweeps this to compare
# strategies without mutating settings. See the Stage 5 ordering comment.
_ORDER_RERANK_WEIGHT: Optional[float] = None


def _min_max_normalise(values: list[float]) -> list[float]:
    """Scale to 0..1. All-equal input maps to 0.5 rather than dividing by zero."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


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

    def _process_query(self, query: str) -> ProcessedQuery:
        """Run query preprocessing: normalize, expand synonyms, generate variants.

        Returns a ProcessedQuery with all forms the retrieval stages need. This
        is where fuzzy correction happens ("smwol" → "SMOWL"), synonym expansion
        ("LMS" → "LMS Moodle Learning Management System"), and paraphrase
        generation ("I forgot my password" → ["I forgot my password", "How do I
        reset my password?", ...]).

        The KB's own vocabulary is passed for fuzzy matching so an out-of-
        vocabulary token is corrected to the nearest *corpus* term, which is
        what makes it safe: the only substitutions available are words this
        knowledge base actually uses.
        """
        settings = self.settings
        vocab = get_lexical_index().vocabulary() if settings.lexical_fuzzy_enabled else None
        return process_query(
            query,
            enable_normalization=settings.query_normalization_enabled,
            enable_synonyms=settings.query_synonym_expansion_enabled,
            enable_multi_query=settings.multi_query_enabled,
            max_variants=settings.multi_query_variants,
            fuzzy_vocabulary=vocab,
        )

    async def _vector_search_variants(
        self,
        variants: list[str],
        where_filter: dict[str, Any] | None,
        n_results: int,
        primary_embedding: Optional[list[float]] = None,
    ) -> list[tuple[str, list[float], dict[str, Any]]]:
        """Embed and search for each variant in parallel.

        Returns [(variant_text, embedding, chroma_results), ...]. The first
        entry is always the primary query (variants[0] == the normalised text)
        and its embedding is reused when the caller already computed it.
        """
        embeddings: list[Optional[list[float]]] = [None] * len(variants)
        if primary_embedding is not None:
            embeddings[0] = primary_embedding

        # Embed all variants that don't have an embedding yet.
        to_embed = [v for i, v in enumerate(variants) if embeddings[i] is None]
        if to_embed:
            fresh = await self.embedding_service.embed_texts_async(to_embed)
            idx = 0
            for i in range(len(variants)):
                if embeddings[i] is None:
                    embeddings[i] = fresh[idx]
                    idx += 1

        # Search all variants concurrently. Each Chroma call is sync, so it
        # goes to a worker thread; the task group is what overlaps them, which
        # is what keeps multi-query inside the latency budget — 4 variants cost
        # roughly one variant's wall-clock, not four.
        results: list[dict[str, Any]] = [{}] * len(embeddings)

        async def search_into(slot: int, embedding: list[float]) -> None:
            results[slot] = await anyio.to_thread.run_sync(
                lambda: query_text_collection(
                    query_embedding=embedding,
                    n_results=n_results,
                    where=where_filter,
                    include_embeddings=True,
                )
            )

        async with anyio.create_task_group() as tg:
            for i, emb in enumerate(embeddings):
                tg.start_soon(search_into, i, emb)  # type: ignore[arg-type]

        return [
            (variants[i], embeddings[i], results[i])  # type: ignore[misc]
            for i in range(len(variants))
        ]

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

    def _rrf_fuse_weighted(
        self, rankings: list[tuple[list[str], float]]
    ) -> dict[str, float]:
        """Weighted RRF over arbitrarily many rankings.

        ``rankings`` is [(ordered_chunk_ids, weight), ...]. Generalises
        :meth:`_rrf_fuse` to the multi-query case, where each query variant
        contributes its own vector ranking (and the expanded query its own BM25
        ranking).

        Why this shape helps recall: a chunk found at rank 8 by three different
        paraphrases accumulates more fused score than one found at rank 3 by a
        single phrasing. Agreement across rewrites is a stronger signal than one
        engine's confidence, and it is exactly the signal that makes retrieval
        insensitive to how the question was worded.
        """
        k = self.settings.rrf_k
        fused: dict[str, float] = {}
        for ranking, weight in rankings:
            if weight <= 0.0:
                continue
            for rank, chunk_id in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + (weight / (k + rank + 1))
        return fused

    async def retrieve_text(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        query_embedding: Optional[list[float]] = None,
        processed: Optional[ProcessedQuery] = None,
    ) -> list[RetrievedChunk]:
        """Hybrid multi-query retrieval: vector×N + BM25 → RRF → MMR → rerank.

        Each stage narrows a wider pool than the last stage needs, which is the
        point: MMR can only add diversity if it has spare candidates to choose
        between, and the cross-encoder can only fix ordering if the right chunk
        is somewhere in its shortlist.

        Which text each engine sees is deliberate and not interchangeable:

        * **vector** — the normalised query, plus one search per paraphrase.
          A bi-encoder wants fluent natural language; synonym-stuffed text
          embeds to a vague centroid between several topics.
        * **BM25** — the synonym-expanded query. More exact tokens is strictly
          better for a term-frequency scorer, and it is how "LMS login" reaches
          a chunk whose title says Moodle.
        * **cross-encoder** — the user's ORIGINAL wording. Its logit is the
          off-topic gate, and it only means "does this passage answer the
          question" while the question is still a real question.
        """
        settings = self.settings
        top_k = top_k or settings.top_k_retrieval
        debug = settings.retrieval_debug_active
        timings: dict[str, float] = {}

        if processed is None:
            processed = self._process_query(query)

        where_filter: dict[str, Any] | None = {"category": category} if category else None
        pool = max(settings.retrieval_candidate_pool, top_k * 3)

        # --- Stage 1: vector candidates, one search per query variant --------
        # variants[0] is the normalised query and carries full weight; the
        # paraphrases are weighted lower (see multi_query_variant_weight) so
        # they can rescue a missed chunk without outvoting the real question.
        variants = processed.variants or [processed.normalized or query]
        if not settings.multi_query_enabled:
            variants = variants[:1]

        start = time.perf_counter()
        variant_results = await self._vector_search_variants(
            variants, where_filter, pool, primary_embedding=query_embedding
        )
        timings["vector_ms"] = (time.perf_counter() - start) * 1000

        # id → (text, meta, embedding, cosine score). The score kept is the best
        # any variant achieved: a chunk that a paraphrase matched strongly is
        # genuinely that relevant to the user's intent, and using the primary
        # query's weaker cosine would then trip the relevance floor below.
        pool_by_id: dict[str, dict[str, Any]] = {}
        vector_rankings: list[tuple[list[str], float]] = []
        primary_embedding: Optional[list[float]] = None

        for idx, (variant, embedding, results) in enumerate(variant_results):
            if idx == 0:
                primary_embedding = embedding
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            embeddings_raw = results.get("embeddings", [])
            embeddings = embeddings_raw[0] if len(embeddings_raw) else []

            for i, chunk_id in enumerate(ids):
                score = self._distance_to_score(distances[i])
                existing = pool_by_id.get(chunk_id)
                if existing is None:
                    pool_by_id[chunk_id] = {
                        "text": documents[i] or "",
                        "meta": metadatas[i] or {},
                        "embedding": embeddings[i] if i < len(embeddings) else None,
                        "score": score,
                        "source": "vector" if idx == 0 else "variant",
                        "found_by": [variant],
                    }
                else:
                    if score > existing["score"]:
                        existing["score"] = score
                    existing["found_by"].append(variant)

            weight = (
                1.0 if idx == 0 else settings.multi_query_variant_weight
            ) * (
                settings.hybrid_vector_weight if settings.hybrid_search_enabled else 1.0
            )
            vector_rankings.append((list(ids), weight))

        if primary_embedding is None:
            primary_embedding = query_embedding
        # Kept for the debug log and for callers that inspect the primary order.
        vector_ranking = vector_rankings[0][0] if vector_rankings else []

        # --- Stage 2: lexical (BM25) candidates -----------------------------
        # Uses the synonym-expanded text, which is the whole point of building
        # it: BM25 can only match tokens that are literally present.
        lexical_ranking: list[str] = []
        if settings.hybrid_search_enabled:
            lexical_query = processed.lexical or processed.normalized or query
            start = time.perf_counter()
            lexical_hits = await anyio.to_thread.run_sync(
                lambda: get_lexical_index().search(
                    lexical_query, k=pool, fuzzy=settings.lexical_fuzzy_enabled
                )
            )
            timings["bm25_ms"] = (time.perf_counter() - start) * 1000
            lexical_ranking = [chunk_id for chunk_id, _ in lexical_hits]

            # BM25 can surface chunks no vector variant returned. Pull their
            # text/embedding so they can compete on equal footing.
            missing = [cid for cid in lexical_ranking if cid not in pool_by_id]
            if missing:
                extra = await anyio.to_thread.run_sync(
                    lambda: fetch_text_chunks_by_id(missing, include_embeddings=True)
                )
                extra_ids = extra["ids"]
                extra_docs = extra["documents"]
                extra_meta = extra["metadatas"]
                extra_emb = extra["embeddings"]
                q_arr = np.asarray(primary_embedding, dtype=np.float32)
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
                        "found_by": ["bm25"],
                    }

        if not pool_by_id:
            if debug:
                logger.info("[retrieval] query=%r → no candidates at all", query)
            return []

        # --- Stage 3: fuse ---------------------------------------------------
        rankings = list(vector_rankings)
        if lexical_ranking:
            rankings.append((lexical_ranking, settings.hybrid_bm25_weight))
        elif len(rankings) == 1:
            # Vector-only, single query: rank position IS the order. Rescale to
            # weight 1.0 so rrf_score stays comparable with a hybrid query's.
            rankings = [(rankings[0][0], 1.0)]
        fused = self._rrf_fuse_weighted(rankings)

        fused_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        fused_ids = [cid for cid in fused_ids if cid in pool_by_id]

        # --- Stage 4: shortlist for the cross-encoder --------------------
        # Straight top-N by fused RRF score. Deliberately NOT MMR, which used to
        # run here.
        #
        # MMR optimises diversity; the cross-encoder judges relevance. Putting
        # MMR first means candidates are discarded for resembling an
        # already-picked chunk *before* the only stage that can tell whether they
        # answer the question has run — and the chunks that most resemble each
        # other are the sibling chunks of the one article that documents the
        # topic, which is exactly the material a procedural answer needs.
        #
        # Measured on "Learning Management System sign in": 1_chunk_0, the best
        # chunk in the KB for that query, never reached the shortlist at all
        # because 1_chunk_1 was selected first and MMR judged it redundant. Over
        # 20 paraphrase/synonym queries, swapping MMR for fused rank moved
        # recall@1 18/20 → 19/20 and recall@k 19/20 → 20/20 with off-topic
        # leakage unchanged at 0/6. It is also strictly cheaper: no pairwise
        # similarity matrix, and it drops the embedding-present precondition that
        # silently excluded BM25-only hits lacking a stored vector.
        #
        # Redundancy is still handled, but downstream and by mechanisms that do
        # not cost relevance: exact-text dedup in stage 6, and _group_adjacent
        # merging sibling chunks into one block at format time.
        selected_ids = fused_ids[: settings.rerank_shortlist]

        # --- Stage 5: cross-encoder rerank ------------------------------
        # Scored against the user's ORIGINAL wording plus ONE fluent paraphrase,
        # keeping the best score per chunk. Never the synonym-expanded text:
        # the cross-encoder's logit doubles as the off-topic gate, and
        # synonym-stuffed input scores like keyword soup, which would make the
        # gate (the thing protecting off-topic precision) meaningless.
        #
        # Why more than one form at all: this cross-encoder is startlingly
        # sensitive to surface wording. Measured on this KB, the *same* six
        # login passages score +5.5/+7.6 for "LMS login" but -8.7/-10.8 for
        # "Moodle login" — the article is titled "How to login to LMS", and the
        # user saying "Moodle" instead falls off a cliff. One phrasing means the
        # gate is partly a test of whether the user guessed the article's own
        # vocabulary, which is exactly what this work is meant to remove.
        #
        # Why exactly two, and not more: measured over 21 on-topic and 8
        # off-topic queries, 1 form let 2 real questions through the gate
        # (90.5% recall) while 2 forms reached 21/21 with off-topic precision
        # still 8/8. At 3 forms, precision broke — "What is the weather
        # tomorrow?" and "Explain quantum entanglement" cleared the gate at
        # -7.0/-7.1. Each extra phrasing is another lottery ticket against a
        # fixed threshold, so the count is a recall/precision dial and 2 is
        # where it is measurably best. It is also cheap: reranking is ~0.4s for
        # 16 passages, so the second pass keeps the query near 1s.
        reranker = get_reranker()
        rerank_texts = [pool_by_id[cid]["text"] for cid in selected_ids]
        # The NORMALISED query, not the raw original: typo correction is exactly
        # what makes a question scoreable here. "moddle login" scores below the
        # gate on every chunk, while "Moodle login" is at least a real question.
        # Normalisation only fixes spelling and expands abbreviations, so the
        # text stays a fluent question — unlike processed.lexical.
        rerank_query = processed.normalized.strip() or processed.original or query
        rerank_forms = [rerank_query]
        # Then alternative phrasings, in generation order. Variants that merely
        # restate the normalised form add a second identical score and waste the
        # slot, so exact repeats are skipped — the useful variant is the one that
        # substitutes the KB's canonical term ("Moodle login" → "LMS login").
        #
        # Generation order is deliberate, and ranking these by "most novel
        # vocabulary" instead is a measured mistake: because the scores are
        # max-pooled, whichever form scores a chunk highest wins outright, so a
        # variant that drifts to a neighbouring topic drags the whole result with
        # it. Selecting for novelty selects for exactly that drift — tried on
        # this KB, it sent "Outlook login" to the Microsoft-365 MFA article for
        # all five slots and dropped recall@1 from 27/28 to 21/28. The variant
        # generator emits closest-paraphrase-first, which is the property worth
        # keeping.
        seen_forms = {rerank_query.lower()}
        if settings.rerank_query_forms > 1:
            for variant in processed.variants:
                if len(rerank_forms) >= settings.rerank_query_forms:
                    break
                key = (variant or "").strip().lower()
                if key and key not in seen_forms:
                    seen_forms.add(key)
                    rerank_forms.append(variant)

        start = time.perf_counter()

        def _score_all() -> Optional[list[float]]:
            best: Optional[list[float]] = None
            for form in rerank_forms:
                scores = reranker.score(form, rerank_texts)
                if scores is None:
                    return None
                if best is None:
                    best = [float(s) for s in scores]
                else:
                    best = [max(b, float(s)) for b, s in zip(best, scores)]
            return best

        rerank_scores = await anyio.to_thread.run_sync(_score_all)
        timings["rerank_ms"] = (time.perf_counter() - start) * 1000

        if rerank_scores is not None:
            # Ordering signal. The cross-encoder score alone is the obvious
            # choice and is what this used to do, but it conflates two jobs the
            # model is not equally good at: deciding whether a passage answers
            # the question (excellent — it gates off-topic queries perfectly) and
            # ranking two passages that both do (noisy). Measured on "Can't
            # access LMS", it put the Microsoft Teams chunk at +1.27 above the
            # actual "How to login to LMS" chunk at -1.43, pushing the right
            # answer past top_k. RRF had it ranked #1.
            #
            # So blend: the cross-encoder still gates (below), but the order is a
            # weighted mix of both signals. They are on incomparable scales —
            # logits roughly -11..+8, RRF scores ~0.01..0.04 — so each is
            # min-max normalised across this shortlist before mixing.
            weight = (
                _ORDER_RERANK_WEIGHT
                if _ORDER_RERANK_WEIGHT is not None
                else settings.rerank_order_weight
            )
            if weight >= 1.0:
                order = sorted(
                    range(len(selected_ids)),
                    key=lambda i: rerank_scores[i],
                    reverse=True,
                )
            else:
                rrf_vals = [fused.get(cid, 0.0) for cid in selected_ids]
                norm_ce = _min_max_normalise(rerank_scores)
                norm_rrf = _min_max_normalise(rrf_vals)
                combined = [
                    weight * norm_ce[i] + (1.0 - weight) * norm_rrf[i]
                    for i in range(len(selected_ids))
                ]
                order = sorted(
                    range(len(selected_ids)),
                    key=lambda i: combined[i],
                    reverse=True,
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
                                      dropped_by_gate, dropped_by_cosine,
                                      processed=processed,
                                      variant_rankings=vector_rankings,
                                      variants=variants,
                                      rerank_forms=rerank_forms,
                                      rerank_scores=rerank_scores,
                                      selected_ids=selected_ids,
                                      fused=fused)
        else:
            logger.info(
                "Retrieved %d chunks (pool=%d variants=%d vector=%d bm25=%d "
                "gated=%d below_cosine=%d) in %.0fms",
                len(chunks),
                len(pool_by_id),
                len(variants),
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
        *,
        processed: Optional[ProcessedQuery] = None,
        variant_rankings: Optional[list[tuple[list[str], float]]] = None,
        variants: Optional[list[str]] = None,
        rerank_forms: Optional[list[str]] = None,
        rerank_scores: Optional[list[float]] = None,
        selected_ids: Optional[list[str]] = None,
        fused: Optional[dict[str, float]] = None,
    ) -> None:
        """Dump the full retrieval trace.

        Covers every stage that can change the answer: how the query was
        rewritten, what each variant retrieved, what BM25 found, how RRF ranked
        them, and what the cross-encoder scored — so a wrong result can be
        attributed to a stage instead of guessed at.

        Only reachable when ``retrieval_debug_active`` is true, which is forced
        off in production — this prints raw user queries and chunk bodies.
        """
        lines = [
            "",
            "=" * 72,
            f"[retrieval] query   : {query!r}",
        ]

        # --- query preprocessing ---
        if processed is not None:
            if processed.normalized != processed.original:
                lines.append(f"[retrieval] normalized: {processed.normalized!r}")
            if processed.corrections:
                lines.append(
                    "[retrieval] spell   : "
                    + ", ".join(f"{k!r}→{v!r}" for k, v in processed.corrections.items())
                )
            if processed.expansions:
                lines.append("[retrieval] synonyms:")
                for term, added in processed.expansions.items():
                    lines.append(f"              {term!r} → {list(added)}")
            if processed.intents:
                lines.append(f"[retrieval] intents : {processed.intents}")
            if processed.lexical != processed.normalized:
                lines.append(f"[retrieval] bm25 qry: {processed.lexical!r}")

        # --- per-variant vector hits ---
        if variants and variant_rankings:
            lines.append(f"[retrieval] variants: {len(variants)}")
            for i, (variant, (ranking, weight)) in enumerate(
                zip(variants, variant_rankings)
            ):
                tag = "primary" if i == 0 else f"variant{i}"
                lines.append(
                    f"   {tag:9s} w={weight:.2f} {variant!r} → {len(ranking)} hits "
                    f"top={ranking[:3]}"
                )

        lines.append(
            f"[retrieval] pool    : {len(pool_by_id)} candidates "
            f"(vector={len(vector_ranking)}, bm25={len(lexical_ranking)})"
        )
        if lexical_ranking:
            lines.append(f"[retrieval] bm25 top: {lexical_ranking[:5]}")

        # --- fusion ---
        if fused:
            top_fused = sorted(fused, key=lambda c: fused[c], reverse=True)[:5]
            lines.append(
                "[retrieval] rrf top : "
                + ", ".join(f"{cid}={fused[cid]:.4f}" for cid in top_fused)
            )

        # --- reranking ---
        if rerank_forms:
            lines.append(f"[retrieval] rerank as: {rerank_forms}")
        if selected_ids is not None and rerank_scores is not None:
            shown = sorted(
                range(len(selected_ids)), key=lambda i: rerank_scores[i], reverse=True
            )
            lines.append(
                f"[retrieval] rerank  : {len(selected_ids)} shortlisted, "
                f"gate={self.settings.rerank_min_score:+.1f}"
            )
            for i in shown:
                kept = "keep" if rerank_scores[i] >= self.settings.rerank_min_score else "DROP"
                lines.append(
                    f"      {kept} {rerank_scores[i]:+7.2f}  {selected_ids[i]}"
                )
        elif selected_ids is not None:
            lines.append(
                f"[retrieval] rerank  : unavailable — {len(selected_ids)} chunk(s) "
                "ordered by cosine instead"
            )

        lines.append(
            "[retrieval] timings : "
            + ", ".join(f"{k}={v:.0f}ms" for k, v in timings.items())
        )
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
        lines.append(f"[retrieval] vector top: {vector_ranking[:5]}")
        lines.append(f"[retrieval] final   : {len(chunks)} chunk(s)")
        for rank, chunk in enumerate(chunks, 1):
            rerank = (
                f" rerank={chunk.rerank_score:+.3f}"
                if chunk.rerank_score is not None
                else " rerank=n/a"
            )
            found_by = pool_by_id.get(chunk.chunk_id, {}).get("found_by") or []
            preview = chunk.text[:120].replace("\n", " ")
            lines.append(
                f"   {rank}. cos={chunk.score:.3f} rrf={chunk.rrf_score:.4f}{rerank} "
                f"src={chunk.retrieval_source} art={chunk.article_id} "
                f"[{chunk.chunk_id}]"
            )
            if found_by:
                lines.append(f"      found_by={found_by}")
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

        The query is preprocessed first (see ``rag.query_processing``) into the
        several forms the retrieval stages need — normalised text for the
        vector search, synonym-expanded text for BM25, paraphrases for
        multi-query, and the untouched original for the cross-encoder — and the
        result is served from a short-TTL cache when the same question repeats.

        When ``db`` is supplied the survivors are hydrated from PostgreSQL
        before returning; without it the caller gets id-only records and can
        hydrate later (or not at all, e.g. in tests).
        """
        processed = self._process_query(query) if self.settings.query_rewrite_enabled else None
        # Cache on the normalised form, so "moddle login" and "Moodle login"
        # share an entry — the whole point of normalisation is that they are
        # the same question.
        cache_text = processed.normalized if processed else query
        cache_key = (cache_text.strip().lower(), category or "")
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

        # The shared embedding is of the *normalised* text: it is what the
        # primary vector search and the MMR/cosine stages are scored against.
        embed_text = processed.normalized if processed else query
        if query_embedding is None:
            query_embedding = await self.embedding_service.embed_query_async(embed_text)

        chunks: list[RetrievedChunk] = []
        images: list[RetrievedImage] = []

        async with anyio.create_task_group() as tg:
            async def _text() -> None:
                nonlocal chunks
                chunks = await self.retrieve_text(
                    query,
                    category=category,
                    query_embedding=query_embedding,
                    processed=processed,
                )

            async def _images() -> None:
                nonlocal images
                images = await self.retrieve_images(
                    embed_text, category=category, query_embedding=query_embedding
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

        if self.settings.group_adjacent_chunks:
            chunks = self._group_adjacent(chunks)

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

    @staticmethod
    def _group_adjacent(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Merge consecutive chunks of the same article into one block.

        Chunking splits an article mid-procedure, so retrieving chunks 2 and 3
        of a numbered walkthrough and presenting them as two separate excerpts
        invites the model to read them as unrelated fragments — or to repeat the
        overlapping text. Stitching them back into one block restores the
        original reading order and the step numbering that goes with it.

        Only *adjacent* indices are merged. Chunks 1 and 4 of the same article
        have real content missing between them, and joining them would imply a
        continuity that is not there.

        Retrieval order is preserved: a merged block takes the position of its
        best-ranked member, so the reranker's judgement still decides what the
        model reads first. The relevance fields (score, rrf_score,
        rerank_score) are likewise taken from that best member — they describe
        why this material was retrieved, and the best member is what earned it.
        """
        if len(chunks) < 2:
            return chunks

        # Group by article, remembering each chunk's rank so the merged block
        # can be placed back at its best member's position.
        by_article: dict[str, list[tuple[int, RetrievedChunk]]] = {}
        for rank, chunk in enumerate(chunks):
            by_article.setdefault(chunk.article_id, []).append((rank, chunk))

        merged: list[tuple[int, RetrievedChunk]] = []
        for members in by_article.values():
            members.sort(key=lambda pair: pair[1].chunk_index)
            run: list[tuple[int, RetrievedChunk]] = []

            def flush(run: list[tuple[int, RetrievedChunk]]) -> None:
                if not run:
                    return
                best_rank, best = min(run, key=lambda pair: pair[0])
                if len(run) == 1:
                    merged.append((best_rank, best))
                    return
                # The chunker prepends a "[Title]" header to every chunk, so a
                # naive join repeats it once per member. Keep the first (it
                # frames the block) and strip the rest, which would otherwise
                # read as several articles run together.
                parts = [run[0][1].text]
                parts.extend(_strip_title_header(c.text) for _, c in run[1:])
                merged.append(
                    (
                        best_rank,
                        replace(
                            best,
                            text="\n\n".join(p for p in parts if p.strip()),
                            chunk_index=run[0][1].chunk_index,
                        ),
                    )
                )

            for pair in members:
                if run and pair[1].chunk_index != run[-1][1].chunk_index + 1:
                    flush(run)
                    run = []
                run.append(pair)
            flush(run)

        merged.sort(key=lambda pair: pair[0])
        return [chunk for _, chunk in merged]

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
