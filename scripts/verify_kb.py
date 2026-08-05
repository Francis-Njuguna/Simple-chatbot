"""Post-reingest integrity check. Exits non-zero on any failure.

Answers one question: is the knowledge base actually usable? Checks structural
integrity (counts, orphans, duplicates) and then that retrieval really returns
something for known-good queries — a populated store that retrieves nothing is
still a broken chatbot, so counts alone are not enough.

Designed to be chained after ingestion:
    ./.venv/Scripts/python.exe -u scripts/reingest.py && \
    ./.venv/Scripts/python.exe -u scripts/verify_kb.py

Import ordering / KMP_DUPLICATE_LIB_OK: see scripts/reingest.py's docstring.
Both matter here too — this process loads torch and chromadb in one go.
"""

import asyncio
import logging
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, ".")

# (query, substring that must appear in at least one retrieved chunk).
# Substrings are matched case-insensitively and kept deliberately short so the
# check survives harmless rewording of the source articles.
SMOKE_QUERIES: list[tuple[str, str]] = [
    ("How do I reset my student portal password?", "password"),
    ("What is SMOWL?", "smowl"),
    ("smwol camera not working", "smowl"),          # typo → tests query rewriting
    ("How do I log in to the LMS?", "moodle"),      # synonym → tests title header
]

# A question the KB does not document. Must return NOTHING, or the cross-encoder
# gate is too loose and the model will be handed irrelevant context to improvise
# from — the original bug.
NEGATIVE_QUERY = "What is the capital of France?"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


class Checker:
    """Accumulates pass/fail results so every check runs before exiting."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""), flush=True)
        if not ok:
            self.failures.append(label)
        return ok


async def main() -> int:
    _setup_logging()
    c = Checker()

    # Embedding model first — see the docstring.
    from backend.app.rag.embeddings import get_embedding_service

    embedder = get_embedding_service()
    await embedder.embed_query_async("warmup")

    from backend.app.database.chroma import count_by_source_type

    counts = count_by_source_type()
    text_n = counts.get("text", 0)
    image_n = counts.get("image", 0)

    print("\n=== 1. Chroma vector counts ===", flush=True)
    print(f"  text vectors  : {text_n}", flush=True)
    print(f"  image vectors : {image_n}", flush=True)
    c.check(text_n > 0, "Chroma text collection is non-empty", f"{text_n} vectors")

    # One full scan serves sections 2 and 3 — ids, documents and metadatas all
    # come back together.
    from backend.app.database.chroma import fetch_all_text_documents

    corpus = fetch_all_text_documents()
    chroma_ids = corpus["ids"]
    docs = corpus["documents"]
    chroma_meta = corpus["metadatas"]

    # Structural integrity against Postgres, the source of truth for metadata.
    print("\n=== 2. PostgreSQL metadata ===", flush=True)
    from sqlalchemy import func, select

    from backend.app.database.models import DocumentMetadata
    from backend.app.database.session import async_session_factory

    async with async_session_factory() as db:
        article_n = (
            await db.execute(select(func.count()).select_from(DocumentMetadata))
        ).scalar_one()
        print(f"  articles      : {article_n}", flush=True)
        c.check(article_n > 0, "Postgres has article metadata")

        # Every Chroma chunk must resolve to a Postgres row, or hydration yields
        # chunks with no title/URL and the answer silently loses its citation.
        chroma_article_ids = {
            str(m.get("article_id")) for m in chroma_meta if m and m.get("article_id")
        }
        pg_article_ids = {
            str(r)
            for r in (
                await db.execute(select(DocumentMetadata.article_id))
            ).scalars().all()
        }

        orphans = chroma_article_ids - pg_article_ids
        c.check(
            not orphans,
            "every Chroma chunk resolves to Postgres metadata",
            f"{len(orphans)} orphaned article_id(s): {sorted(orphans)[:5]}"
            if orphans
            else "no orphans",
        )

        unindexed = pg_article_ids - chroma_article_ids
        c.check(
            not unindexed,
            "every Postgres article has at least one vector",
            f"{len(unindexed)} article(s) with no chunks: {sorted(unindexed)[:5]}"
            if unindexed
            else "all indexed",
        )

        # chunk_count is written during ingestion; if it disagrees with what is
        # actually in Chroma then the two stores have drifted apart.
        declared = (
            await db.execute(select(func.sum(DocumentMetadata.chunk_count)))
        ).scalar() or 0
        c.check(
            declared == text_n,
            "Postgres chunk_count total matches Chroma vector count",
            f"Postgres declares {declared}, Chroma holds {text_n}",
        )

    print("\n=== 3. Embedding integrity ===", flush=True)
    c.check(
        len(chroma_meta) == text_n and len(chroma_ids) == text_n,
        "metadata/id rows match vector count",
        f"{len(chroma_ids)} ids, {len(chroma_meta)} metadata, {text_n} vectors",
    )
    c.check(
        len(set(chroma_ids)) == len(chroma_ids),
        "no duplicate chunk ids",
        f"{len(chroma_ids) - len(set(chroma_ids))} duplicate id(s)",
    )

    # Confirm the stored vectors are real and correctly shaped, rather than
    # trusting that a write which reported success produced usable embeddings.
    if chroma_ids:
        from backend.app.database.chroma import fetch_text_chunks_by_id

        sample_ids = chroma_ids[:5]
        sample = fetch_text_chunks_by_id(sample_ids, include_embeddings=True)
        embs = sample.get("embeddings")
        if embs is None:
            embs = [4]
        expected_dim = 384  # all-MiniLM-L6-v2 / nomic-embed-text default
        try:
            sample_vec = await embedder.embed_query_async("dim-probe")
            expected_dim = len(sample_vec)
        except Exception:
            pass
        shapes_ok = bool(embs) and all(
            e is not None and len(e) == expected_dim for e in embs
        )
        c.check(
            shapes_ok,
            f"sampled embeddings are {expected_dim}-dim and non-null",
            f"{len(embs)} sampled",
        )

    # Duplicate chunk bodies waste the context window and crowd out distinct
    # material, since the retriever dedupes only within a single result set.
    normalised = [d.strip() for d in docs if d and d.strip()]
    unique_n = len(set(normalised))
    dup_n = len(normalised) - unique_n
    c.check(dup_n == 0, "no duplicate chunk bodies", f"{dup_n} duplicate(s)")
    c.check(
        len(normalised) == len(docs),
        "no empty chunk bodies",
        f"{len(docs) - len(normalised)} empty",
    )

    print("\n=== 4. Retrieval smoke tests ===", flush=True)
    from backend.app.rag.lexical import get_lexical_index
    from backend.app.rag.reranker import get_reranker

    get_lexical_index().rebuild()
    get_reranker().warmup()

    from backend.app.rag.retriever import get_retriever

    retriever = get_retriever()

    async with async_session_factory() as db:
        for query, expected in SMOKE_QUERIES:
            chunks, _images, _processed = await retriever.retrieve(query, db=db)
            blob = " ".join(ch.text for ch in chunks).lower()
            hit = expected.lower() in blob
            c.check(
                bool(chunks) and hit,
                f"{query!r}",
                f"{len(chunks)} chunk(s), expected {expected!r} "
                f"{'found' if hit else 'MISSING'}",
            )

        print("\n=== 5. Negative control (must retrieve nothing) ===", flush=True)
        chunks, _, _ = await retriever.retrieve(NEGATIVE_QUERY, db=db)
        c.check(
            not chunks,
            f"off-topic query returns no context: {NEGATIVE_QUERY!r}",
            f"got {len(chunks)} chunk(s) — cross-encoder gate "
            f"(rerank_min_score) may be too loose"
            if chunks
            else "correctly empty",
        )

    print("\n" + "=" * 70, flush=True)
    if c.failures:
        print(f"VERIFICATION FAILED — {len(c.failures)} check(s) failed:", flush=True)
        for f in c.failures:
            print(f"  - {f}", flush=True)
        return 1
    print("VERIFICATION PASSED — knowledge base is populated and retrievable.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
