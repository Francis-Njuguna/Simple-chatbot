"""Source citation deduplication — one citation per article.

Retrieval works in chunks; several chunks of the same article routinely survive
reranking. Before dedup, that rendered the same title and URL once per chunk in
the widget's "Sources & References" list. The citation list should name
*documents*, not excerpts.
"""

from __future__ import annotations

from backend.app.rag.retriever import RetrievedChunk
from backend.app.services.rag_service import RAGService


_seq = 0


def _chunk(article_id: str, title: str, url: str, score: float) -> RetrievedChunk:
    global _seq
    _seq += 1
    return RetrievedChunk(
        chunk_id=f"c{_seq}",
        text="x",
        article_id=article_id,
        category="IT Support",
        chunk_index=0,
        score=score,
        title=title,
        url=url,
    )


def test_one_citation_per_article() -> None:
    chunks = [
        _chunk("a1", "MFA Setup", "https://kb/a1", 0.81),
        _chunk("a1", "MFA Setup", "https://kb/a1", 0.74),
        _chunk("a1", "MFA Setup", "https://kb/a1", 0.55),
        _chunk("a2", "Moodle Login", "https://kb/a2", 0.63),
    ]

    sources = RAGService._build_sources(chunks)

    assert [s.article_id for s in sources] == ["a1", "a2"]


def test_keeps_the_first_chunk_of_each_article() -> None:
    """The kept chunk is the first one, not the numerically highest ``score``.

    Chunks arrive in the order retrieval and reranking put them in, and that
    ordering is authoritative — after a cross-encoder pass the leading chunk is
    the most relevant one even though ``score`` still holds cosine similarity,
    which the reranker deliberately does not overwrite. Picking ``max(score)``
    here would quietly re-rank inside the citation list, so it is not done. The
    scores below are out of order to pin exactly that.
    """
    chunks = [
        _chunk("a1", "MFA Setup", "https://kb/a1", 0.55),
        _chunk("a1", "MFA Setup", "https://kb/a1", 0.81),
        _chunk("a1", "MFA Setup", "https://kb/a1", 0.74),
    ]

    sources = RAGService._build_sources(chunks)

    assert len(sources) == 1
    assert sources[0].score == 0.55


def test_keeps_article_order() -> None:
    chunks = [
        _chunk("a3", "Email", "https://kb/a3", 0.5),
        _chunk("a1", "MFA Setup", "https://kb/a1", 0.9),
        _chunk("a2", "Moodle", "https://kb/a2", 0.7),
    ]

    assert [s.article_id for s in RAGService._build_sources(chunks)] == ["a3", "a1", "a2"]


def test_empty_input() -> None:
    assert RAGService._build_sources([]) == []
