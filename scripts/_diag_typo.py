"""Why does 'moddle login' return nothing when 'LMS login' returns 5 chunks?

Compares the rerank scores a typo'd query and its corrected form get for the
SAME shortlist, which isolates whether the loss is in candidate generation or
in the cross-encoder gate.
"""

import sys

sys.path.insert(0, ".")

import anyio  # noqa: E402


async def main():
    from backend.app.rag.retriever import get_retriever
    from backend.app.rag.reranker import get_reranker
    from backend.app.rag.query_processing import process_query
    from backend.app.rag.lexical import get_lexical_index
    from backend.app.config import get_settings

    settings = get_settings()
    r = get_retriever()
    vocab = get_lexical_index().vocabulary()

    for q in ("moddle login", "Moodle login", "LMS login"):
        p = process_query(q, fuzzy_vocabulary=vocab)
        print(f"\n{'=' * 70}\n{q!r}\n{'=' * 70}")
        print(f"  original   : {p.original!r}")
        print(f"  normalized : {p.normalized!r}")
        print(f"  corrections: {p.corrections}")
        print(f"  lexical    : {p.lexical[:100]!r}")

        chunks = await r.retrieve_text(q)
        print(f"  -> retrieve_text returned {len(chunks)} chunks")

    # Same passages, three different query texts through the cross-encoder.
    print(f"\n{'=' * 70}\nCROSS-ENCODER: same passages, different query text\n{'=' * 70}")
    print(f"gate = rerank_min_score = {settings.rerank_min_score}")

    emb = await r.embedding_service.embed_query_async("Moodle login")
    from backend.app.database.chroma import query_text_collection

    res = query_text_collection(query_embedding=emb, n_results=6)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    rr = get_reranker()
    for qtext in ("moddle login", "Moodle login", "LMS login",
                  "LMS login Learning Management System Moodle e-learning sign in logon"):
        scores = rr.score(qtext, docs)
        passed = sum(1 for s in scores if s >= settings.rerank_min_score)
        print(f"\n  query={qtext[:55]!r}")
        print(f"    scores : {[round(float(s), 2) for s in scores]}")
        print(f"    passing gate: {passed}/{len(scores)}")
        for m, s in list(zip(metas, scores))[:3]:
            print(f"      {(m.get('title') or '?')[:44]:44s} {float(s):7.2f}")


anyio.run(main)
