"""Time each retrieval stage separately to locate a hang."""

import sys
import time

sys.path.insert(0, ".")

import anyio  # noqa: E402


def stamp(label, start):
    print(f"  {label}: {time.perf_counter() - start:.2f}s", flush=True)


async def main():
    t = time.perf_counter()
    from backend.app.rag.retriever import get_retriever
    from backend.app.rag.lexical import get_lexical_index
    from backend.app.rag.query_processing import process_query

    stamp("imports", t)

    t = time.perf_counter()
    idx = get_lexical_index()
    idx.ensure_loaded()
    stamp("lexical index build", t)

    t = time.perf_counter()
    vocab = idx.vocabulary()
    stamp(f"vocabulary ({len(vocab)} terms)", t)

    t = time.perf_counter()
    p = process_query("LMS login", fuzzy_vocabulary=vocab)
    stamp("process_query", t)
    print(f"    normalized={p.normalized!r}", flush=True)
    print(f"    lexical={p.lexical[:120]!r}", flush=True)
    print(f"    variants={p.variants}", flush=True)

    r = get_retriever()

    t = time.perf_counter()
    emb = await r.embedding_service.embed_query_async("LMS login")
    stamp(f"embed 1 query (dim {len(emb)})", t)

    t = time.perf_counter()
    embs = await r.embedding_service.embed_texts_async(p.variants)
    stamp(f"embed {len(p.variants)} variants", t)

    t = time.perf_counter()
    res = await r._vector_search_variants(p.variants, None, 40, primary_embedding=emb)
    stamp(f"_vector_search_variants ({len(res)} result sets)", t)

    t = time.perf_counter()
    hits = idx.search(p.lexical, k=40, fuzzy=True)
    stamp(f"bm25 search ({len(hits)} hits)", t)

    t = time.perf_counter()
    chunks = await r.retrieve_text("LMS login")
    stamp(f"full retrieve_text ({len(chunks)} chunks)", t)


anyio.run(main)
