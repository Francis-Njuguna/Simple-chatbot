"""Throwaway: verify retrieval returns text+images and the prompt carries images."""

import asyncio
import time

from backend.app.rag.llm import get_llm_service
from backend.app.rag.retriever import get_retriever

QUERIES = [
    "I cannot log in to the LMS, what should I do?",
    "How do I set up Microsoft Authenticator?",
    "How do I install SMOWL for my exam?",
]


async def main() -> None:
    r = get_retriever()
    llm = get_llm_service()

    # Warm the model + collections so timings reflect steady state, not cold start.
    t0 = time.perf_counter()
    await r.embed_query("warmup")
    r.format_context([])
    print(f"[warmup] {time.perf_counter() - t0:.2f}s\n")

    for q in QUERIES:
        t0 = time.perf_counter()
        qe = await r.embed_query(q)
        t_embed = time.perf_counter() - t0

        t0 = time.perf_counter()
        chunks, images = await r.retrieve(q, query_embedding=qe)
        t_ret = time.perf_counter() - t0

        ctx = r.format_context(chunks)
        img_ctx = r.format_images(images)

        print(f"=== {q}")
        print(f"    embed={t_embed*1000:.0f}ms retrieve={t_ret*1000:.0f}ms "
              f"chunks={len(chunks)} images={len(images)}")
        print(f"    titles: {[c.title for c in chunks][:4]}")
        print(f"    ctx_chars={len(ctx)}")
        print("    --- image block ---")
        for line in img_ctx.splitlines():
            print(f"    {line}")

        # Confirm the images actually land in the rendered prompt.
        msgs = llm._build_messages(q, ctx, "No prior conversation.", img_ctx)
        human = msgs[1].content
        print(f"    images_in_prompt={img_ctx.splitlines()[0][:40]!r} present="
              f"{img_ctx.splitlines()[0] in human}")
        print(f"    prompt_chars={len(human)}\n")


asyncio.run(main())
