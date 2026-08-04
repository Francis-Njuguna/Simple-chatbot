"""Sweep MMR diversity against the yield of co-relevant sibling chunks.

The problem this measures: MMR penalises a candidate for resembling what it has
already picked, and chunks of the SAME article resemble each other by
construction. So for a question answered by one multi-step article, MMR evicts
the very siblings that complete the procedure and backfills the shortlist with
unrelated articles the cross-encoder gate then discards — leaving 1 chunk where
the fused ranking had 5 good ones.

Reported per setting: chunks kept, and how many of RRF's top-5 survived into the
cross-encoder shortlist (the number MMR is throwing away).

Off-topic queries are included because widening what reaches the shortlist must
not widen what escapes the gate.
"""

import sys

sys.path.insert(0, ".")

import anyio  # noqa: E402

from backend.app.config import get_settings  # noqa: E402
from backend.app.rag.retriever import get_retriever  # noqa: E402

ON_TOPIC = [
    ("MFA", {"10", "11"}),
    ("2FA", {"10", "11"}),
    ("Authenticator setup", {"10", "11"}),
    ("Microsoft Authenticator", {"10", "11"}),
    ("How do I log into Moodle?", {"1"}),
    ("LMS login", {"1"}),
    ("moddle login", {"1"}),
    ("Student email", {"4"}),
    ("Outlook login", {"4"}),
    ("VAS exam", {"14", "15", "16", "17", "22"}),
    ("SMOWL camera", {"14", "15", "18"}),
    ("portal pwd", {"9", "3"}),
    ("I forgot my password", {"9"}),
    ("can't login", {"1", "9", "3"}),
    ("login problem", {"1", "9", "3"}),
]

OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the world cup in 2018?",
    "What is the weather tomorrow?",
    "Explain quantum entanglement",
    "Best pizza recipe",
]

# (mmr_diversity, mmr_shortlist). lambda is the RELEVANCE weight:
#   mmr = lambda * relevance - (1 - lambda) * max_similarity_to_selected
# so 0.3 means diversity outweighs relevance more than 2:1. 1.0 is MMR off.
# 0.3 is what .env currently sets; the config default says 0.7.
SETTINGS = [(0.3, 16), (0.5, 16), (0.7, 16), (0.85, 16), (1.0, 16)]


async def main() -> int:
    r = get_retriever()
    settings = get_settings()
    from backend.app.rag.reranker import get_reranker

    await r.embedding_service.embed_query_async("warmup")
    get_reranker().score("warmup", ["warmup passage"])
    print("(warm)\n")

    for diversity, shortlist in SETTINGS:
        object.__setattr__(settings, "mmr_diversity", diversity)
        object.__setattr__(settings, "mmr_shortlist", shortlist)

        counts: list[int] = []
        hits1 = 0
        hitsk = 0
        misses: list[str] = []
        thin: list[str] = []
        for q, expected in ON_TOPIC:
            chunks = await r.retrieve_text(q)
            counts.append(len(chunks))
            if len(chunks) < 4:
                thin.append(f"{q}={len(chunks)}")
            arts = [(c._raw_meta or {}).get("article_id") for c in chunks]
            rank = next((i for i, a in enumerate(arts) if a in expected), None)
            if rank == 0:
                hits1 += 1
            if rank is not None:
                hitsk += 1
            else:
                misses.append(q)

        leaked = []
        for q in OFF_TOPIC:
            if await r.retrieve_text(q):
                leaked.append(q)

        n = len(ON_TOPIC)
        print(f"lambda={diversity:.2f} shortlist={shortlist}")
        print(f"   kept   : min={min(counts)} mean={sum(counts) / len(counts):.1f} "
              f"under4={sum(1 for c in counts if c < 4)}/{n}")
        print(f"   recall : @1={hits1}/{n} @k={hitsk}/{n}")
        print(f"   offtopic leaked: {len(leaked)}/{len(OFF_TOPIC)} {leaked}")
        if thin:
            print(f"   thin (<4): {thin}")
        if misses:
            print(f"   misses : {misses}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(anyio.run(main))
