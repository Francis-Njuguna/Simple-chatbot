"""Does MMR belong before the cross-encoder?

MMR maximises diversity; the cross-encoder judges relevance. Running MMR first
means candidates are discarded for resembling an already-picked chunk *before*
anything has assessed whether they answer the question. Measured symptom on
"Learning Management System sign in": 1_chunk_0 — the best chunk in the KB for
that query — never reaches the shortlist, because 1_chunk_1 was picked first and
MMR scored 1_chunk_0 as redundant with it.

This compares three shortlisting strategies feeding the same reranker:
  mmr   — current behaviour
  rrf   — top-N by fused RRF score (pure relevance, no diversity objective)
  cos   — top-N by cosine, as a control

TODO #8 in the brief prescribes "larger candidate pool -> rerank -> keep the
best 4-6", with no diversity stage in between, so `rrf` is also the closer
reading of the requirement.
"""

import sys

sys.path.insert(0, ".")

import anyio  # noqa: E402
import numpy as np  # noqa: E402

from backend.app.config import get_settings  # noqa: E402
from backend.app.rag import retriever as R  # noqa: E402

CASES = [
    ("Learning Management System sign in", {"1"}),
    ("How do I log into Moodle?", {"1"}),
    ("LMS login", {"1"}),
    ("moddle login", {"1"}),
    ("e-learning platform login", {"1"}),
    ("MFA", {"10", "11"}),
    ("2FA setup", {"10", "11"}),
    ("verification app enrollment", {"10", "11"}),
    ("Student email", {"4"}),
    ("Outlook login", {"4"}),
    ("corporate email access", {"4"}),
    ("VAS exam", {"14", "15", "16", "17", "22"}),
    ("SMOWL camera", {"14", "15", "18"}),
    ("webcam exam supervision", {"14", "15", "21"}),
    ("I forgot my password", {"9", "12", "3"}),
    ("SIS password reset", {"9", "12", "3"}),
    ("can't login", {"1", "9", "3"}),
    ("where do I sign in", {"1", "9", "3"}),
    ("How do I use Microsoft Teams?", {"5"}),
    ("What is My Loft?", {"6"}),
]

OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the world cup in 2018?",
    "What is the weather tomorrow?",
    "Explain quantum entanglement",
    "Best pizza recipe",
]

_real_mmr = R.HybridRetriever._mmr_select_vectorised


def patch(strategy: str) -> None:
    """Swap the shortlist selector. Signature matches the real one."""
    if strategy == "mmr":
        R.HybridRetriever._mmr_select_vectorised = _real_mmr
        return

    @staticmethod
    def selector(query_embedding, candidate_embeddings, k, lambda_param):
        # Candidates arrive in fused-RRF order, so "first k" IS top-k by RRF.
        if strategy == "rrf":
            return list(range(min(k, len(candidate_embeddings))))
        sims = candidate_embeddings @ np.asarray(query_embedding, dtype=np.float32)
        return sorted(range(len(sims)), key=lambda i: -sims[i])[:k]

    R.HybridRetriever._mmr_select_vectorised = selector


async def main() -> int:
    r = R.get_retriever()
    settings = get_settings()
    from backend.app.rag.reranker import get_reranker

    await r.embedding_service.embed_query_async("warmup")
    get_reranker().score("warmup", ["warmup passage"])
    print(f"(warm; mmr_diversity={settings.mmr_diversity} "
          f"shortlist={settings.mmr_shortlist} top_k={settings.top_k_retrieval})\n")

    for strategy in ("mmr", "rrf", "cos"):
        patch(strategy)
        r.clear_cache()
        h1 = hk = 0
        misses = []
        for q, expected in CASES:
            chunks = await r.retrieve_text(q)
            arts = [(c._raw_meta or {}).get("article_id") for c in chunks]
            rank = next((i for i, a in enumerate(arts) if a in expected), None)
            if rank == 0:
                h1 += 1
            if rank is not None:
                hk += 1
            else:
                misses.append(q)

        leaked = []
        for q in OFF_TOPIC:
            r.clear_cache()
            if await r.retrieve_text(q):
                leaked.append(q)

        n = len(CASES)
        print(f"{strategy:4s}  recall@1={h1}/{n} ({100 * h1 / n:.1f}%)  "
              f"recall@k={hk}/{n} ({100 * hk / n:.1f}%)  "
              f"offtopic_leaked={len(leaked)}/{len(OFF_TOPIC)}")
        if misses:
            print(f"      misses: {misses}")
        if leaked:
            print(f"      leaked: {leaked}")
        print()

    R.HybridRetriever._mmr_select_vectorised = _real_mmr
    return 0


if __name__ == "__main__":
    raise SystemExit(anyio.run(main))
