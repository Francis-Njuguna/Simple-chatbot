"""Should the cross-encoder decide the final ORDER, or only the gate?

Evidence that these are different jobs. On "Can't access LMS" the cross-encoder
scores 5_chunk_0 (Microsoft Teams) +1.27 and 1_chunk_0 ("How to login to LMS",
the right answer, and RRF's #1 at 0.0356) only -1.43 — so with top_k=5 the
correct chunk is pushed off the end. Yet the same model gates off-topic queries
perfectly (6/6 declined). It is reliable at "does this passage answer the
question at all" and noisy at "which of these two on-topic passages is better".

Strategies compared (all keep the cross-encoder gate unchanged):
  rerank — order by cross-encoder score (current behaviour)
  rrf    — gate by cross-encoder, order by fused RRF score
  blend  — order by normalised rerank + normalised RRF, weighted

Reports recall@1 and recall@k on paraphrase/synonym/typo queries plus off-topic
leakage, so a gain in ordering cannot hide a loss in precision.
"""

import sys

sys.path.insert(0, ".")

import anyio  # noqa: E402

from backend.app.rag import retriever as R  # noqa: E402

CASES = [
    ("Can't access LMS", {"1"}),
    ("Learning Management System sign in", {"1"}),
    ("How do I log into Moodle?", {"1"}),
    ("LMS login", {"1"}),
    ("moddle login", {"1"}),
    ("e-learning platform login", {"1"}),
    ("How do I log in to the LMS?", {"1"}),
    ("MFA", {"10", "11"}),
    ("2FA setup", {"10", "11"}),
    ("verification app enrollment", {"10", "11"}),
    ("How do I set up Microsoft Authenticator?", {"11", "10"}),
    ("Student email", {"4"}),
    ("Outlook login", {"4"}),
    ("corporate email access", {"4"}),
    ("How do I access my student email?", {"4"}),
    ("VAS exam", {"14", "15", "16", "17", "22"}),
    ("SMOWL camera", {"14", "15", "18"}),
    ("webcam exam supervision", {"14", "15", "21"}),
    ("What is SMOWL proctoring?", {"14", "15", "18", "21"}),
    ("I forgot my password", {"9", "12", "3"}),
    ("SIS password reset", {"9", "12", "3"}),
    ("How do I reset my student portal password?", {"9", "12", "3"}),
    ("can't login", {"1", "9", "3"}),
    ("where do I sign in", {"1", "9", "3"}),
    ("login problem", {"1", "9", "3"}),
    ("How do I use Microsoft Teams?", {"5"}),
    ("What is My Loft?", {"6"}),
    ("How do I contact the AmIU help desk?", {"2"}),
    ("How do lecturers mark exams?", {"20"}),
    ("How do I register for supplementary exams?", {"13"}),
]

OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the world cup in 2018?",
    "What is the weather tomorrow?",
    "Explain quantum entanglement",
    "Best pizza recipe",
    "How do I submit an assignment in Moodle?",
]

STRATEGIES = [
    ("rerank", 1.0),
    ("blend-0.75", 0.75),
    ("blend-0.5", 0.5),
    ("blend-0.25", 0.25),
    ("rrf", 0.0),
]


async def run(weight: float) -> tuple[int, int, int, list[str]]:
    """weight = share given to the cross-encoder in the ordering (1.0 = current)."""
    R._ORDER_RERANK_WEIGHT = weight
    r = R.get_retriever()
    r.clear_cache()

    h1 = hk = 0
    misses: list[str] = []
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

    leaked = 0
    for q in OFF_TOPIC:
        r.clear_cache()
        if await r.retrieve_text(q):
            leaked += 1
    return h1, hk, leaked, misses


async def main() -> int:
    from backend.app.rag.reranker import get_reranker

    r = R.get_retriever()
    await r.embedding_service.embed_query_async("warmup")
    get_reranker().score("warmup", ["warmup passage"])
    print(f"(warm; {len(CASES)} on-topic, {len(OFF_TOPIC)} off-topic)\n")

    n = len(CASES)
    for name, weight in STRATEGIES:
        h1, hk, leaked, misses = await run(weight)
        print(f"{name:<11} recall@1={h1:>2}/{n} ({100 * h1 / n:5.1f}%)  "
              f"recall@k={hk:>2}/{n} ({100 * hk / n:5.1f}%)  "
              f"offtopic_leaked={leaked}/{len(OFF_TOPIC)}")
        if misses:
            print(f"            misses: {misses}")
    R._ORDER_RERANK_WEIGHT = None
    return 0


if __name__ == "__main__":
    raise SystemExit(anyio.run(main))
