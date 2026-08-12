"""Smoke-check the multi-query retrieval path against the real KB.

Each query declares the article(s) that genuinely answer it, and the check
reports where the best of those landed in the ranking. That is the actual
success criterion — "a user should get the same answer regardless of wording"
means every phrasing reaches the *right article*, which is not the same as
every phrasing agreeing with the others (two wordings can agree on a wrong
article, and two correct answers can legitimately differ when the KB splits a
topic across articles).

Also runs off-topic queries, which must return nothing.

Needs Chroma, not Postgres: titles are read from the [Title] header the chunker
prepends to every chunk body, so no hydration step is involved.
"""

import re
import sys
import time

sys.path.insert(0, ".")

import anyio  # noqa: E402

from backend.app.rag.retriever import get_retriever  # noqa: E402

_TITLE_RE = re.compile(r"^\s*\[([^\]]{1,200})\]")

# query -> article_ids that genuinely answer it. Any one of them counts as a
# hit; several are listed where the KB really does split a topic (e.g. MFA is
# covered by both "Set up your Microsoft 365 sign-in" and "Setting up Microsoft
# Authenticator").
GROUPS: dict[str, list[tuple[str, set[str]]]] = {
    "LMS/Moodle": [
        ("How do I log into Moodle?", {"1"}),
        ("LMS login", {"1"}),
        ("Can't access LMS", {"1"}),
        ("moddle login", {"1"}),
        ("Learning Management System login", {"1"}),
    ],
    "Email": [
        ("Student email", {"4"}),
        ("Outlook login", {"4"}),
        ("Corporate email", {"4"}),
        ("University email", {"4"}),
    ],
    "MFA": [
        ("Authenticator setup", {"10", "11"}),
        ("MFA", {"10", "11"}),
        ("2FA", {"10", "11"}),
        ("Microsoft Authenticator", {"10", "11"}),
    ],
    "VAS": [
        ("VAS exam", {"14", "15", "16", "17", "22"}),
        ("Assessment system", {"14", "15", "16", "17", "22"}),
        ("Online exam", {"14", "15", "16", "17", "22"}),
    ],
    "SMOWL": [
        ("SMOWL camera", {"14", "15", "18"}),
        ("Proctoring software", {"14", "15", "18"}),
        ("Exam monitoring", {"14", "15", "18"}),
    ],
    # Not a "same answer" group — four different topics that happen to be
    # abbreviated. Each is checked against its own article.
    "Abbreviations": [
        ("portal pwd", {"9", "3"}),
        ("mfa", {"10", "11"}),
        ("vas exam", {"14", "15", "16", "17", "22"}),
        ("smowl cam", {"14", "15", "18"}),
    ],
    "NL variations": [
        ("I forgot my password", {"9"}),
        ("can't login", {"1", "9", "3"}),
        ("unable to access", {"1", "9", "3"}),
        ("where do I sign in", {"1", "9", "3"}),
        ("login problem", {"1", "9", "3"}),
    ],
}

OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the world cup in 2018?",
    "What is the weather tomorrow?",
    "Explain quantum entanglement",
    "Best pizza recipe",
]


def _title(chunk) -> str:
    """Article title, read from the [Title] header the chunker prepends.

    Chroma does not carry titles (PostgreSQL owns them) and this check runs
    without a DB session, so the header is the only source available here.
    """
    match = _TITLE_RE.match(chunk.text or "")
    if match:
        return match.group(1)
    meta = chunk._raw_meta or {}
    return meta.get("article_id") or chunk.chunk_id


def _article(chunk) -> str:
    return (chunk._raw_meta or {}).get("article_id") or "?"


async def main() -> int:
    retriever = get_retriever()
    latencies: list[float] = []
    failures: list[str] = []
    hits_at_1 = 0
    hits_at_k = 0
    total = 0

    # Warm both models first. A cold sentence-transformers load is ~35s and a
    # cold cross-encoder ~14s; including either in the timings would report a
    # latency the production process (which loads once at startup) never sees.
    from backend.app.rag.reranker import get_reranker

    await retriever.embedding_service.embed_query_async("warmup")
    get_reranker().score("warmup", ["warmup passage"])
    print("(models warm — timings below are steady-state)\n")

    for name, cases in GROUPS.items():
        print("=" * 78)
        print(f"GROUP: {name}")
        print("=" * 78)
        for query, expected in cases:
            total += 1
            start = time.perf_counter()
            chunks = await retriever.retrieve_text(query)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            articles = [_article(c) for c in chunks]
            rank = next(
                (i for i, a in enumerate(articles) if a in expected), None
            )
            if rank == 0:
                hits_at_1 += 1
                hits_at_k += 1
                verdict = "HIT@1"
            elif rank is not None:
                hits_at_k += 1
                verdict = f"hit@{rank + 1}"
            else:
                verdict = "MISS"
                failures.append(
                    f"{name}: {query!r} expected article(s) {sorted(expected)}, "
                    f"got {articles or ['<NOTHING>']}"
                )

            print(f"\n  {query!r}  ({elapsed:.2f}s, {len(chunks)} chunks) -> {verdict}")
            for c in chunks[:3]:
                mark = "*" if _article(c) in expected else " "
                print(
                    f"    {mark} [{_article(c):>3s}] {_title(c)[:42]:42s} "
                    f"cos={c.score:.3f} "
                    f"rerank={c.rerank_score if c.rerank_score is None else round(c.rerank_score, 2)}"
                )
        print()

    print("=" * 78)
    print("OFF-TOPIC (must return 0 chunks)")
    print("=" * 78)
    off_ok = 0
    for q in OFF_TOPIC:
        chunks = await retriever.retrieve_text(q)
        if not chunks:
            off_ok += 1
            print(f"  {q!r} -> OK (declined)")
        else:
            print(f"  {q!r} -> LEAKED {len(chunks)}")
            for c in chunks[:3]:
                print(f"      [{_article(c):>3s}] {_title(c)[:42]:42s} rerank={c.rerank_score}")
            failures.append(f"off-topic: {q!r} returned {len(chunks)} chunks")

    hits_any = hits_at_k
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  recall@1   : {hits_at_1}/{total} ({100 * hits_at_1 / total:.1f}%)")
    print(f"  recall@k   : {hits_any}/{total} ({100 * hits_any / total:.1f}%)")
    print(
        f"  off-topic  : {off_ok}/{len(OFF_TOPIC)} declined "
        f"({100 * off_ok / len(OFF_TOPIC):.1f}% precision)"
    )
    if latencies:
        latencies.sort()
        p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
        print(
            f"  latency    : mean={sum(latencies) / len(latencies):.2f}s "
            f"p95={p95:.2f}s max={latencies[-1]:.2f}s  (target < 3s)"
        )

    if failures:
        print(f"\n!! {len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll queries reached an expected article; no off-topic leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(anyio.run(main))
