"""Measure what the final chunk set actually looks like: size, and how much
text is duplicated between the chunks handed to the LLM.

Two things the TODO asks about:
  * "keep only the best 4-6 chunks" — is the kept count landing in that band?
  * "merge duplicates" — the current dedup is exact-text only. The chunker
    overlaps consecutive chunks, so two chunks of one article can share a large
    span of text without being byte-identical. This reports that overlap.
"""

import sys

sys.path.insert(0, ".")

import anyio  # noqa: E402

from backend.app.rag.retriever import get_retriever  # noqa: E402

QUERIES = [
    "How do I log into Moodle?", "LMS login", "moddle login",
    "Student email", "Outlook login",
    "Authenticator setup", "MFA", "2FA",
    "VAS exam", "Online exam",
    "SMOWL camera", "Proctoring software",
    "portal pwd", "I forgot my password", "can't login",
    "unable to access", "where do I sign in", "login problem",
]


def shingles(text: str, n: int = 8) -> set[str]:
    """Word n-grams — near-duplicate detection that ignores small edits."""
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


async def main() -> int:
    r = get_retriever()
    from backend.app.rag.reranker import get_reranker

    await r.embedding_service.embed_query_async("warmup")
    get_reranker().score("warmup", ["warmup passage"])
    print("(warm)\n")

    counts: list[int] = []
    grouped_counts: list[int] = []
    overlaps: list[tuple[str, str, str, float]] = []

    for q in QUERIES:
        chunks = await r.retrieve_text(q)
        counts.append(len(chunks))
        grouped = r._group_adjacent(chunks)
        grouped_counts.append(len(grouped))

        arts = [f"{c.article_id}:{c.chunk_index}" for c in chunks]
        print(f"{q!r:32s} -> {len(chunks)} kept, {len(grouped)} after grouping  {arts}")

        # Pairwise near-duplicate check on what the LLM would actually receive.
        for i in range(len(grouped)):
            for j in range(i + 1, len(grouped)):
                a, b = shingles(grouped[i].text), shingles(grouped[j].text)
                if not a or not b:
                    continue
                jac = len(a & b) / min(len(a), len(b))
                if jac > 0.15:
                    overlaps.append(
                        (q, grouped[i].chunk_id, grouped[j].chunk_id, jac)
                    )

    print("\n" + "=" * 70)
    print(f"kept chunks     : min={min(counts)} max={max(counts)} "
          f"mean={sum(counts) / len(counts):.1f}  (target band 4-6)")
    print(f"after grouping  : min={min(grouped_counts)} max={max(grouped_counts)} "
          f"mean={sum(grouped_counts) / len(grouped_counts):.1f}")
    below = [c for c in counts if c < 4]
    print(f"queries under 4 : {len(below)}/{len(counts)}")
    print(f"\nnear-duplicate pairs (>15% shingle overlap): {len(overlaps)}")
    for q, a, b, jac in overlaps[:15]:
        print(f"   {jac:.0%}  {a} <-> {b}   ({q!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(anyio.run(main))
