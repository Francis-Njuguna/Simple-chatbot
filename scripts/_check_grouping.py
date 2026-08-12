"""Check adjacent-chunk grouping: merging, ordering, and non-adjacent safety."""

import sys

sys.path.insert(0, ".")

from backend.app.rag.retriever import RetrievedChunk, get_retriever  # noqa: E402


def mk(cid, article, index, text, score=0.5):
    return RetrievedChunk(
        chunk_id=cid, text=text, article_id=article, chunk_index=index,
        score=score, category="LMS",
    )


def show(label, out):
    print(f"\n{label}")
    for c in out:
        preview = c.text.replace("\n", " | ")[:90]
        print(f"   art={c.article_id} idx={c.chunk_index} {preview!r}")


r = get_retriever()
g = r._group_adjacent

# 1. Adjacent chunks of one article merge; retrieval order is preserved.
out = g([
    mk("1_c1", "1", 1, "step two"),
    mk("4_c0", "4", 0, "other article"),
    mk("1_c2", "1", 2, "step three"),
    mk("1_c0", "1", 0, "step one"),
])
show("1. adjacent merge (expect art=1 idx=0 merged 3-in-1, THEN art=4)", out)
assert len(out) == 2, out
assert out[0].article_id == "1" and out[0].text == "step one\n\nstep two\n\nstep three"
assert out[1].article_id == "4"

# 2. Non-adjacent chunks must NOT merge — content is missing between them.
out = g([mk("1_c0", "1", 0, "intro"), mk("1_c4", "1", 4, "conclusion")])
show("2. non-adjacent (expect 2 separate blocks)", out)
assert len(out) == 2, out

# 3. Two separate runs in one article stay separate.
out = g([
    mk("1_c0", "1", 0, "a"), mk("1_c1", "1", 1, "b"),
    mk("1_c5", "1", 5, "y"), mk("1_c6", "1", 6, "z"),
])
show("3. two runs (expect 'a|b' and 'y|z')", out)
assert len(out) == 2 and out[0].text == "a\n\nb" and out[1].text == "y\n\nz"

# 4. Merged block inherits the BEST-ranked member's position and scores.
out = g([
    mk("1_c9", "1", 9, "low rank first", score=0.9),
    mk("1_c8", "1", 8, "second", score=0.3),
])
show("4. score/position from best-ranked member", out)
assert out[0].score == 0.9, out[0].score

# 5. Real corpus: does merging duplicate the [Title] header?
print("\n5. real chunks — checking for duplicated [Title] headers")
from backend.app.rag.lexical import get_lexical_index  # noqa: E402

idx = get_lexical_index()
idx.ensure_loaded()
real = [
    mk(cid, cid.split("_")[0], int(cid.split("_")[-1]), doc)
    for cid, doc in zip(idx._corpus_ids, idx._corpus_docs)
    if cid.startswith("2_chunk_")
]
out = g(real)
merged_text = out[0].text
n_headers = merged_text.count("[Knowledgebase")
print(f"   merged {len(real)} chunks -> {len(out)} block(s), "
      f"[Title] header appears {n_headers}x")
print(f"   first 300 chars: {merged_text[:300]!r}")

print("\nAll grouping assertions passed.")
