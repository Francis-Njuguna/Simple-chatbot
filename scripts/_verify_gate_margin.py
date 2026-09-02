"""Does the int8 score drift come near flipping the rerank_min_score gate?

`_verify_score_multi.py` establishes that int8 perturbs cross-encoder logits by
up to ~0.25 while fp32 is identical to 1e-6, and `bench_quality.py` shows no
recall or precision change on the 56-query eval set. Neither answers the
question that actually matters:

    the gate at rerank_min_score=-8.0 is an ABSOLUTE comparison, so the risk is
    not "did any verdict change on these queries" but "how much headroom is
    there before one does".

A no-regression pass with 0.05 of margin is luck; the same pass with 3.0 of
margin is a property. This script measures the margin directly.

For every eval query it scores the real shortlist twice — fp32 and int8, same
process, same passages — and reports, per query:

  * the fp32 and int8 max logit (the chunk that decides answer vs decline)
  * which side of the gate each lands on
  * the distance from the gate, i.e. how much drift it would take to flip

The verdict is the SMALLEST margin across the whole set, compared against the
observed worst-case int8 drift. A margin many times the drift means the
quantization cannot reach the gate on this KB; a margin near it means int8 is
one rephrasing away from changing a student-visible decision.

Run:  ./.venv/Scripts/python.exe -u scripts/_verify_gate_margin.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.rag.reranker import CrossEncoderReranker  # noqa: E402
from scripts.eval_set import EVAL_QUERIES  # noqa: E402


def _load_kb_chunks() -> list[str]:
    """Chunk the real KB the way ingestion does, so passages are production-shaped."""
    import json

    from backend.app.ingest.chunker import TextChunker

    chunker = TextChunker()
    raw_dir = Path("data/raw")
    texts: list[str] = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            article = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(article, dict):
            continue
        body = article.get("text") or ""
        if not body:
            continue
        chunks = chunker.chunk_article(
            article_id=str(article.get("article_id", "")),
            title=str(article.get("title", "")),
            category=article.get("category"),
            url=str(article.get("url", "")),
            text=str(body),
        )
        for chunk in chunks:
            text = chunk.get("text") if isinstance(chunk, dict) else chunk
            if text:
                texts.append(str(text))
    return texts


def _build(quantize: bool) -> CrossEncoderReranker:
    """A reranker with its own settings, so both precisions coexist in one process.

    Goes through ``__init__`` (which installs the load lock) and then overrides
    ``settings``: constructing via ``__new__`` skips the lock and dies inside
    ``_ensure_model``.
    """
    import threading

    reranker = CrossEncoderReranker()
    reranker.settings = Settings(
        _env_file=None, RERANK_QUANTIZE=quantize, RERANK_ENABLED=True
    )
    reranker._model = None
    reranker._load_failed = False
    reranker._lock = threading.Lock()
    reranker.score("warmup", ["warmup passage"])
    return reranker


def main() -> int:
    settings = Settings()
    gate = float(settings.rerank_min_score)
    forms = max(int(settings.rerank_query_forms), 1)
    shortlist = int(settings.rerank_shortlist)
    torch.set_num_threads(settings.torch_num_threads or torch.get_num_threads())

    passages_all = _load_kb_chunks()
    if not passages_all:
        print("FAIL could not load KB chunks from data/raw — nothing to score")
        return 1

    print(f"gate (rerank_min_score) {gate}")
    print(f"shortlist               {shortlist}")
    print(f"query forms             {forms}")
    print(f"KB chunks available     {len(passages_all)}")
    print()

    fp32 = _build(quantize=False)
    int8 = _build(quantize=True)

    # Cheap lexical prefilter to pick each query's shortlist: the point is to get
    # production-plausible passages in front of the gate, not to reproduce RRF.
    def shortlist_for(query: str) -> list[str]:
        terms = {t for t in query.lower().split() if len(t) > 3}
        scored = [
            (sum(1 for t in terms if t in p.lower()), i, p)
            for i, p in enumerate(passages_all)
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [p for _, _, p in scored[:shortlist]]

    header = (
        f"{'kind':<15}{'fp32 max':>11}{'int8 max':>11}{'drift':>9}"
        f"{'fp32':>7}{'int8':>7}{'margin':>9}  query"
    )
    print(header)
    print("-" * len(header))

    worst_margin = float("inf")
    worst_margin_query = ""
    worst_drift = 0.0
    flips = 0
    rows = 0

    for query, _expected, kind in EVAL_QUERIES:
        passages = shortlist_for(query)
        if not passages:
            continue
        # Duplicate the query to reach production's pair count (shortlist x forms)
        # rather than inventing a second phrasing. Batch SHAPE is what drives int8
        # drift — 16-scored-together != 8+8 — so matching 16 pairs matters. The
        # max-pool over identical forms yields the single-form logit, which is
        # conservative: a genuine second phrasing can only raise the max, never
        # lower it, so real margins are >= the ones reported here.
        queries = [query] * forms
        a = fp32.score_multi(queries, passages)
        b = int8.score_multi(queries, passages)
        if a is None or b is None:
            print(f"{kind:<15}{'n/a':>11}{'n/a':>11}   reranker unavailable  {query[:40]}")
            continue

        a_max, b_max = max(a), max(b)
        drift = max(abs(x - y) for x, y in zip(a, b))
        a_pass = a_max >= gate
        b_pass = b_max >= gate
        margin = abs(a_max - gate)

        worst_drift = max(worst_drift, drift)
        rows += 1
        if margin < worst_margin:
            worst_margin, worst_margin_query = margin, query
        flipped = a_pass != b_pass
        flips += int(flipped)

        mark = " <== FLIPPED" if flipped else ""
        print(
            f"{kind:<15}{a_max:>11.4f}{b_max:>11.4f}{drift:>9.2e}"
            f"{'answer' if a_pass else 'DECLINE':>7}"
            f"{'answer' if b_pass else 'DECLINE':>7}"
            f"{margin:>9.3f}  {query[:38]}{mark}"
        )

    print()
    print(f"queries scored              {rows}")
    print(f"worst int8 drift observed   {worst_drift:.3e}")
    print(f"smallest gate margin        {worst_margin:.3f}  ({worst_margin_query[:50]!r})")
    print(f"gate verdict flips          {flips}")
    print()

    if rows == 0:
        print("FAIL no queries scored")
        return 1
    if flips:
        print(f"FAIL int8 flipped {flips} answer/decline verdict(s). Set RERANK_QUANTIZE=false.")
        return 1
    if worst_drift <= 0:
        print("WARN int8 produced no drift at all — is the quantization actually applied?")
        return 1

    ratio = worst_margin / worst_drift
    print(f"safety ratio (margin / drift) {ratio:.1f}x")
    if ratio < 3.0:
        print(
            f"FAIL margin {worst_margin:.3f} is only {ratio:.1f}x the {worst_drift:.3f} "
            "drift — too close to the gate to call int8 safe."
        )
        return 1
    print(
        f"PASS every verdict clears the gate by at least {worst_margin:.3f}, "
        f"{ratio:.1f}x the worst {worst_drift:.3f} int8 drift."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
