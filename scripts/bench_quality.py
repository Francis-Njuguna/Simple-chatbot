"""Retrieval-quality benchmark against the labeled eval set.

Reports the three numbers that actually decide answer quality, per query class:

* **recall@k** — did retrieval surface *any* article that answers the question?
  If this is 0 the LLM cannot possibly answer correctly, no matter how good the
  prompt is. This is the ceiling on answer quality.
* **precision** — what fraction of returned chunks came from an expected article?
  Low precision means the model is reading distractors alongside the answer.
* **confidence separation** — mean confidence on `covered` queries minus mean
  confidence on `offtopic` ones. A confidence score that does not separate these
  is not measuring anything, however high its absolute values look.

Retrieval only — no LLM calls — so it runs in seconds and is safe to run on every
change. ``bench_e2e.py`` covers end-to-end latency and answer text.

Usage:
    ./.venv/Scripts/python.exe -u scripts/bench_quality.py
    ./.venv/Scripts/python.exe -u scripts/bench_quality.py --json out.json
    ./.venv/Scripts/python.exe -u scripts/bench_quality.py --compare baseline.json
"""

import asyncio
import json
import logging
import os
import statistics
import sys
import time

# Must precede any torch import — see scripts/reingest.py for the OpenMP story.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, ".")

from scripts.eval_set import EVAL_QUERIES  # noqa: E402


def _setup_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s",
                        stream=sys.stdout, force=True)
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "urllib3", "openai",
                  "backend.app.rag.retriever", "backend.app.rag.lexical"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


async def main() -> None:
    _setup_logging()
    json_out = None
    compare_to = None
    if "--json" in sys.argv:
        json_out = sys.argv[sys.argv.index("--json") + 1]
    if "--compare" in sys.argv:
        compare_to = sys.argv[sys.argv.index("--compare") + 1]

    # Embedding model first, then Chroma — the import-order constraint.
    from backend.app.rag.embeddings import get_embedding_service

    await get_embedding_service().embed_query_async("warmup")

    from backend.app.rag.lexical import get_lexical_index
    from backend.app.rag.reranker import get_reranker

    get_lexical_index().rebuild()
    get_reranker().warmup()
    get_reranker().score("warmup", ["warmup passage"])

    from backend.app.database.session import async_session_factory
    from backend.app.rag.retriever import get_retriever

    retriever = get_retriever()

    rows: list[dict] = []
    for query, expected, kind in EVAL_QUERIES:
        # Distinct queries, and the cache is keyed by query text, so these are
        # cold numbers rather than cache hits.
        async with async_session_factory() as db:
            t0 = time.perf_counter()
            chunks, images = await retriever.retrieve(query, db=db)
            elapsed = (time.perf_counter() - t0) * 1000

        got = [c.article_id for c in chunks]
        expected_set = set(expected)
        hits = [a for a in got if a in expected_set]

        if kind == "offtopic":
            # Correct behaviour is returning nothing at all.
            recall = 1.0 if not chunks else 0.0
            precision = 1.0 if not chunks else 0.0
        else:
            recall = 1.0 if hits else 0.0
            precision = (len(hits) / len(got)) if got else 0.0

        confidence = retriever.compute_confidence(chunks)
        rows.append({
            "query": query, "kind": kind, "expected": expected,
            "got": got, "n_chunks": len(chunks), "n_images": len(images),
            "recall": recall, "precision": precision,
            "confidence": round(confidence, 4), "ms": round(elapsed, 1),
        })

    # ---------------- report ----------------
    print()
    print(f"{'kind':<9} {'rec':>4} {'prec':>5} {'conf':>5} {'n':>3} {'ms':>6}  query")
    print("-" * 96)
    for r in rows:
        flag = " " if r["recall"] == 1.0 else "*"
        print(f"{r['kind']:<9} {r['recall']:>4.0f} {r['precision']:>5.2f} "
              f"{r['confidence']:>5.2f} {r['n_chunks']:>3} {r['ms']:>6.0f} {flag} {r['query'][:52]}")

    print()
    print("=" * 96)
    summary: dict = {}
    for kind in ("covered", "partial", "offtopic"):
        group = [r for r in rows if r["kind"] == kind]
        if not group:
            continue
        summary[kind] = {
            "n": len(group),
            "recall": statistics.mean(r["recall"] for r in group),
            "precision": statistics.mean(r["precision"] for r in group),
            "confidence": statistics.mean(r["confidence"] for r in group),
            "chunks": statistics.mean(r["n_chunks"] for r in group),
        }
        s = summary[kind]
        print(f"{kind:<9} n={s['n']:<3} recall={s['recall']:.3f}  "
              f"precision={s['precision']:.3f}  mean_conf={s['confidence']:.3f}  "
              f"mean_chunks={s['chunks']:.1f}")

    all_ms = [r["ms"] for r in rows]
    summary["latency_ms"] = {
        "mean": statistics.mean(all_ms),
        "p95": sorted(all_ms)[min(len(all_ms) - 1, int(0.95 * len(all_ms)))],
        "max": max(all_ms),
    }
    print(f"{'latency':<9} mean={summary['latency_ms']['mean']:.0f}ms  "
          f"p95={summary['latency_ms']['p95']:.0f}ms  max={summary['latency_ms']['max']:.0f}ms")

    # The headline quality number: a confidence score is only useful if it is
    # higher on answerable queries than on nonsense.
    if "covered" in summary and "offtopic" in summary:
        sep = summary["covered"]["confidence"] - summary["offtopic"]["confidence"]
        summary["confidence_separation"] = sep
        print(f"{'conf sep':<9} covered - offtopic = {sep:+.3f}   "
              f"(higher is better; <=0 means the score is uninformative)")
    print("=" * 96)

    payload = {"summary": summary, "rows": rows}

    if json_out:
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {json_out}")

    if compare_to:
        with open(compare_to, encoding="utf-8") as fh:
            base = json.load(fh)
        print(f"\nvs {compare_to}:")
        print(f"{'metric':<28} {'before':>9} {'after':>9} {'delta':>9}")
        print("-" * 58)
        for kind in ("covered", "partial", "offtopic"):
            if kind not in summary or kind not in base["summary"]:
                continue
            for metric in ("recall", "precision", "confidence"):
                b = base["summary"][kind][metric]
                a = summary[kind][metric]
                print(f"{kind + '.' + metric:<28} {b:>9.3f} {a:>9.3f} {a - b:>+9.3f}")
        if "confidence_separation" in summary and "confidence_separation" in base["summary"]:
            b = base["summary"]["confidence_separation"]
            a = summary["confidence_separation"]
            print(f"{'confidence_separation':<28} {b:>9.3f} {a:>9.3f} {a - b:>+9.3f}")
        b = base["summary"]["latency_ms"]["mean"]
        a = summary["latency_ms"]["mean"]
        print(f"{'latency_ms.mean':<28} {b:>9.0f} {a:>9.0f} {a - b:>+9.0f}")

        # Per-query regressions matter more than the aggregate: a mean that
        # improves while three queries break is not an improvement.
        base_rows = {r["query"]: r for r in base["rows"]}
        regressions = [
            r for r in rows
            if r["query"] in base_rows and r["recall"] < base_rows[r["query"]]["recall"]
        ]
        if regressions:
            print(f"\n!! {len(regressions)} RECALL REGRESSION(S):")
            for r in regressions:
                print(f"   {r['query']}  (expected {r['expected']}, got {r['got']})")
        else:
            print("\nno per-query recall regressions")


if __name__ == "__main__":
    asyncio.run(main())
