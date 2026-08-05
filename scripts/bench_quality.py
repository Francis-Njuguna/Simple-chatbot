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

from scripts.eval_set import EVAL_QUERIES, STABILITY_GROUPS  # noqa: E402

# Report order. Driven off the eval set rather than hardcoded so adding a query
# class does not silently vanish from the summary — the previous hardcoded tuple
# meant a new class was scored per-query but omitted from every aggregate.
_KIND_ORDER = [
    "covered", "synonym", "typo", "conversational",
    "short", "long", "partial", "offtopic",
]
KINDS = [k for k in _KIND_ORDER if any(k == kind for _, _, kind in EVAL_QUERIES)]
KINDS += sorted({kind for _, _, kind in EVAL_QUERIES} - set(_KIND_ORDER))


def _jaccard(a: set, b: set) -> float:
    """Overlap of two sets in [0, 1]. Two empty sets count as identical.

    Both phrasings retrieving nothing IS consistent behaviour — the failure
    this metric hunts is two phrasings of the same question disagreeing, and
    "both declined" is agreement.
    """
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


async def measure_stability(retriever, session_factory) -> dict:
    """Pairwise top-k article overlap within each synonym group.

    This is the metric per-query recall cannot express. Five phrasings of
    "I forgot my password" can each retrieve a *correct* article while
    retrieving five *different* correct articles — recall reads 1.000 and the
    user still gets a different answer depending on wording.

    Every unordered pair within a group is compared on its retrieved article
    set; the group score is the mean. A group scoring 1.000 means every
    phrasing collapsed onto exactly the same articles, which is the Objective-1
    target stated as a number.
    """
    results: dict[str, dict] = {}
    for group, queries in STABILITY_GROUPS.items():
        retrieved: dict[str, set] = {}
        confidences: dict[str, float] = {}
        for query in queries:
            async with session_factory() as db:
                chunks, _images, processed = await retriever.retrieve(query, db=db)
            retrieved[query] = {c.article_id for c in chunks}
            confidences[query] = retriever.compute_confidence(chunks, processed)

        pairs: list[float] = []
        for i, qa in enumerate(queries):
            for qb in queries[i + 1:]:
                pairs.append(_jaccard(retrieved[qa], retrieved[qb]))

        # Top-1 agreement: did every phrasing land on the same single best
        # article? Stricter than Jaccard and closer to what the user perceives,
        # since the top article dominates the assembled context.
        tops = [sorted(retrieved[q])[:1] for q in queries]
        top1_values = [t[0] for t in tops if t]
        top1_agree = (
            len(set(top1_values)) == 1 and len(top1_values) == len(queries)
        )

        results[group] = {
            "n_queries": len(queries),
            "mean_overlap": statistics.mean(pairs) if pairs else 0.0,
            "min_overlap": min(pairs) if pairs else 0.0,
            "top1_agreement": top1_agree,
            "conf_spread": (
                max(confidences.values()) - min(confidences.values())
                if confidences else 0.0
            ),
            "per_query": {q: sorted(retrieved[q]) for q in queries},
        }
    return results



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
            chunks, images, processed = await retriever.retrieve(query, db=db)
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

        confidence = retriever.compute_confidence(chunks, processed)
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
    for kind in KINDS:
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

    # ---------------- synonym stability ----------------
    stability = await measure_stability(retriever, async_session_factory)
    print()
    print("SYNONYM STABILITY — do differently-worded versions of the same question")
    print("retrieve the same articles? (1.000 = identical retrieval)")
    print("-" * 96)
    print(f"{'group':<18} {'n':>3} {'mean':>6} {'min':>6} {'top1':>6} {'confΔ':>7}")
    for group, s in stability.items():
        print(f"{group:<18} {s['n_queries']:>3} {s['mean_overlap']:>6.3f} "
              f"{s['min_overlap']:>6.3f} {'yes' if s['top1_agreement'] else 'NO':>6} "
              f"{s['conf_spread']:>7.3f}")

    overall = statistics.mean(s["mean_overlap"] for s in stability.values())
    worst = min(s["min_overlap"] for s in stability.values())
    agreed = sum(1 for s in stability.values() if s["top1_agreement"])
    print("-" * 96)
    print(f"{'OVERALL':<18} mean_overlap={overall:.3f}  worst_pair={worst:.3f}  "
          f"top1_agreement={agreed}/{len(stability)}")
    print("=" * 96)

    summary["stability"] = {
        "mean_overlap": overall,
        "worst_pair": worst,
        "top1_agreement": agreed / len(stability) if stability else 0.0,
        "groups": {g: {k: v for k, v in s.items() if k != "per_query"}
                   for g, s in stability.items()},
    }

    payload = {"summary": summary, "rows": rows, "stability_detail": stability}

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
        for kind in KINDS:
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

        if "stability" in base["summary"]:
            for metric in ("mean_overlap", "worst_pair", "top1_agreement"):
                b = base["summary"]["stability"][metric]
                a = summary["stability"][metric]
                print(f"{'stability.' + metric:<28} {b:>9.3f} {a:>9.3f} {a - b:>+9.3f}")

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

        # Stability regressions are the ones this whole effort exists to
        # prevent, and they are invisible in recall — a group can hold
        # recall=1.000 while its phrasings drift onto different articles.
        base_stab = base["summary"].get("stability", {}).get("groups", {})
        if base_stab:
            drops = [
                (g, base_stab[g]["mean_overlap"], s["mean_overlap"])
                for g, s in stability.items()
                if g in base_stab
                and s["mean_overlap"] < base_stab[g]["mean_overlap"] - 1e-9
            ]
            if drops:
                print(f"\n!! {len(drops)} STABILITY REGRESSION(S):")
                for group, before, after in drops:
                    print(f"   {group}: {before:.3f} -> {after:.3f}")
            else:
                print("no synonym-stability regressions")


if __name__ == "__main__":
    asyncio.run(main())
