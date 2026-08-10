"""End-to-end ANSWER quality benchmark, through the real ``RAGService``.

What this covers that bench_quality.py and bench_e2e.py do not
-------------------------------------------------------------
* ``bench_quality.py`` scores *retrieval* — did the right article surface. It
  never calls the LLM, so it cannot see a correct retrieval turned into a wrong
  answer.
* ``bench_e2e.py`` calls the retriever and the LLM directly, assembling context
  itself. It deliberately bypasses ``RAGService``, so it exercises none of the
  session/persistence logic the connection-lifecycle refactor changed.

This runs ``RAGService.chat()`` — the exact code path a real /chat request takes,
including the short ``db_scope`` blocks, hydration, persistence and message ids.
It is therefore the test that would catch a refactor that kept latency and
retrieval intact while quietly breaking the produced answer.

How answers are graded
----------------------
Without a human or a judge model, "is this answer good" is not directly
computable, so this scores the properties that are checkable and that a broken
pipeline actually breaks:

* **grounding**   — fraction of the answer's KB-specific claims that appear in
  the retrieved context. Computed as overlap of distinctive terms, so it detects
  the failure that matters: an answer invented rather than read.
* **decline correctness** — off-topic questions MUST be declined and covered
  questions MUST NOT be. This is binary and is the single clearest quality
  signal in the whole suite; a pipeline that lost its context silently starts
  declining everything.
* **citations**  — did the response carry sources, and do they match what
  retrieval actually returned.
* **persistence** — did the turn produce a message id and a session id, i.e.
  did the post-LLM ``db_scope`` actually commit. A refactor that dropped the
  persist scope would still return a perfect answer.
* **length / confidence** — regression signals against a stored baseline rather
  than absolute targets.

Usage
-----
    ./.venv/Scripts/python.exe -u scripts/bench_answers.py
    ./.venv/Scripts/python.exe -u scripts/bench_answers.py --json answers.json
    ./.venv/Scripts/python.exe -u scripts/bench_answers.py --compare answers.json
    ./.venv/Scripts/python.exe -u scripts/bench_answers.py --show    # print answers
    ./.venv/Scripts/python.exe -u scripts/bench_answers.py --repeat 3  # consistency
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time

# Must precede any torch import — see scripts/reingest.py for the OpenMP story.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, ".")


# One representative query per class, plus the cases that break in distinct
# ways. Kept small because every row is a real LLM call against a rate-limited
# free tier; the point is coverage of failure *modes*, not sample size.
CASES: list[tuple[str, str, list[str]]] = [
    # (query, kind, terms the answer should contain if it is grounded)
    ("How do I reset my student portal password?", "covered", ["password", "portal"]),
    ("What is SMOWL proctoring?", "covered", ["smowl"]),
    ("How do I set up Microsoft Authenticator?", "covered", ["authenticator"]),
    ("How do I log in to the LMS?", "covered", ["login", "lms", "moodle"]),
    ("How do I access my student email?", "covered", ["email"]),
    ("How do I contact the AmIU help desk?", "covered", ["help", "desk"]),
    ("How do I register for supplementary exams?", "covered", ["exam"]),

    # Same questions, vocabulary the article never uses. Tests that synonym
    # expansion survives all the way to the answer, not just to retrieval.
    ("I forgot my SIS password", "synonym", ["password"]),
    ("how do I turn on 2FA", "synonym", ["authenticator", "mfa", "2fa"]),
    ("moodle login help", "synonym", ["login"]),

    # Misspellings. A pipeline that lost spell-correction answers these wrongly
    # while still answering their clean forms perfectly.
    ("reset my studnet portal pasword", "typo", ["password"]),
    ("smwol camera not working", "typo", ["camera", "smowl"]),
    ("athenticator app setup", "typo", ["authenticator"]),

    # Natural phrasing, no keyword overlap with article titles.
    ("my webcam isn't being detected during the online exam", "conversational",
     ["camera", "webcam"]),
    ("trying to check my university email but it won't let me in", "conversational",
     ["email"]),

    # Bare keywords — the hardest retrieval case, and the one most likely to
    # produce a confidently wrong answer.
    ("password", "short", ["password"]),
    ("proctoring", "short", ["proctor", "smowl"]),

    # The KB covers the adjacent topic but not the exact ask. Correct behaviour
    # is to answer what IS covered and say plainly what is not — neither a flat
    # decline nor an invented answer.
    ("I cannot access my assignments", "partial", []),
    ("Where do I check my grades?", "partial", []),

    # Nothing in the KB relates. MUST decline.
    ("What is the capital of France?", "offtopic", []),
    ("How do I bake a chocolate cake?", "offtopic", []),
    ("Who won the 2022 World Cup?", "offtopic", []),
]

# ``LLMService.generate_answer`` catches every exception and RETURNS
# ``_error_message(exc)`` as the answer rather than raising (llm.py:266-271).
# A benchmark that only watches for exceptions therefore reports a 100% success
# rate during a total provider outage, with every "answer" being an error
# string. These are that string's stable clauses — see llm.py:248-252.
LLM_ERROR_MARKERS = (
    "could not generate an answer",
    "not a gap in the knowledge base",
    "check the server logs",
)

# Rule 8 of SYSTEM_PROMPT tells the model to decline off-topic questions and to
# "vary the wording naturally", so no fixed phrase list can be complete. These
# catch the common phrasings; ``classify_answer`` adds a structural fallback for
# the ones that word it differently.
DECLINE_MARKERS = (
    "isn't covered", "is not covered", "not covered", "outside the scope",
    "outside my scope", "not able to", "don't have information",
    "do not have information", "no information", "cannot help with",
    "can't help with", "not something i can", "unrelated to",
    "not related to", "i don't have", "not in the knowledge base",
    "no relevant articles", "doesn't appear in", "does not appear in",
    "isn't something", "beyond what", "not part of",
)

# Rule 8 also mandates that a decline point at the help desk and list the
# in-scope topics. That combination — help-desk pointer, no numbered procedure —
# identifies a decline whose wording the list above misses.
HELPDESK_MARKERS = ("helpdesk.amref.ac.ke", "help desk", "knowledge base")

_STEP_RE = re.compile(r"(^|\n)\s*(?:\d+[.)]|step\s+\d)", re.IGNORECASE)

# Words too common to indicate grounding. Kept deliberately short: the goal is
# to drop filler, not to build a real stopword list.
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "with", "from", "by", "as", "it", "its",
    "you", "your", "we", "our", "they", "their", "i", "me", "my", "will",
    "can", "could", "should", "would", "may", "might", "do", "does", "did",
    "have", "has", "had", "not", "no", "yes", "please", "here", "there",
    "how", "what", "when", "where", "which", "who", "why", "step", "steps",
    "click", "select", "open", "go", "see", "use", "using", "need", "want",
    "help", "also", "any", "all", "more", "most", "some", "other", "into",
    "out", "up", "down", "after", "before", "once", "each", "both", "about",
}


def _terms(text: str) -> set[str]:
    """Distinctive lowercase words, 4+ chars, minus filler."""
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) >= 4 and w not in _STOP
    }


def grounding_score(answer: str, context: str) -> float:
    """Fraction of the answer's distinctive terms that appear in the context.

    A proxy for "did the model read the retrieved articles or improvise". Not a
    correctness measure — a grounded answer can still be unhelpful — but the
    failure it detects (fluent text with no support in the KB) is exactly what a
    broken retrieval-to-context path produces, and that failure is invisible to
    every other metric here.

    Scored only on answers long enough to be meaningful; a two-line decline has
    too few terms for the ratio to mean anything.
    """
    answer_terms = _terms(answer)
    if len(answer_terms) < 8:
        return float("nan")
    context_terms = _terms(context)
    if not context_terms:
        return 0.0
    return len(answer_terms & context_terms) / len(answer_terms)


def llm_failed(answer: str) -> bool:
    """True when the 'answer' is really ``LLMService._error_message``.

    Necessary because ``generate_answer`` swallows every exception and returns
    the error text as the answer, so an outage otherwise scores as 22 successful
    requests that all mysteriously declined.
    """
    low = answer.lower()
    return any(m in low for m in LLM_ERROR_MARKERS)


def declined(answer: str) -> bool:
    """True when the answer declines rather than solving the problem.

    Two ways to qualify, because rule 8 tells the model to "vary the wording
    naturally" and a fixed phrase list cannot keep up:

    * an explicit decline phrase, or
    * the structural signature of one — points at the help desk, gives no
      numbered procedure, and is short. A real answer under this prompt is
      numbered steps and runs far longer.
    """
    low = answer.lower()
    if any(m in low for m in DECLINE_MARKERS):
        return True
    points_elsewhere = any(m in low for m in HELPDESK_MARKERS)
    return points_elsewhere and not _STEP_RE.search(answer) and len(answer) < 700


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "urllib3", "openai",
                  "backend.app.rag.retriever", "backend.app.rag.lexical"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


async def run_case(service, retriever, query: str, kind: str, expect_terms: list[str]) -> dict:
    """One query through the real service, graded."""
    # Retrieve separately to obtain the context the service will build, so
    # grounding can be scored against it. Same query text means the retrieval
    # cache serves the service's own call, so this does not double the work or
    # change what the service sees.
    chunks, images, _processed = await retriever.retrieve(query)
    context = retriever.format_context(chunks)

    t0 = time.perf_counter()
    error = ""
    try:
        response = await service.chat(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        answer = response.answer
        row = {
            "confidence": round(response.confidence, 4),
            "n_sources": len(response.sources),
            "source_ids": [s.article_id for s in response.sources],
            "n_images": len(response.images),
            # Proof the post-LLM db_scope committed. A refactor that dropped the
            # persist scope returns a perfect answer with no message id.
            "persisted": bool(response.message_id) and bool(response.session_id),
        }
    except BaseException as exc:  # noqa: BLE001 - a failure is a result
        elapsed_ms = (time.perf_counter() - t0) * 1000
        answer, error = "", f"{type(exc).__name__}: {exc}"
        row = {"confidence": 0.0, "n_sources": 0, "source_ids": [],
               "n_images": 0, "persisted": False}

    did_decline = declined(answer) if answer else False
    provider_failed = llm_failed(answer) if answer else False
    should_decline = kind == "offtopic"
    # `partial` is deliberately excluded from the pass/fail: the correct answer
    # there both answers and declines in part, so neither verdict is wrong.
    # A provider failure is excluded too — it says nothing about answer quality,
    # and scoring it as one would blame the pipeline for an outage.
    decline_ok = (
        None if kind == "partial" or error or provider_failed
        else did_decline == should_decline
    )

    hit_terms = [t for t in expect_terms if t in answer.lower()]

    row.update({
        "query": query,
        "kind": kind,
        "ok": not error and not provider_failed,
        "error": error or ("LLM provider failed: " + answer[:120] if provider_failed else ""),
        "provider_failed": provider_failed,
        "ms": round(elapsed_ms, 1),
        "answer_chars": len(answer),
        "answer": answer,
        "retrieved_ids": [c.article_id for c in chunks],
        "declined": did_decline,
        "decline_ok": decline_ok,
        "grounding": grounding_score(answer, context) if answer and not provider_failed else 0.0,
        "expected_terms": expect_terms,
        "hit_terms": hit_terms,
        "term_recall": (len(hit_terms) / len(expect_terms)) if expect_terms else None,
    })
    return row


def _mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # drop NaN
    return statistics.mean(clean) if clean else 0.0


def summarize(rows: list[dict]) -> dict:
    kinds = sorted({r["kind"] for r in rows}, key=lambda k: [
        "covered", "synonym", "typo", "conversational", "short", "partial", "offtopic"
    ].index(k))

    summary: dict = {"by_kind": {}}
    for kind in kinds:
        group = [r for r in rows if r["kind"] == kind]
        graded = [r for r in group if r["decline_ok"] is not None]
        summary["by_kind"][kind] = {
            "n": len(group),
            "ok": sum(1 for r in group if r["ok"]),
            "decline_accuracy": (
                sum(1 for r in graded if r["decline_ok"]) / len(graded)
                if graded else None
            ),
            "grounding": _mean([r["grounding"] for r in group if r["ok"]]),
            "confidence": _mean([r["confidence"] for r in group]),
            "answer_chars": _mean([float(r["answer_chars"]) for r in group]),
            "term_recall": _mean(
                [r["term_recall"] for r in group if r["term_recall"] is not None]
            ),
            "ms": _mean([r["ms"] for r in group]),
        }

    graded = [r for r in rows if r["decline_ok"] is not None]
    answered = [r for r in rows if r["ok"] and r["kind"] != "offtopic"]
    summary["overall"] = {
        "n": len(rows),
        "errors": sum(1 for r in rows if not r["ok"]),
        "provider_failures": sum(1 for r in rows if r.get("provider_failed")),
        "decline_accuracy": (
            sum(1 for r in graded if r["decline_ok"]) / len(graded) if graded else 0.0
        ),
        "grounding": _mean([r["grounding"] for r in answered]),
        "persisted": sum(1 for r in rows if r["persisted"]),
        "persist_rate": (
            sum(1 for r in rows if r["persisted"]) / max(1, sum(1 for r in rows if r["ok"]))
        ),
        "mean_ms": _mean([r["ms"] for r in rows]),
        "p95_ms": (
            sorted(r["ms"] for r in rows)[min(len(rows) - 1, int(0.95 * len(rows)))]
            if rows else 0.0
        ),
    }
    return summary


def print_report(rows: list[dict], summary: dict, show: bool) -> None:
    print()
    print(f"{'kind':<14} {'grnd':>5} {'conf':>5} {'src':>4} {'chars':>6} "
          f"{'ms':>7} {'dec':>4} {'persist':>8}  query")
    print("-" * 108)
    for r in rows:
        mark = {True: "ok", False: "BAD", None: "n/a"}[r["decline_ok"]]
        grounding = f"{r['grounding']:.2f}" if r["grounding"] == r["grounding"] else "  - "
        print(
            f"{r['kind']:<14} {grounding:>5} {r['confidence']:>5.2f} "
            f"{r['n_sources']:>4} {r['answer_chars']:>6} {r['ms']:>7.0f} "
            f"{mark:>4} {'yes' if r['persisted'] else 'NO':>8}  {r['query'][:38]}"
        )
        if r["error"]:
            print(f"{'':>14} !! {r['error'][:88]}")
        if show and r["answer"]:
            for line in r["answer"].splitlines():
                print(f"{'':>16}| {line[:100]}")
            print()

    print()
    print("=" * 108)
    print(f"{'kind':<14} {'n':>3} {'decline':>8} {'grnd':>6} {'conf':>6} "
          f"{'terms':>6} {'chars':>7} {'ms':>8}")
    for kind, s in summary["by_kind"].items():
        decline = f"{s['decline_accuracy']:.2f}" if s["decline_accuracy"] is not None else "   -"
        print(f"{kind:<14} {s['n']:>3} {decline:>8} {s['grounding']:>6.3f} "
              f"{s['confidence']:>6.3f} {s['term_recall']:>6.2f} "
              f"{s['answer_chars']:>7.0f} {s['ms']:>8.0f}")

    o = summary["overall"]
    print("-" * 108)
    if o["provider_failures"]:
        print(f"  !! {o['provider_failures']}/{o['n']} requests got an LLM error string back "
              "instead of an answer — the provider is failing, not the pipeline.")
        print("     Those rows are excluded from the quality figures below.")
    print(f"  decline accuracy : {o['decline_accuracy']:.3f}  "
          "(off-topic declined AND covered answered; 1.000 required)")
    print(f"  grounding        : {o['grounding']:.3f}  "
          "(answer terms found in retrieved context)")
    print(f"  persisted        : {o['persisted']}/{o['n']}  "
          "(message_id + session_id returned — proves the persist scope committed)")
    print(f"  errors           : {o['errors']}/{o['n']}")
    print(f"  latency          : mean {o['mean_ms'] / 1000:.2f}s   "
          f"p95 {o['p95_ms'] / 1000:.2f}s")
    print("=" * 108)

    failures = [r for r in rows if r["decline_ok"] is False]
    if failures:
        print()
        print(f"!! {len(failures)} DECLINE FAILURE(S) — the clearest quality signal:")
        for r in failures:
            what = "declined a covered question" if r["declined"] else "answered an off-topic question"
            print(f"   [{r['kind']}] {r['query']}")
            print(f"      {what}; retrieved {r['retrieved_ids'] or 'nothing'}")

    unpersisted = [r for r in rows if r["ok"] and not r["persisted"]]
    if unpersisted:
        print()
        print(f"!! {len(unpersisted)} answer(s) returned without a message id — "
              "the post-LLM DB scope did not commit")

    ungrounded = [
        r for r in rows
        if r["ok"] and not r["declined"] and r["grounding"] == r["grounding"]
        and r["grounding"] < 0.30
    ]
    if ungrounded:
        print()
        print(f"!! {len(ungrounded)} answer(s) below 0.30 grounding — possible fabrication:")
        for r in ungrounded:
            print(f"   [{r['kind']}] {r['query']}  (grounding {r['grounding']:.2f})")


def print_comparison(summary: dict, base: dict) -> None:
    print()
    print(f"vs baseline:")
    print(f"{'metric':<34} {'before':>9} {'after':>9} {'delta':>9}")
    print("-" * 64)
    for metric in ("decline_accuracy", "grounding", "persist_rate", "mean_ms"):
        if metric not in base["overall"]:
            continue
        b, a = base["overall"][metric], summary["overall"][metric]
        print(f"{'overall.' + metric:<34} {b:>9.3f} {a:>9.3f} {a - b:>+9.3f}")
    for kind, s in summary["by_kind"].items():
        if kind not in base.get("by_kind", {}):
            continue
        for metric in ("grounding", "confidence"):
            b, a = base["by_kind"][kind][metric], s[metric]
            if abs(a - b) < 0.005:
                continue
            print(f"{kind + '.' + metric:<34} {b:>9.3f} {a:>9.3f} {a - b:>+9.3f}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", dest="json_out", help="Write results to this file")
    parser.add_argument("--compare", help="Compare against a previous --json file")
    parser.add_argument("--show", action="store_true", help="Print every answer in full")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Run each query N times to measure answer consistency (default 1)",
    )
    parser.add_argument(
        "--kinds", help="Comma-separated subset of query kinds to run",
    )
    parser.add_argument(
        "--model",
        help="Override OPENAI_MODEL for this run only (does not touch .env). "
             "Use when the configured NIM model has no free-tier capacity.",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)

    # Set before any app module reads settings — get_settings() is cached.
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    # Embedding model first, then Chroma — the import-order constraint.
    from backend.app.rag.embeddings import get_embedding_service

    await get_embedding_service().embed_query_async("warmup")

    from backend.app.rag.lexical import get_lexical_index
    from backend.app.rag.reranker import get_reranker

    get_lexical_index().rebuild()
    get_reranker().warmup()
    get_reranker().score("warmup", ["warmup passage"])

    from backend.app.database.session import pool_stats
    from backend.app.rag.retriever import get_retriever
    from backend.app.services.rag_service import RAGService

    retriever = get_retriever()
    # Built exactly as the API builds it: no DB session, its own short scopes.
    service = RAGService()

    cases = CASES
    if args.kinds:
        wanted = {k.strip() for k in args.kinds.split(",")}
        cases = [c for c in CASES if c[1] in wanted]

    print("=" * 108)
    print("END-TO-END ANSWER QUALITY  (through RAGService.chat — the real request path)")
    print("=" * 108)
    from backend.app.config import get_settings

    settings = get_settings()
    print(f"  provider  : {settings.llm_provider} / {os.environ.get('OPENAI_MODEL', '?')}"
          f"  (timeout {settings.llm_timeout}s, retries {settings.llm_max_retries})")
    print(f"  queries   : {len(cases)}" + (f" x {args.repeat} repeats" if args.repeat > 1 else ""))
    print(f"  db pool   : {pool_stats()}")

    rows: list[dict] = []
    unstable: list[tuple[str, str, float]] = []
    for repeat in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- pass {repeat + 1}/{args.repeat} ---")
        for query, kind, terms in cases:
            row = await run_case(service, retriever, query, kind, terms)
            row["pass"] = repeat
            rows.append(row)

    # A connection held past its scope shows up here as a non-zero count while
    # the process is otherwise idle.
    final_pool = pool_stats()

    summary = summarize(rows)
    summary["pool_after"] = final_pool
    print_report([r for r in rows if r["pass"] == 0], summary, args.show)

    print(f"\n  db pool after {len(rows)} chats: checked_out="
          f"{final_pool.get('checked_out')} (0 expected — no leaked connections)")

    # Answer consistency across repeats: same question, same pipeline, different
    # answer lengths means non-determinism the user would perceive as the bot
    # "changing its mind".
    if args.repeat > 1:
        print()
        print("CONSISTENCY ACROSS REPEATS")
        print("-" * 108)
        by_query: dict[str, list[dict]] = {}
        for r in rows:
            by_query.setdefault(r["query"], []).append(r)
        for query, group in by_query.items():
            declines = {r["declined"] for r in group}
            lengths = [r["answer_chars"] for r in group]
            spread = (max(lengths) - min(lengths)) / max(1, statistics.mean(lengths))
            if len(declines) > 1:
                unstable.append((query, "decline flip-flopped", spread))
            elif spread > 0.5:
                unstable.append((query, f"length varied {spread * 100:.0f}%", spread))
        if unstable:
            for query, why, _ in sorted(unstable, key=lambda x: -x[2]):
                print(f"  {why:<28} {query[:60]}")
        else:
            print("  all queries answered consistently across repeats")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json_out}")

    if args.compare:
        with open(args.compare, encoding="utf-8") as fh:
            base = json.load(fh)["summary"]
        print_comparison(summary, base)

    # Non-zero exit on the failures that are unambiguous, so this can gate a
    # change. Grounding and confidence are reported but not gated — they are
    # comparative signals, not thresholds. A provider outage exits 2 rather than
    # 1: it means "not measured", not "quality regressed", and the two must not
    # look alike to a caller.
    o = summary["overall"]
    if o["provider_failures"]:
        return 2
    return 1 if (o["errors"] or o["decline_accuracy"] < 1.0 or unpersisted_count(rows)) else 0


def unpersisted_count(rows: list[dict]) -> int:
    return sum(1 for r in rows if r["ok"] and not r["persisted"])


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
