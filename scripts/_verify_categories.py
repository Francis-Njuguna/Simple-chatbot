"""Ad-hoc end-to-end check: does the live /api/v1/chat endpoint answer, per query class?

This is the proof that was never taken after the nemotron model swap (commit
5ce6cc5): config resolving to a live model and the server booting clean say
nothing about whether the widget gets a real answer back.

It talks HTTP to a running server rather than importing ``RAGService`` directly,
because the thing under test is the deployed path the widget uses — routing,
serialisation, rate limiting and all.

Query classes come from ``scripts/eval_set.py``'s taxonomy so the verdicts stay
comparable with the retrieval benchmarks:

    covered / synonym / typo / conversational / short / partial / offtopic

Verdicts per row
----------------
    OK       real grounded answer (covered-style classes)
    DECLINE  model declined — correct for offtopic, a failure for covered
    LLM_ERR  the provider-outage error string (llm.py:248-252) — the exact
             failure the model swap was meant to fix
    HTTP_ERR non-200

Usage
-----
    ./.venv/Scripts/python.exe -u scripts/_verify_categories.py
    ./.venv/Scripts/python.exe -u scripts/_verify_categories.py --port 8001 --show
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict

# (query, class) — two per class where the class has distinct failure modes.
# Kept to 20 rows: every row is a real LLM call against a rate-limited endpoint.
CASES: list[tuple[str, str]] = [
    # ---- covered: KB documents this; answering is mandatory ----
    ("How do I log in to the LMS?", "covered"),
    ("How do I set up Microsoft Authenticator?", "covered"),
    ("What is SMOWL proctoring?", "covered"),
    ("How do I reset my student portal password?", "covered"),
    ("How do I access my student email?", "covered"),
    ("How do I register for supplementary exams?", "covered"),
    ("What is My Loft?", "covered"),
    ("How do I contact the AmIU help desk?", "covered"),

    # ---- synonym: right topic, vocabulary the article never uses ----
    ("How do I log into Moodle?", "synonym"),
    ("how do I turn on 2FA", "synonym"),
    ("SIS password reset", "synonym"),

    # ---- typo: realistic help-desk misspellings ----
    ("moddle login", "typo"),
    ("athenticator app setup", "typo"),
    ("smwol camera not working", "typo"),

    # ---- conversational: natural phrasing, no title keyword overlap ----
    ("my webcam isn't being detected during the online exam", "conversational"),
    ("trying to check my university email but it won't let me in", "conversational"),

    # ---- short: bare keywords, most likely to be confidently wrong ----
    ("password", "short"),
    ("proctoring", "short"),

    # ---- partial: adjacent topic covered, exact ask is not ----
    ("Where do I check my grades?", "partial"),

    # ---- offtopic: MUST decline ----
    ("What is the capital of France?", "offtopic"),
]

# llm.py returns _error_message(exc) AS THE ANSWER on any provider failure
# instead of raising, so a harness that only watches for exceptions reports
# 100% success during a total outage. These are that string's stable clauses.
LLM_ERROR_MARKERS = (
    "could not generate an answer",
    "not a gap in the knowledge base",
    "check the server logs",
)

DECLINE_MARKERS = (
    "isn't covered", "is not covered", "not covered", "outside the scope",
    "outside my scope", "don't have information", "do not have information",
    "no information", "cannot help with", "can't help with",
    "not something i can", "unrelated to", "not related to",
    "not in the knowledge base", "no relevant articles",
    "doesn't appear in", "does not appear in", "beyond what", "not part of",
)

# The chain-of-thought leak seen on raw nemotron probes: if the model's
# reasoning reaches the content field, answers start like an internal monologue.
COT_MARKERS = (
    "okay, the user", "the user is asking", "let me think", "i need to figure",
    "first, i should", "<think>",
)

# First sentence, where a genuine refusal always lives.
_FIRST_SENTENCE_RE = re.compile(r"^.*?(?:[.!?](?:\s|$)|\n)", re.DOTALL)


def _norm(text: str) -> str:
    """Lowercase with typographic punctuation folded to ASCII.

    The model emits U+2019, so a raw ``"don't have information" in answer`` is
    dead code — it silently never fires, and the markers that happen to lack an
    apostrophe carry the whole check. That hid a real decline from this harness.
    """
    return text.lower().replace("\u2019", "'").replace("\u201b", "'")


def classify(answer: str) -> str:
    """OK | DECLINE | LLM_ERR, judged by where the refusal sits.

    Matching DECLINE_MARKERS anywhere is wrong and produced four false FAILs on
    the first run: a *correct* grounded answer routinely closes with a scope
    caveat ("...that part is not covered; contact the Help Desk"), which rule 8
    of SYSTEM_PROMPT actively asks for. The marker is evidence of compliance.

    A genuine decline refuses in its opening sentence; a caveat arrives only
    after the answer has been given. Position is therefore the whole test — a
    step-list heuristic would misgrade the declines rule 8 tells to enumerate
    in-scope topics.
    """
    low = _norm(answer)
    if any(m in low for m in LLM_ERROR_MARKERS):
        return "LLM_ERR"
    opening = (_FIRST_SENTENCE_RE.match(low) or [low[:200]])[0]
    return "DECLINE" if any(m in opening for m in DECLINE_MARKERS) else "OK"


def post(url: str, body: dict, timeout: float) -> tuple[int, dict | str, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            dt = time.perf_counter() - t0
            try:
                return resp.status, json.loads(raw), dt
            except json.JSONDecodeError:
                return resp.status, raw, dt
    except urllib.error.HTTPError as exc:
        dt = time.perf_counter() - t0
        return exc.code, exc.read().decode("utf-8", "replace")[:300], dt
    except Exception as exc:  # noqa: BLE001 - report, don't crash the sweep
        dt = time.perf_counter() - t0
        return 0, f"{type(exc).__name__}: {exc}", dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--show", action="store_true", help="print full answers")
    ap.add_argument("--json", help="write raw results here")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/api/v1/chat"
    # A fresh session id per query: a shared session accumulates history and the
    # later rows would be testing conversation memory, not retrieval.
    rows = []

    print(f"POST {url}   ({len(CASES)} queries)\n")
    header = f"{'#':>2}  {'class':<15} {'verdict':<8} {'s':>6} {'conf':>5} {'src':>3}  query"
    print(header)
    print("-" * len(header))

    for i, (query, kind) in enumerate(CASES, 1):
        status, payload, dt = post(
            url,
            {"message": query, "session_id": str(uuid.uuid4())},
            args.timeout,
        )
        if status != 200 or not isinstance(payload, dict):
            verdict, conf, nsrc, answer = "HTTP_ERR", 0.0, 0, str(payload)[:200]
        else:
            answer = payload.get("answer", "")
            verdict = classify(answer)
            conf = float(payload.get("confidence", 0.0))
            nsrc = len(payload.get("sources", []))

        low = answer.lower()
        cot = any(m in low for m in COT_MARKERS)

        rows.append({
            "query": query, "kind": kind, "verdict": verdict, "status": status,
            "seconds": round(dt, 2), "confidence": round(conf, 3),
            "sources": nsrc, "chars": len(answer), "cot_leak": cot,
            "answer": answer,
        })
        flag = "  <-- CoT LEAK" if cot else ""
        print(f"{i:>2}  {kind:<15} {verdict:<8} {dt:>6.1f} {conf:>5.2f} {nsrc:>3}  {query[:46]}{flag}")
        if args.show:
            print(f"      {answer[:600]}\n")

    # ---- per-class summary ----
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)

    print("\nPER-CLASS SUMMARY")
    print(f"{'class':<15} {'n':>2} {'OK':>3} {'DECL':>4} {'ERR':>3} {'HTTP':>4} {'med s':>6} {'expected':<12} verdict")
    print("-" * 78)

    overall_pass = True
    for kind in ("covered", "synonym", "typo", "conversational", "short", "partial", "offtopic"):
        rs = by_kind.get(kind)
        if not rs:
            continue
        n = len(rs)
        ok = sum(r["verdict"] == "OK" for r in rs)
        dec = sum(r["verdict"] == "DECLINE" for r in rs)
        err = sum(r["verdict"] == "LLM_ERR" for r in rs)
        http = sum(r["verdict"] == "HTTP_ERR" for r in rs)
        times = sorted(r["seconds"] for r in rs)
        med = times[len(times) // 2]

        if kind == "offtopic":
            want, good = "DECLINE", dec == n
        elif kind == "partial":
            want, good = "answer|decline", err == 0 and http == 0
        else:
            want, good = "OK", ok == n
        overall_pass &= good
        print(f"{kind:<15} {n:>2} {ok:>3} {dec:>4} {err:>3} {http:>4} {med:>6.1f} {want:<12} {'PASS' if good else 'FAIL'}")

    errs = sum(r["verdict"] == "LLM_ERR" for r in rows)
    https = sum(r["verdict"] == "HTTP_ERR" for r in rows)
    cots = sum(r["cot_leak"] for r in rows)
    times = sorted(r["seconds"] for r in rows)
    print(
        f"\nTOTAL {len(rows)} queries | LLM_ERR {errs} | HTTP_ERR {https} | CoT leaks {cots}"
        f" | latency min {times[0]:.1f}s p50 {times[len(times)//2]:.1f}s max {times[-1]:.1f}s"
    )
    print("OVERALL:", "PASS" if overall_pass else "FAIL")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print("wrote", args.json)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
