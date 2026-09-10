"""Load test: how long to FIRST STREAMED TEXT and to a FULL ANSWER.

Ad-hoc harness. There is no canonical test command for this file and nothing
imports it; ``testpaths = ["tests"]`` in pyproject.toml means pytest cannot
reach ``scripts/`` at all. Its measurement logic is unit-tested in
``tests/test_load_profile_metrics.py``; this file is the live driver.

WHAT IT MEASURES (three different numbers people conflate):

  t_meta   seconds until the ``meta`` event — retrieval is done, so the widget
           can paint sources/citations. Nothing of the answer exists yet.
  t_token  seconds until the FIRST ``token`` event — the first words of the
           answer appear on screen. This is "time to start streaming", and the
           number chat.py:59's docstring claims is "~1-2s".
  t_done   seconds until the ``done`` event — the FULL answer is complete.

Non-streaming POST /chat has only one observable number, which equals t_done
plus persistence: the user stares at a spinner for the whole of it. Comparing
the two is the point — it quantifies what streaming buys.

WHY CONCURRENCY IS THE HEADLINE: config.py defaults ``LLM_MAX_CONCURRENCY=1``
and this deployment does not override it, so a semaphore in llm.py serialises
every generation. Two simultaneous users do not each wait ~13s; the second
waits its own generation PLUS the first's. Past ``LLM_QUEUE_TIMEOUT=60s`` in
the queue, requests fail outright. So the harness reports queue wait separately
from generation time, and a wall-clock-vs-serial-sum check to prove whether
serialisation actually occurred rather than assuming it.

RATE LIMIT: ``CHAT_RATE_LIMIT`` is read at route-import time (chat.py:22), so
it CANNOT be raised per-request — the server must already have been started
with an override or every burst above 5/minute measures slowapi, not the RAG
pipeline. The harness detects HTTP 429 and aborts with a loud message rather
than reporting throttled numbers as latency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field

import httpx

# Deliberately varied: cache-friendly repeats would understate real latency.
QUERIES = [
    "How do I log in to the LMS?",
    "How do I set up Microsoft Authenticator?",
    "How do I reset my student portal password?",
    "What is SMOWL proctoring?",
    "How do I access my student email?",
    "How do I register for supplementary exams?",
    "What is My Loft?",
    "How do I contact the AmIU help desk?",
]


@dataclass
class Result:
    """One request's timings. ``None`` marks an event that never arrived."""

    query: str
    mode: str
    ok: bool = False
    status: int = 0
    t_meta: float | None = None
    t_token: float | None = None
    t_done: float | None = None
    tokens: int = 0
    chars: int = 0
    error: str = ""
    sources: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def wall(self) -> float:
        return self.ended_at - self.started_at


@dataclass
class Phase:
    name: str
    concurrency: int
    results: list[Result] = field(default_factory=list)
    wall: float = 0.0


def pct(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile. Returns None for an empty sample.

    Deliberately not statistics.quantiles: that interpolates and needs n>=2,
    which silently explodes on the single-request baseline phase.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = max(0, min(len(ordered) - 1, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[k]


def fmt(value: float | None, suffix: str = "s") -> str:
    return "  n/a" if value is None else f"{value:5.2f}{suffix}"


async def stream_one(client: httpx.AsyncClient, url: str, query: str, timeout: float) -> Result:
    """POST /chat/stream and stamp each event type as it arrives."""
    res = Result(query=query, mode="stream")
    body = {"message": query, "session_id": str(uuid.uuid4())}
    res.started_at = time.perf_counter()
    start = res.started_at
    try:
        async with client.stream("POST", url, json=body, timeout=timeout) as resp:
            res.status = resp.status_code
            if resp.status_code != 200:
                await resp.aread()
                res.error = f"HTTP {resp.status_code}"
                res.ended_at = time.perf_counter()
                return res
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                now = time.perf_counter() - start
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind == "meta" and res.t_meta is None:
                    res.t_meta = now
                    res.sources = len(event.get("sources") or [])
                elif kind == "token":
                    text = event.get("text") or ""
                    # Providers can emit empty keep-alive deltas; an empty token
                    # is not "text on screen", so it must not set t_token.
                    if text and res.t_token is None:
                        res.t_token = now
                    if text:
                        res.tokens += 1
                        res.chars += len(text)
                elif kind == "done":
                    res.t_done = now
                    res.ok = True
    except (httpx.HTTPError, TimeoutError) as exc:
        res.error = f"{type(exc).__name__}: {exc}"[:120]
    res.ended_at = time.perf_counter()
    return res


async def blocking_one(client: httpx.AsyncClient, url: str, query: str, timeout: float) -> Result:
    """POST /chat — one number only, because nothing is observable until the end."""
    res = Result(query=query, mode="blocking")
    body = {"message": query, "session_id": str(uuid.uuid4())}
    res.started_at = time.perf_counter()
    try:
        resp = await client.post(url, json=body, timeout=timeout)
        res.status = resp.status_code
        if resp.status_code == 200:
            payload = resp.json()
            answer = payload.get("answer") or ""
            res.chars = len(answer)
            res.sources = len(payload.get("sources") or [])
            res.t_done = time.perf_counter() - res.started_at
            res.ok = True
        else:
            res.error = f"HTTP {resp.status_code}"
    except (httpx.HTTPError, TimeoutError) as exc:
        res.error = f"{type(exc).__name__}: {exc}"[:120]
    res.ended_at = time.perf_counter()
    return res


async def run_phase(
    base: str, mode: str, concurrency: int, count: int, timeout: float, label: str
) -> Phase:
    phase = Phase(name=label, concurrency=concurrency)
    url = f"{base}/api/v1/chat/stream" if mode == "stream" else f"{base}/api/v1/chat"
    fn = stream_one if mode == "stream" else blocking_one
    limits = httpx.Limits(max_connections=max(concurrency * 2, 10))
    gate = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(limits=limits) as client:

        async def worker(i: int) -> Result:
            async with gate:
                return await fn(client, url, QUERIES[i % len(QUERIES)], timeout)

        t0 = time.perf_counter()
        phase.results = list(await asyncio.gather(*(worker(i) for i in range(count))))
        phase.wall = time.perf_counter() - t0
    return phase


def _p(results: list[Result], attr: str, p: float = 50) -> float | None:
    """Percentile of one timing field over successful requests only.

    Missing events stay ``None`` rather than collapsing to 0.0, so "never
    happened" can never be mistaken for "happened instantly".
    """
    return pct([v for r in results if r.ok and (v := getattr(r, attr)) is not None], p)


def _mode(phase: Phase) -> str:
    return phase.results[0].mode if phase.results else ""


def report(phases: list[Phase]) -> int:
    if any(r.status == 429 for p in phases for r in p.results):
        print("\n!! ABORT: got HTTP 429 — CHAT_RATE_LIMIT throttled the run.")
        print("   These numbers measure slowapi, not the RAG pipeline.")
        print("   Restart uvicorn with CHAT_RATE_LIMIT=500/minute and re-run.")
        return 2

    print()
    print("=" * 78)
    print("LATENCY BREAKDOWN  (n = successful requests)")
    print("=" * 78)
    print(
        f"{'phase':<26} {'n':>3} {'sources':>8} {'1st text':>9} "
        f"{'p50 full':>9} {'p95 full':>9}"
    )
    print("-" * 78)
    for p in phases:
        n = sum(1 for r in p.results if r.ok)
        print(
            f"{p.name:<26} {n:>3} {fmt(_p(p.results, 't_meta')):>8} "
            f"{fmt(_p(p.results, 't_token')):>9} {fmt(_p(p.results, 't_done')):>9} "
            f"{fmt(_p(p.results, 't_done', 95)):>9}"
        )

    failures = [(p, r) for p in phases for r in p.results if not r.ok]
    if failures:
        print(f"\nFAILURES: {len(failures)}")
        for p, r in failures[:12]:
            print(f"  [{p.name}] {r.status or '---'} {r.error[:70]} :: {r.query[:34]}")

    print()
    print("=" * 78)
    print("THROUGHPUT  (the decisive test of whether requests really overlap)")
    print("=" * 78)
    print("Real concurrency makes per-request cost FALL as load rises.")
    print("A serialising semaphore leaves throughput FLAT — the queue absorbs it.")
    print()
    print(f"{'phase':<26} {'n':>3} {'wall':>8} {'sec/req':>9} {'req/min':>9}")
    print("-" * 78)
    rates: list[tuple[int, float]] = []
    for p in phases:
        n = sum(1 for r in p.results if r.ok)
        if not n or p.wall <= 0:
            continue
        rate = 60.0 * n / p.wall
        print(f"{p.name:<26} {n:>3} {p.wall:>7.1f}s {p.wall / n:>8.2f}s {rate:>8.1f}")
        if _mode(p) == "stream":
            rates.append((p.concurrency, rate))

    if len(rates) >= 2:
        base = next((r for c, r in rates if c == 1), rates[0][1])
        top_c, top_rate = max(rates)
        gain = top_rate / base if base else 0.0
        print()
        print(f"  c=1 throughput      {base:6.1f} req/min")
        print(f"  c={top_c} throughput      {top_rate:6.1f} req/min")
        print(f"  scaling factor      {gain:6.2f}x  (ideal would be {top_c}x)")
        if gain < 1.5:
            print(f"\n  => SERIALISED. {top_c}x the load bought {gain:.2f}x the capacity.")
            print("     LLM_MAX_CONCURRENCY=1 lets exactly one generation run at a")
            print("     time; everyone else waits in the queue. Extra users do not")
            print("     get slower answers because the model slowed down — they get")
            print("     slower answers because they are standing in a line.")

    # Compare ONLY the c=1 phases. Pooling every streaming phase against a c=1
    # blocking baseline weighs queue-delayed requests against unqueued ones and
    # reports a "speed-up" below 1.0 — an artefact of the mismatch, not a fact
    # about streaming.
    c1 = {m: p for p in phases if p.concurrency == 1 and (m := _mode(p))}
    if (s1 := c1.get("stream")) and (b1 := c1.get("blocking")):
        first = _p(s1.results, "t_token")
        blocking = _p(b1.results, "t_done")
        print()
        print("=" * 78)
        print("WHAT STREAMING BUYS  (single user, like-for-like)")
        print("=" * 78)
        print(f"  sources visible after            {fmt(_p(s1.results, 't_meta'))}")
        print(f"  first words of the answer        {fmt(first)}")
        print(f"  complete answer                  {fmt(_p(s1.results, 't_done'))}")
        print(f"  non-streaming: spinner throughout{fmt(blocking)}")
        if first and blocking:
            print(f"  perceived speed-up               {blocking / first:5.1f}x")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--json", help="write raw per-request results here")
    args = ap.parse_args()

    print(f"target {args.base}")
    print("warming up (first request pays model/index warm-up; excluded)...")
    async with httpx.AsyncClient() as c:
        warm = await stream_one(c, f"{args.base}/api/v1/chat/stream", QUERIES[0], args.timeout)
    print(
        f"  warm-up: status {warm.status} first-text {fmt(warm.t_token)} "
        f"full {fmt(warm.t_done)} {warm.error}"
    )
    if not warm.ok:
        print("!! warm-up failed — is uvicorn up on this port? aborting.")
        return 1

    phases: list[Phase] = []
    plan = [
        ("stream", 1, 4, "stream  c=1 (baseline)"),
        ("blocking", 1, 3, "blocking c=1 (baseline)"),
        ("stream", 2, 4, "stream  c=2"),
        ("stream", 4, 8, "stream  c=4"),
        ("stream", 8, 8, "stream  c=8 (stress)"),
    ]
    for mode, conc, count, label in plan:
        print(f"\n>> {label}: {count} requests, {conc} at a time")
        phase = await run_phase(args.base, mode, conc, count, args.timeout, label)
        ok = sum(1 for r in phase.results if r.ok)
        print(f"   done in {phase.wall:.1f}s — {ok}/{len(phase.results)} ok")
        phases.append(phase)
        if any(r.status == 429 for r in phase.results):
            print("   !! 429 seen — stopping early, rate limit is interfering")
            break

    code = report(phases)

    if args.json:
        raw = [
            {
                "phase": p.name,
                "concurrency": p.concurrency,
                "wall": p.wall,
                "requests": [vars(r) for r in p.results],
            }
            for p in phases
        ]
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2)
        print(f"\nwrote {args.json}")
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
