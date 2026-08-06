"""Concurrent-user pressure test against the live /chat endpoint.

Ramps virtual users (10, 20, 30, 50 by default) and reports the latency
distribution, error taxonomy, and throughput at each level, so the point where
the service stops meeting the <10s objective is visible rather than inferred.

What this measures that scripts/bench_e2e.py does not
-----------------------------------------------------
bench_e2e runs queries *serially, in-process*: it is a clean measurement of one
request's cost with nothing contending for the reranker, the DB pool, or the
LLM provider's own concurrency budget. That is the right tool for "is a single
query fast enough" and the wrong one for "what happens at 30 users", because
every shared resource in this app is only a bottleneck under contention:

  * **DB pool** — db_pool_size(5) + db_max_overflow(10) = 15 concurrent
    connections, then requests block for up to db_pool_timeout(30s). Above 15
    users, queue time appears in total latency but in no single stage timer.
  * **Cross-encoder** — one process-wide model, scored via anyio.to_thread, so
    it serialises on the default thread-limiter capacity (40).
  * **LLM provider** — has its own rate limit and concurrency ceiling. This is
    usually the first thing to break, and it breaks as 429s, not slowness.

Because these bottlenecks are *queues*, mean latency hides them: throughput can
look fine while the slowest decile is timing out. Everything here is reported
as a distribution, and errors are broken out by cause rather than counted.

Rate limiting
-------------
/chat is limited per-IP (CHAT_RATE_LIMIT, default 20/minute). A load test from
one machine trips this within seconds, after which the run measures slowapi
rather than the server. Raise it for the test run and RESTART the server —
slowapi's decorator captures the value at import time:

    # .env
    CHAT_RATE_LIMIT=100000/minute
    RATE_LIMIT=100000/minute

The script pre-flights this and refuses to start a ramp that would be
meaningless; 429s during a run are counted separately and flagged loudly.

Cache
-----
Each virtual user draws from a pool of distinct queries and appends a unique
suffix per request by default, so the retrieval cache (TTL 300s) does not turn
level 2 into a measurement of dict lookups. Pass --cache-realistic for the
opposite: a realistic hot-cache mix where users repeat popular questions.

Usage
-----
    ./.venv/Scripts/python.exe -u scripts/loadtest.py
    ./.venv/Scripts/python.exe -u scripts/loadtest.py --levels 10,20,30,50,100
    ./.venv/Scripts/python.exe -u scripts/loadtest.py --requests-per-user 5
    ./.venv/Scripts/python.exe -u scripts/loadtest.py --cache-realistic
    ./.venv/Scripts/python.exe -u scripts/loadtest.py --json results.json
    ./.venv/Scripts/python.exe -u scripts/loadtest.py --url http://staging:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Literal

try:
    import httpx
except ImportError:  # pragma: no cover - dependency is in requirements-backend
    print("httpx is required:  uv pip install httpx", file=sys.stderr)
    raise SystemExit(2) from None


# Realistic help-desk traffic: a mix of covered topics, typos, conversational
# phrasing and one off-topic query. Drawn from the labeled eval set so the
# answers are known-good and a quality regression under load is visible as a
# change in the decline rate rather than as silence.
QUERIES: list[str] = [
    "How do I reset my student portal password?",
    "What is SMOWL proctoring?",
    "How do I set up Microsoft Authenticator?",
    "How do I log in to the LMS?",
    "How do I access my student email?",
    "How do I register for supplementary exams?",
    "How do I use Microsoft Teams?",
    "What is My Loft?",
    "How do I contact the AmIU help desk?",
    "moddle login",
    "smwol camera not working",
    "athenticator app setup",
    "2FA setup",
    "I cannot access my assignments",
    "my webcam isn't being detected during the online exam",
    "trying to check my university email but it won't let me in",
    "password",
    "exam registration",
    "What is the capital of France?",
]

# Latency objective from the project brief. Reported per level as pass/fail on
# p95 — a mean under budget with a p95 over it means most users are fine and a
# meaningful minority are not, which is the failure the mean is built to hide.
TARGET_S = 10.0

ErrorKind = Literal["rate_limited", "timeout", "server_error", "connection", "other"]


@dataclass
class Result:
    """One request's outcome."""

    ok: bool
    latency_s: float
    status: int | None = None
    error_kind: ErrorKind | None = None
    error_detail: str = ""
    confidence: float = 0.0
    n_sources: int = 0
    answer_chars: int = 0
    declined: bool = False


@dataclass
class LevelReport:
    """Aggregated outcome for one concurrency level."""

    users: int
    requests: int
    wall_s: float
    results: list[Result] = field(default_factory=list)

    @property
    def ok_results(self) -> list[Result]:
        return [r for r in self.results if r.ok]

    @property
    def latencies(self) -> list[float]:
        return [r.latency_s for r in self.ok_results]

    @property
    def error_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            if not r.ok and r.error_kind:
                counts[r.error_kind] = counts.get(r.error_kind, 0) + 1
        return counts

    @property
    def error_rate(self) -> float:
        return 1.0 - (len(self.ok_results) / len(self.results)) if self.results else 0.0

    @property
    def throughput(self) -> float:
        """Completed *successful* requests per second over the level's wall time."""
        return len(self.ok_results) / self.wall_s if self.wall_s > 0 else 0.0


def pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile.

    Not interpolated: at these sample sizes an interpolated p99 invents a value
    between two observations, and the whole point of p99 here is that it is a
    real request someone actually waited for.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[idx]


def classify_error(exc: BaseException | None, status: int | None) -> tuple[ErrorKind, str]:
    """Map a failure to a cause.

    The distinction matters for interpretation: a 429 means the limiter is still
    on and the run is invalid; a timeout means a real queue somewhere; a 500
    means something threw. Collapsing these into one "errors" count is how a
    misconfigured test gets reported as a capacity finding.
    """
    if status == 429:
        return "rate_limited", "rate limited (429)"
    if status is not None and status >= 500:
        return "server_error", f"HTTP {status}"
    if status is not None and status >= 400:
        return "other", f"HTTP {status}"
    if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout)):
        return "timeout", type(exc).__name__
    if isinstance(exc, httpx.ConnectError):
        return "connection", "connection refused / reset"
    return "other", f"{type(exc).__name__}: {exc}"


DECLINE_MARKERS = ("isn't covered", "is not covered", "outside", "not able to", "don't have")


async def one_request(
    client: httpx.AsyncClient, url: str, message: str, timeout_s: float
) -> Result:
    """Issue one /chat request and time it end to end."""
    t0 = time.perf_counter()
    try:
        response = await client.post(
            url,
            json={"message": message},
            timeout=timeout_s,
        )
        latency = time.perf_counter() - t0

        if response.status_code != 200:
            kind, detail = classify_error(None, response.status_code)
            return Result(
                ok=False,
                latency_s=latency,
                status=response.status_code,
                error_kind=kind,
                error_detail=detail,
            )

        body = response.json()
        answer = body.get("answer", "") or ""
        return Result(
            ok=True,
            latency_s=latency,
            status=200,
            confidence=float(body.get("confidence", 0.0)),
            n_sources=len(body.get("sources", []) or []),
            answer_chars=len(answer),
            declined=any(m in answer.lower() for m in DECLINE_MARKERS),
        )
    except BaseException as exc:  # noqa: BLE001 - every failure is a data point
        latency = time.perf_counter() - t0
        kind, detail = classify_error(exc, None)
        return Result(
            ok=False, latency_s=latency, error_kind=kind, error_detail=detail
        )


async def run_level(
    base_url: str,
    users: int,
    requests_per_user: int,
    timeout_s: float,
    unique_queries: bool,
    level_index: int,
) -> LevelReport:
    """Run one concurrency level: `users` virtual users, N requests each.

    All users start together at a barrier rather than being spawned in a loop.
    A staggered start would let the first users finish before the last begin,
    which measures a lower effective concurrency than the label claims.
    """
    url = f"{base_url.rstrip('/')}/api/v1/chat"
    results: list[Result] = []

    # One connection pool, sized to the level, so the client is never the
    # bottleneck being measured. httpx defaults to 100 max connections but only
    # 20 keepalive, which would itself queue at higher levels.
    limits = httpx.Limits(max_connections=users + 10, max_keepalive_connections=users + 10)

    start_barrier = asyncio.Barrier(users)

    async with httpx.AsyncClient(limits=limits) as client:

        async def virtual_user(user_id: int) -> list[Result]:
            await start_barrier.wait()
            out: list[Result] = []
            for req_index in range(requests_per_user):
                # Spread users across the query pool so they are not all hitting
                # the same cache entry, and offset by level so level N+1 does not
                # inherit level N's warm cache for free.
                q_index = (user_id * 3 + req_index + level_index * 7) % len(QUERIES)
                message = QUERIES[q_index]
                if unique_queries:
                    # Defeat the 300s retrieval cache. Appended as a distinct
                    # clause rather than random noise so the query stays
                    # well-formed and retrieval still has to do real work.
                    message = f"{message} (ref {level_index}-{user_id}-{req_index})"
                out.append(await one_request(client, url, message, timeout_s))
            return out

        t0 = time.perf_counter()
        gathered = await asyncio.gather(
            *[virtual_user(u) for u in range(users)], return_exceptions=True
        )
        wall = time.perf_counter() - t0

    for item in gathered:
        if isinstance(item, BaseException):
            results.append(
                Result(ok=False, latency_s=0.0, error_kind="other", error_detail=repr(item))
            )
        else:
            results.extend(item)

    return LevelReport(users=users, requests=len(results), wall_s=wall, results=results)


async def preflight(base_url: str, timeout_s: float) -> bool:
    """Verify the server is up and the rate limiter will not invalidate the run.

    Sends a small burst and checks for 429s. Running a full ramp against a
    limiter set to 20/minute produces a chart of the limiter, so this refuses
    rather than letting it happen quietly.
    """
    print("pre-flight")
    health_url = f"{base_url.rstrip('/')}/health"
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(health_url, timeout=10.0)
            r.raise_for_status()
            print(f"  [OK]   server healthy at {base_url}")
        except BaseException as exc:  # noqa: BLE001
            print(f"  [FAIL] server not reachable at {health_url}: {exc}")
            print("\n  Start it with:")
            print("    ./.venv/Scripts/python.exe -m uvicorn backend.app.main:app --port 8000")
            return False

        # 25 rapid requests: above the 20/minute default, below anything sane
        # for a load test. If none are refused, the limit has been raised.
        chat_url = f"{base_url.rstrip('/')}/api/v1/chat"
        probes = await asyncio.gather(
            *[
                client.post(chat_url, json={"message": f"ping {i}"}, timeout=timeout_s)
                for i in range(25)
            ],
            return_exceptions=True,
        )
        n_429 = sum(
            1
            for p in probes
            if isinstance(p, httpx.Response) and p.status_code == 429
        )
        if n_429:
            print(f"  [FAIL] {n_429}/25 probe requests were rate limited (429).")
            print()
            print("  The ramp would measure the rate limiter, not the server.")
            print("  Set these in .env and RESTART the server (slowapi reads the")
            print("  limit at import time, so a reload is not enough):")
            print()
            print("    CHAT_RATE_LIMIT=100000/minute")
            print("    RATE_LIMIT=100000/minute")
            return False
        print("  [OK]   rate limiter raised (25-request burst, no 429s)")

    # Warm the server's singletons. The first real request after a cold start
    # pays cross-encoder and HNSW load costs that belong to startup, not to the
    # concurrency level that happens to run first.
    print("  ...    warming server (1 request)")
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        await one_request(client, chat_url, QUERIES[0], timeout_s)
        print(f"  [OK]   warm request completed in {time.perf_counter() - t0:.2f}s")
    return True


def print_level(report: LevelReport) -> None:
    """Print one level's results."""
    lat = report.latencies
    p95 = pct(lat, 95)
    verdict = "PASS" if lat and p95 <= TARGET_S and report.error_rate == 0 else "FAIL"

    print()
    print(f"--- {report.users} concurrent users " + "-" * 46)
    print(
        f"  requests {report.requests}  |  ok {len(report.ok_results)}  |  "
        f"errors {report.requests - len(report.ok_results)} "
        f"({report.error_rate * 100:.1f}%)  |  wall {report.wall_s:.1f}s"
    )

    if lat:
        print(
            f"  latency  mean {statistics.mean(lat):5.2f}s  "
            f"p50 {pct(lat, 50):5.2f}s  p95 {p95:5.2f}s  "
            f"p99 {pct(lat, 99):5.2f}s  max {max(lat):5.2f}s"
        )
        print(
            f"  throughput {report.throughput:.2f} req/s  |  "
            f"target p95 <{TARGET_S:.0f}s -> {verdict}"
        )
        # Quality under load. If answers get shorter or confidence drops as
        # concurrency rises, something is degrading that latency alone misses.
        declined = sum(1 for r in report.ok_results if r.declined)
        mean_conf = statistics.mean([r.confidence for r in report.ok_results])
        mean_chars = statistics.mean([r.answer_chars for r in report.ok_results])
        print(
            f"  answers  mean confidence {mean_conf:.3f}  |  "
            f"mean length {mean_chars:.0f} chars  |  declined {declined}"
        )
    else:
        print("  latency  no successful requests")

    if report.error_counts:
        print("  errors by cause:")
        for kind, count in sorted(report.error_counts.items(), key=lambda kv: -kv[1]):
            sample = next(
                (r.error_detail for r in report.results if r.error_kind == kind), ""
            )
            print(f"    {kind:<14} {count:>4}   e.g. {sample}")


def print_summary(reports: list[LevelReport], db_ceiling: int | None) -> None:
    """Print the cross-level comparison table and the interpretation."""
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"{'users':>6} {'ok':>6} {'err%':>7} {'p50':>8} {'p95':>8} "
        f"{'p99':>8} {'max':>8} {'req/s':>8}  verdict"
    )
    for r in reports:
        lat = r.latencies
        if not lat:
            print(f"{r.users:>6} {0:>6} {r.error_rate * 100:>6.1f}% " + " " * 34 + "  NO DATA")
            continue
        p95 = pct(lat, 95)
        verdict = "PASS" if p95 <= TARGET_S and r.error_rate == 0 else "FAIL"
        print(
            f"{r.users:>6} {len(r.ok_results):>6} {r.error_rate * 100:>6.1f}% "
            f"{pct(lat, 50):>7.2f}s {p95:>7.2f}s {pct(lat, 99):>7.2f}s "
            f"{max(lat):>7.2f}s {r.throughput:>7.2f}  {verdict}"
        )

    print()
    print("interpretation")

    # Saturation: the level where throughput stops rising. Past this point extra
    # users only add queue time, which is the practical capacity ceiling.
    best = max(reports, key=lambda r: r.throughput, default=None)
    if best and best.throughput > 0:
        print(
            f"  peak throughput {best.throughput:.2f} req/s at {best.users} users"
        )
        later = [r for r in reports if r.users > best.users]
        if later and all(r.throughput <= best.throughput for r in later):
            print(
                f"  throughput does not improve past {best.users} users — "
                "added load becomes queue time, not work"
            )

    # First level to miss the objective.
    failed = [
        r for r in reports if r.latencies and pct(r.latencies, 95) > TARGET_S
    ]
    if failed:
        first = failed[0]
        print(
            f"  p95 first exceeds {TARGET_S:.0f}s at {first.users} users "
            f"({pct(first.latencies, 95):.2f}s)"
        )
    else:
        print(f"  p95 stayed within {TARGET_S:.0f}s at every level tested")

    # Attribute errors, since the causes have completely different fixes.
    if any(r.error_counts for r in reports):
        print()
        print("  error causes seen:")
        if any("rate_limited" in r.error_counts for r in reports):
            print(
                "    rate_limited — limiter still active; results INVALID. "
                "Raise CHAT_RATE_LIMIT and restart."
            )
        if any("timeout" in r.error_counts for r in reports):
            print(
                "    timeout      — requests queued past the client timeout. Usual "
                "causes: DB pool exhaustion, LLM provider concurrency limit."
            )
        if any("server_error" in r.error_counts for r in reports):
            print(
                "    server_error — 5xx from the app. Check logs/app.log; likely "
                "a pool timeout surfacing as an unhandled exception."
            )
        if any("connection" in r.error_counts for r in reports):
            print(
                "    connection   — the server refused or reset. Uvicorn's backlog "
                "is saturated, or the process died."
            )

    if db_ceiling is not None:
        crossing = [r for r in reports if r.users > db_ceiling]
        if crossing:
            print()
            print(
                f"  note: the DB pool allows {db_ceiling} concurrent connections "
                f"(db_pool_size + db_max_overflow)."
            )
            print(
                f"        levels above {db_ceiling} ({', '.join(str(r.users) for r in crossing)}) "
                "queue for a connection; that wait"
            )
            print(
                "        appears in total latency but in no per-stage timer. Raise "
                "DB_POOL_SIZE / DB_MAX_OVERFLOW to move it."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concurrent-user pressure test for the /chat endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Base server URL")
    parser.add_argument(
        "--levels",
        default="10,20,30,50",
        help="Comma-separated concurrency levels (default: 10,20,30,50)",
    )
    parser.add_argument(
        "--requests-per-user",
        type=int,
        default=3,
        help="Requests each virtual user sends (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--cache-realistic",
        action="store_true",
        help="Let users repeat queries so the retrieval cache can hit "
        "(default: every request is unique, worst case)",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=5.0,
        help="Seconds to idle between levels so queues drain (default: 5)",
    )
    parser.add_argument("--json", dest="json_out", help="Write raw results to this JSON file")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the health and rate-limit checks (not recommended)",
    )
    args = parser.parse_args()

    try:
        levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    except ValueError:
        print(f"invalid --levels: {args.levels!r}", file=sys.stderr)
        return 2
    if not levels:
        print("no levels given", file=sys.stderr)
        return 2

    # Read the pool ceiling so the report can attribute queueing to it. Best
    # effort: the script must still run when it cannot import app config (e.g.
    # pointed at a remote server from a different checkout).
    db_ceiling: int | None = None
    try:
        sys.path.insert(0, ".")
        from backend.app.config import get_settings

        s = get_settings()
        db_ceiling = s.db_pool_size + s.db_max_overflow
    except Exception:  # noqa: BLE001
        pass

    total_requests = sum(levels) * args.requests_per_user
    print("=" * 78)
    print("CONCURRENT USER PRESSURE TEST")
    print("=" * 78)
    print(f"  target        : {args.url}")
    print(f"  levels        : {', '.join(str(x) for x in levels)} concurrent users")
    print(f"  per user      : {args.requests_per_user} requests")
    print(f"  total         : {total_requests} requests (= {total_requests} LLM calls)")
    print(f"  cache         : {'realistic (repeats allowed)' if args.cache_realistic else 'defeated (every query unique)'}")
    print(f"  timeout       : {args.timeout:.0f}s per request")
    if db_ceiling:
        print(f"  db pool       : {db_ceiling} concurrent connections")
    print()

    async def run_all() -> list[LevelReport]:
        if not args.skip_preflight:
            if not await preflight(args.url, args.timeout):
                raise SystemExit(1)

        reports: list[LevelReport] = []
        for i, users in enumerate(levels):
            print()
            print(f"running level {i + 1}/{len(levels)}: {users} concurrent users...")
            report = await run_level(
                base_url=args.url,
                users=users,
                requests_per_user=args.requests_per_user,
                timeout_s=args.timeout,
                unique_queries=not args.cache_realistic,
                level_index=i,
            )
            print_level(report)
            reports.append(report)

            # Let queues drain so the next level starts from a quiet server
            # rather than inheriting the previous level's backlog.
            if i < len(levels) - 1 and args.settle > 0:
                await asyncio.sleep(args.settle)
        return reports

    try:
        reports = asyncio.run(run_all())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    print_summary(reports, db_ceiling)

    if args.json_out:
        payload: dict[str, Any] = {
            "url": args.url,
            "levels": levels,
            "requests_per_user": args.requests_per_user,
            "cache_realistic": args.cache_realistic,
            "db_ceiling": db_ceiling,
            "results": [
                {
                    "users": r.users,
                    "requests": r.requests,
                    "ok": len(r.ok_results),
                    "error_rate": r.error_rate,
                    "errors_by_cause": r.error_counts,
                    "wall_s": r.wall_s,
                    "throughput_rps": r.throughput,
                    "latency": {
                        "mean": statistics.mean(r.latencies) if r.latencies else None,
                        "p50": pct(r.latencies, 50),
                        "p95": pct(r.latencies, 95),
                        "p99": pct(r.latencies, 99),
                        "max": max(r.latencies) if r.latencies else None,
                    },
                    "latencies_s": r.latencies,
                }
                for r in reports
            ],
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nraw results written to {args.json_out}")

    # Non-zero exit when any level missed the objective, so this can gate CI.
    any_fail = any(
        not r.latencies or pct(r.latencies, 95) > TARGET_S or r.error_rate > 0
        for r in reports
    )
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
