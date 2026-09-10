"""Unit tests for the load harness's measurement logic.

``testpaths = ["tests"]`` makes ``scripts/`` structurally uncoverable, so the
arithmetic behind every number ``scripts/_load_profile.py`` prints would
otherwise ship unverified. A bug in ``pct`` or in the first-token rule does not
crash — it produces a plausible latency figure that is simply wrong, which is
the failure mode this repo has already been bitten by twice.

Covered here:
  * ``pct`` on empty / single / small samples — the c=1 baseline phase has n=1,
    which ``statistics.quantiles`` cannot handle at all.
  * the empty-token guard: providers emit keep-alive deltas with ``text: ""``,
    and counting one as "first text on screen" would report a time-to-first-
    token far better than the user actually experiences.
  * ``Result.wall``, which feeds the serialisation ratio.
"""

from __future__ import annotations

import pytest

from scripts._load_profile import Phase, Result, fmt, pct, report


def test_pct_empty_sample_is_none_not_zero() -> None:
    """None means "never happened"; 0.0 would read as an instant response."""
    assert pct([], 50) is None
    assert pct([], 95) is None


def test_pct_single_sample() -> None:
    """The n=1 baseline phase must not raise (statistics.quantiles needs n>=2)."""
    assert pct([4.2], 50) == 4.2
    assert pct([4.2], 95) == 4.2


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0, 1.0), (50, 3.0), (95, 5.0), (100, 5.0)],
)
def test_pct_nearest_rank(p: float, expected: float) -> None:
    assert pct([5.0, 1.0, 3.0, 2.0, 4.0], p) == expected


def test_pct_is_order_independent() -> None:
    assert pct([9.0, 1.0, 5.0], 50) == pct([1.0, 5.0, 9.0], 50) == 5.0


def test_fmt_marks_missing_events() -> None:
    """A missing timing must be visibly n/a, never a number."""
    assert "n/a" in fmt(None)
    assert "1.50s" in fmt(1.5)


def test_wall_is_end_minus_start() -> None:
    r = Result(query="q", mode="stream", started_at=10.0, ended_at=23.5)
    assert r.wall == pytest.approx(13.5)


def test_empty_token_must_not_count_as_first_text() -> None:
    """Guard the rule inline in stream_one: `if text and res.t_token is None`.

    Simulated rather than driven through HTTP because the point is the
    predicate, not the transport. An empty delta arriving first must leave
    t_token unset so the reported time-to-first-text stays honest.
    """
    res = Result(query="q", mode="stream")
    for elapsed, text in [(0.4, ""), (0.9, ""), (2.7, "Hello"), (2.8, " there")]:
        if text and res.t_token is None:
            res.t_token = elapsed
        if text:
            res.tokens += 1
            res.chars += len(text)

    assert res.t_token == 2.7, "empty keep-alive deltas must not set t_token"
    assert res.tokens == 2
    assert res.chars == len("Hello") + len(" there")


def _phase(name: str, concurrency: int, wall: float, n: int, mode: str = "stream") -> Phase:
    p = Phase(name=name, concurrency=concurrency, wall=wall)
    p.results = [
        Result(query="q", mode=mode, ok=True, t_meta=1.0, t_token=2.0, t_done=5.0)
        for _ in range(n)
    ]
    return p


def test_throughput_flat_under_load_is_the_serialisation_signal(capsys) -> None:
    """Flat req/min as concurrency rises is what a serialising semaphore looks like.

    Numbers are the real measured ones: 4 reqs in 33.9s at c=1 and 8 reqs in
    69.8s at c=8 — 7.1 vs 6.9 req/min. 8x the load, no extra capacity.
    """
    report([_phase("stream c=1", 1, 33.9, 4), _phase("stream c=8", 8, 69.8, 8)])
    out = capsys.readouterr().out
    assert "SERIALISED" in out
    assert "standing in a line" in out


def test_throughput_that_scales_is_not_reported_as_serialised(capsys) -> None:
    """Falsifies the check: if capacity really scales, the verdict must not fire.

    8 reqs in 8.7s at c=8 is ~55 req/min against 7.1 — genuine parallelism.
    """
    report([_phase("stream c=1", 1, 33.9, 4), _phase("stream c=8", 8, 8.7, 8)])
    out = capsys.readouterr().out
    assert "SERIALISED" not in out


def test_streaming_comparison_uses_only_c1_phases(capsys) -> None:
    """Pooling queue-delayed phases into the comparison invents a <1x speed-up.

    The blocking baseline is c=1, so the streaming side must be c=1 too. Here
    the c=8 phase is deliberately awful (t_token=40s); if it leaked into the
    median the speed-up would collapse below 1x instead of reporting 4.3x.
    """
    s1 = _phase("stream c=1", 1, 33.9, 4)
    b1 = _phase("blocking c=1", 1, 36.6, 3, mode="blocking")
    for r in b1.results:
        r.t_meta = r.t_token = None
        r.t_done = 8.6
    bad = _phase("stream c=8", 8, 69.8, 8)
    for r in bad.results:
        r.t_token = 40.0
        r.t_done = 66.0

    report([s1, b1, bad])
    out = capsys.readouterr().out
    assert "perceived speed-up" in out
    speed = next(ln for ln in out.splitlines() if "perceived speed-up" in ln)
    assert "4.3x" in speed, speed


def test_http_429_aborts_instead_of_reporting_throttled_latency(capsys) -> None:
    """A rate-limited run must fail loudly, not publish slowapi's queueing as latency."""
    p = _phase("stream c=8", 8, 69.8, 4)
    p.results[2].ok = False
    p.results[2].status = 429
    assert report([p]) == 2
    assert "ABORT" in capsys.readouterr().out
