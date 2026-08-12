"""Regression tests for the database connection lifecycle.

These exist because of a specific, measured production failure. Under a 20-user
load test the service produced exactly 15 successes and 5 failures, every failure
landing at ~30.5s — ``db_pool_timeout``. The cause was not pool size. The chat
path took its ``AsyncSession`` from a FastAPI ``yield`` dependency, and FastAPI
unwinds those only *after* the response is sent, so one pooled connection stayed
checked out for the whole request: through embedding, Chroma, BM25, the
cross-encoder, and the multi-second NVIDIA NIM call. With
``pool_size(5) + max_overflow(10) = 15`` the service could not exceed 15
in-flight requests no matter how idle the CPU was.

Each test below pins one property of the fix, phrased so that reintroducing the
old design fails it:

1. ``test_connection_released_before_llm``      — nothing is checked out during generation.
2. ``test_llm_failure_does_not_leak``           — a raising LLM leaks no connection.
3. ``test_repeated_requests_do_not_accumulate`` — checkout count returns to baseline.
4. ``test_concurrency_above_pool_ceiling``      — 20+ concurrent requests, zero pool timeouts.

Tests 1 and 2 are pure unit tests over an instrumented scope and need no
database. Tests 3 and 4 exercise the real engine and pool — that is the whole
point of them — and skip when PostgreSQL is not reachable.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from sqlalchemy import delete, select, text

from backend.app.database.models import AnalyticsLog, ChatMessage, Session
from backend.app.database.session import db_scope, engine, pool_stats
from backend.app.rag.retriever import RetrievedChunk, RetrievedImage
from backend.app.services.rag_service import RAGService
from backend.app.utils.exceptions import DatabaseUnavailableError


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


@dataclass
class ScopeTracker:
    """An instrumented stand-in for ``db_scope`` that records its own lifetime.

    ``open_labels`` is the assertion surface: if it is non-empty at a given
    moment, a connection would have been checked out at that moment under the
    real scope.
    """

    open_labels: list[str] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    max_concurrent: int = 0

    @asynccontextmanager
    async def scope(self, label: str = "db"):
        self.open_labels.append(label)
        self.history.append(("enter", label))
        self.max_concurrent = max(self.max_concurrent, len(self.open_labels))
        try:
            yield _FakeSession()
        finally:
            self.open_labels.remove(label)
            self.history.append(("exit", label))


class _FakeSession:
    """Minimal AsyncSession surface used by ``RAGService``."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    async def execute(self, *_args: Any, **_kwargs: Any) -> "_FakeResult":
        return _FakeResult()

    def add(self, obj: Any) -> None:
        # Mirror the Python-side PK default so ``str(obj.id)`` works after flush,
        # exactly as it does for the real models.
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()
        self.added.append(obj)

    async def flush(self) -> None:
        return None


class _FakeResult:
    def scalar_one_or_none(self) -> None:
        return None

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[Any]:
        return []


class FakeRetriever:
    """Retrieval with the timing characteristics that matter and none of the cost.

    ``hydrate_results`` is the only method that touches the database, matching
    the real retriever: Chroma, BM25 and the cross-encoder never do.
    """

    def __init__(self) -> None:
        self.hydrate_calls = 0

    async def embed_query(self, _message: str) -> list[float]:
        await asyncio.sleep(0)
        return [0.0, 1.0]

    async def retrieve(self, _message: str, **_kwargs: Any):
        # Non-zero so the event loop actually interleaves under concurrency.
        await asyncio.sleep(0.01)
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="content",
            article_id="a1",
            category=None,
            chunk_index=0,
            score=0.9,
        )
        return [chunk], [], {"intent": "test"}

    async def hydrate_results(self, _db: Any, chunks: list[Any], images: list[Any]):
        self.hydrate_calls += 1
        return chunks, images

    def format_context(self, _chunks: list[Any]) -> str:
        return "CONTEXT"

    def format_images(self, _images: list[Any]) -> str:
        return ""

    def compute_confidence(self, _chunks: list[Any], _processed: Any) -> float:
        return 0.5


class ProbingLLM:
    """LLM double that samples connection state at the moment of generation.

    ``probe`` is called while "the LLM is running". Whatever it returns is what
    the test asserts on, which keeps the pool-vs-tracker distinction out of the
    double itself.
    """

    def __init__(self, probe, *, delay: float = 0.0, raises: BaseException | None = None) -> None:
        self._probe = probe
        self._delay = delay
        self._raises = raises
        self.samples: list[Any] = []

    async def generate_answer(self, **_kwargs: Any) -> str:
        self.samples.append(self._probe())
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises is not None:
            raise self._raises
        return "an answer"

    async def stream_answer(self, **_kwargs: Any):
        self.samples.append(self._probe())
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises is not None:
            raise self._raises
        for token in ("an ", "answer"):
            yield token


# ---------------------------------------------------------------------------
# 1 & 2 — lifecycle, no database required
# ---------------------------------------------------------------------------


async def test_connection_released_before_llm() -> None:
    """No scope is open while the LLM generates.

    This is the assertion the old design fails. It held one session from the
    first query to the end of the response, so a probe taken during generation
    would have seen it open.
    """
    tracker = ScopeTracker()
    llm = ProbingLLM(probe=lambda: list(tracker.open_labels))
    service = RAGService(
        retriever=FakeRetriever(), llm_service=llm, session_scope=tracker.scope
    )

    response = await service.chat("How do I reset my password?")

    assert response.answer == "an answer"
    assert llm.samples == [[]], (
        f"a DB scope was open during generation: {llm.samples[0]} — "
        "the connection must be released before the NIM call"
    )
    # Sanity: the test would also pass if no DB work happened at all.
    labels = [label for action, label in tracker.history if action == "enter"]
    assert labels == ["chat_prepare", "chat_hydrate", "chat_persist"]
    # Each scope closed before the next opened — no nesting, no overlap.
    assert tracker.max_concurrent == 1
    assert tracker.open_labels == []


async def test_streaming_holds_no_connection_across_tokens() -> None:
    """The streaming path is the worst case for the old design, so pin it too.

    There, the connection stayed checked out not merely for generation but until
    the client finished consuming the response.
    """
    tracker = ScopeTracker()
    llm = ProbingLLM(probe=lambda: list(tracker.open_labels))
    service = RAGService(
        retriever=FakeRetriever(), llm_service=llm, session_scope=tracker.scope
    )

    events = []
    async for event in service.chat_stream("How do I reset my password?"):
        # Assert per token: the scope must be closed for the whole stream, not
        # merely at the start of it.
        if event["type"] == "token":
            assert tracker.open_labels == []
        events.append(event)

    assert llm.samples == [[]]
    assert [e["type"] for e in events] == ["meta", "token", "token", "done"]
    assert events[-1]["message_id"]
    assert tracker.open_labels == []


async def test_probe_detects_the_old_request_scoped_design() -> None:
    """Negative control for the test above.

    ``test_connection_released_before_llm`` asserts the probe sees no open
    scope — which it would also do if the probe were simply blind. Here the old
    design is reproduced deliberately (one scope open across the whole request,
    as the FastAPI ``yield`` dependency used to be) and the probe must see it.
    If this test ever fails, the one above has stopped proving anything.
    """
    tracker = ScopeTracker()
    llm = ProbingLLM(probe=lambda: list(tracker.open_labels))
    service = RAGService(
        retriever=FakeRetriever(), llm_service=llm, session_scope=tracker.scope
    )

    async with tracker.scope("request_scoped"):
        await service.chat("How do I reset my password?")

    assert llm.samples == [["request_scoped"]], (
        "the probe cannot observe a request-scoped connection, so "
        "test_connection_released_before_llm passes vacuously"
    )


async def test_llm_failure_does_not_leak() -> None:
    """An exception during generation leaves no scope open."""
    tracker = ScopeTracker()
    llm = ProbingLLM(
        probe=lambda: list(tracker.open_labels),
        raises=RuntimeError("NIM returned 503"),
    )
    service = RAGService(
        retriever=FakeRetriever(), llm_service=llm, session_scope=tracker.scope
    )

    with pytest.raises(RuntimeError, match="NIM returned 503"):
        await service.chat("How do I reset my password?")

    assert tracker.open_labels == [], "a DB scope survived an LLM failure"
    # The failure happened after retrieval and before persistence, so the
    # persist scope must never have opened.
    assert [label for action, label in tracker.history if action == "enter"] == [
        "chat_prepare",
        "chat_hydrate",
    ]


# ---------------------------------------------------------------------------
# 3 & 4 — real engine and pool
# ---------------------------------------------------------------------------


async def _database_reachable() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - any failure means "cannot test this here"
        return False


async def _cleanup_fixture_rows() -> None:
    """Delete rows these tests created, children before parents.

    ``chat_messages.session_id`` and ``feedback.message_id`` are real foreign
    keys, so the session row cannot go first. Assistant messages are matched by
    session rather than by content — their text comes from the LLM double, not
    from the test's prompt, so a content prefix would miss them and leave the
    delete blocked.
    """
    async with db_scope("test_cleanup") as db:
        session_ids = (
            (
                await db.execute(
                    select(Session.id).where(Session.title.like("regression-test-%"))
                )
            )
            .scalars()
            .all()
        )
        if not session_ids:
            return
        await db.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(session_ids)))
        await db.execute(delete(AnalyticsLog).where(AnalyticsLog.session_id.in_(session_ids)))
        await db.execute(delete(Session).where(Session.id.in_(session_ids)))


@pytest.fixture
async def live_db():
    # pytest-asyncio gives each test its own event loop, and an asyncpg
    # connection is bound to the loop that created it. A connection pooled by
    # the previous test therefore raises when this test reuses it, which would
    # show up as a spurious skip. Dropping the pool makes the engine rebuild it
    # on the current loop; the reconnect cost is irrelevant at this scale.
    await engine.dispose()

    if not await _database_reachable():
        pytest.skip("PostgreSQL not reachable — pool tests need the real engine")
    try:
        yield
    finally:
        # These tests write real chat rows. Clean up so a dev database is not
        # gradually filled with fixture traffic. In ``finally`` so a failing
        # assertion does not leave rows behind for the next run to trip over.
        await _cleanup_fixture_rows()


async def test_pool_stats_are_safe_to_expose(live_db: None) -> None:
    """Pool diagnostics carry counts, never credentials."""
    stats = pool_stats()
    assert stats["ceiling"] == stats["pool_size"] + stats["max_overflow"]
    assert set(stats) >= {"pool_size", "max_overflow", "pool_timeout_s", "ceiling"}
    blob = repr(stats).lower()
    for secret in ("password", "postgresql", "asyncpg", "@", "localhost"):
        assert secret not in blob, f"pool_stats leaked {secret!r}"


async def test_db_scope_releases_immediately(live_db: None) -> None:
    """``db_scope`` returns its connection to the pool on block exit."""
    before = pool_stats()["checked_out"]

    async with db_scope("test_probe") as db:
        await db.execute(text("SELECT 1"))
        during = pool_stats()["checked_out"]

    after = pool_stats()["checked_out"]
    assert during == before + 1, "db_scope did not check out a connection"
    assert after == before, f"connection not released: {before} → {after}"


async def test_repeated_requests_do_not_accumulate(live_db: None) -> None:
    """Checkouts return to baseline after repeated requests.

    A leak is cumulative and permanent for the life of the process, so it shows
    up as a checkout count that ratchets upward across requests even though each
    one finished.
    """
    service = RAGService(
        retriever=FakeRetriever(),
        llm_service=ProbingLLM(probe=lambda: pool_stats().get("checked_out")),
    )

    # One request first so any lazily-created connection is already in the pool
    # and does not read as growth.
    await service.chat("regression-test-warm")
    baseline = pool_stats()["checked_out"]

    readings = []
    for i in range(6):
        await service.chat(f"regression-test-{i}")
        readings.append(pool_stats()["checked_out"])

    assert readings == [baseline] * 6, (
        f"checked-out connections drifted from {baseline}: {readings}"
    )


async def test_no_connection_held_during_llm_with_real_pool(live_db: None) -> None:
    """The same proof as test 1, but read off the real pool.

    The unit test asserts the scopes are closed; this asserts the pool agrees.
    """
    samples: list[int] = []
    service = RAGService(
        retriever=FakeRetriever(),
        llm_service=ProbingLLM(
            probe=lambda: samples.append(pool_stats()["checked_out"]), delay=0.05
        ),
    )
    idle = pool_stats()["checked_out"]
    await service.chat("regression-test-probe")

    assert samples == [idle], (
        f"{samples[0] - idle} connection(s) were checked out during generation"
    )


@pytest.mark.parametrize("users", [20, 30])
async def test_concurrency_above_pool_ceiling(live_db: None, users: int) -> None:
    """N concurrent requests above the pool ceiling produce zero pool timeouts.

    This is the regression test for the original symptom. The LLM double sleeps
    500ms — long enough that all N requests overlap in "generation", which is
    precisely the window the old design spent holding a connection. With a
    ceiling of 15, the old code produced 15 successes and N-15 timeouts here;
    the fix must produce N successes.

    Deliberately does NOT raise pool_size: the point is that the existing
    5 + 10 pool is sufficient once connections are held only for DB work.
    """
    ceiling = pool_stats()["ceiling"]
    assert users > ceiling, (
        f"{users} users is not above the pool ceiling ({ceiling}); "
        "this test would prove nothing"
    )

    peak = 0

    def probe() -> None:
        nonlocal peak
        peak = max(peak, pool_stats()["checked_out"])

    service = RAGService(
        retriever=FakeRetriever(),
        llm_service=ProbingLLM(probe=probe, delay=0.5),
    )

    outcomes = await asyncio.gather(
        *[service.chat(f"regression-test-conc-{users}-{i}") for i in range(users)],
        return_exceptions=True,
    )

    pool_timeouts = [o for o in outcomes if isinstance(o, DatabaseUnavailableError)]
    other_errors = [
        o for o in outcomes
        if isinstance(o, BaseException) and not isinstance(o, DatabaseUnavailableError)
    ]
    assert not pool_timeouts, f"{len(pool_timeouts)}/{users} requests exhausted the pool"
    assert not other_errors, f"unexpected failures: {other_errors[:3]}"
    assert len(outcomes) == users

    # The real payoff: peak concurrent checkouts stayed far below the ceiling
    # even though all N requests were in flight at once.
    assert peak <= ceiling, f"peak checkouts {peak} exceeded ceiling {ceiling}"
    assert pool_stats()["checked_out"] <= 5, "connections outlived their requests"
