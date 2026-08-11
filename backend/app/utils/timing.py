"""Lightweight per-stage timing utilities for the RAG request pipeline.

Usage
-----
    tracker = StageTimer("chat")
    with tracker.stage("embedding"):
        ...
    async with tracker.astage("llm"):
        ...
    tracker.log(logger)

The tracker keeps a millisecond breakdown of every named stage so we can see
exactly where a request spends its time. It is intentionally dependency-free
and adds negligible overhead (a couple of ``perf_counter`` calls per stage).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from logging import Logger

# When set, every StageTimer constructed inside this context registers itself
# here. Diagnostics need a request's stage breakdown as data, and the timer is
# a local variable inside the service method — reachable only by parsing the
# log line it emits. A ContextVar rather than a module global because requests
# run concurrently in one event loop, and a global would collect whichever
# request happened to be running.
_timer_sink: ContextVar[list["StageTimer"] | None] = ContextVar(
    "stage_timer_sink", default=None
)


@contextmanager
def collect_timers():
    """Collect every StageTimer created in this context.

    Intended for benchmarks and diagnostics, not for the request path::

        with collect_timers() as timers:
            await service.chat("...")
        breakdown = timers[0].stages
    """
    collected: list[StageTimer] = []
    token = _timer_sink.set(collected)
    try:
        yield collected
    finally:
        _timer_sink.reset(token)


class StageTimer:
    """Accumulates elapsed time per named stage of a request."""

    def __init__(self, label: str = "request") -> None:
        self.label = label
        self._start = time.perf_counter()
        self.stages: dict[str, float] = {}
        sink = _timer_sink.get()
        if sink is not None:
            sink.append(self)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = (time.perf_counter() - start) * 1000.0

    @asynccontextmanager
    async def astage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = (time.perf_counter() - start) * 1000.0

    def mark(self, name: str, elapsed_ms: float) -> None:
        self.stages[name] = elapsed_ms

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def report(self) -> str:
        parts = [f"{name}={ms:.0f}ms" for name, ms in self.stages.items()]
        parts.append(f"TOTAL={self.total_ms:.0f}ms")
        return f"[timing:{self.label}] " + " ".join(parts)

    def log(self, logger: Logger) -> None:
        logger.info(self.report())
