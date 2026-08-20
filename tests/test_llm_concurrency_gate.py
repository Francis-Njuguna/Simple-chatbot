from __future__ import annotations

import asyncio

import pytest

from backend.app.config import Settings
from backend.app.rag.llm import LLMQueueBusyError, LLMService


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


def _service(llm, **overrides) -> LLMService:  # noqa: ANN001
    values = {
        "LLM_PROVIDER": "agentrouter",
        "AGENTROUTER_API_KEY": "sk-test-not-real",
        "AGENTROUTER_BASE_URL": "https://agentrouter.example/v1",
        "OPENAI_API_BASE": "",
        "LLM_MAX_CONCURRENCY": 1,
        "LLM_QUEUE_TIMEOUT": 0.2,
    }
    values.update(overrides)
    service = LLMService.__new__(LLMService)
    service.settings = Settings(**values)
    service._llm = llm
    return service


@pytest.mark.asyncio
async def test_provider_calls_never_exceed_the_configured_concurrency() -> None:
    class _CountingLLM:
        active = 0
        maximum = 0

        async def ainvoke(self, _prompt):  # noqa: ANN001
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await asyncio.sleep(0.03)
                return _Response("ok")
            finally:
                self.active -= 1

    llm = _CountingLLM()
    service = _service(llm, LLM_QUEUE_TIMEOUT=2)

    answers = await asyncio.gather(*(service.complete("system", str(i)) for i in range(5)))

    assert answers == ["ok"] * 5
    assert llm.maximum == 1


@pytest.mark.asyncio
async def test_queue_timeout_is_reported_without_starting_another_provider_call() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockingLLM:
        calls = 0

        async def ainvoke(self, _prompt):  # noqa: ANN001
            self.calls += 1
            entered.set()
            await release.wait()
            return _Response("done")

    llm = _BlockingLLM()
    service = _service(llm, LLM_QUEUE_TIMEOUT=0.05)
    first = asyncio.create_task(service.complete("system", "first"))
    await entered.wait()

    with pytest.raises(LLMQueueBusyError):
        await service.complete("system", "second")

    streamed = "".join([part async for part in service.stream_answer("q", "ctx")])
    assert "handling other requests" in streamed
    assert llm.calls == 1

    release.set()
    assert await first == "done"


@pytest.mark.asyncio
async def test_stream_releases_slot_after_normal_completion_and_error() -> None:
    class _StreamingLLM:
        streams = 0

        def astream(self, _prompt):  # noqa: ANN001
            self.streams += 1

            async def generate():
                if self.streams == 1:
                    yield _Response("first")
                    return
                raise RuntimeError("provider failed")
                yield  # pragma: no cover

            return generate()

        async def ainvoke(self, _prompt):  # noqa: ANN001
            return _Response("slot available")

    service = _service(_StreamingLLM())

    assert "".join([part async for part in service.stream_answer("q", "ctx")]) == "first"
    failed = "".join([part async for part in service.stream_answer("q", "ctx")])
    assert "could not generate an answer" in failed
    assert await service.complete("system", "user") == "slot available"


@pytest.mark.asyncio
async def test_cancelled_stream_releases_slot() -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()

    class _CancellableLLM:
        def astream(self, _prompt):  # noqa: ANN001
            async def generate():
                try:
                    entered.set()
                    await asyncio.Event().wait()
                    yield _Response("never")
                finally:
                    closed.set()

            return generate()

        async def ainvoke(self, _prompt):  # noqa: ANN001
            return _Response("after cancellation")

    service = _service(_CancellableLLM())

    async def consume() -> None:
        async for _ in service.stream_answer("q", "ctx"):
            pass

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(closed.wait(), timeout=0.2)
    assert await service.complete("system", "user") == "after cancellation"
