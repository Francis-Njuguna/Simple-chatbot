"""The LLM failure budget — what a student waits through when generation stalls.

These tests exist because the budget is a *product*, not a value.
``LLM_TIMEOUT=30`` and ``LLM_MAX_RETRIES=2`` each look reasonable alone and
together mean ~91 seconds, which is exactly what three real requests spent in
the LLM stage on 2026-08-12 (``logs/app.log``: ``llm_first_token=90897ms``,
``95560ms``, ``94930ms``) — two of them ending in an error message shown to the
student. Nobody chose 91s. It was arithmetic nobody performed.

So the assertions are about *bounds*, not happy paths:

* the transport worst case equals the product, and is reported at startup;
* the granular connect/read split actually reaches the OpenAI client instead of
  being silently replaced by a scalar (it is passed twice, and only one of the
  two wins — see the note at the ``ChatOpenAI`` call in ``rag.llm``);
* time-to-first-token stays bounded even when the model streams content-free
  deltas forever, which reasoning models do by design;
* the bound is *released* once real text appears, so a long answer is never
  truncated merely for being long;
* and the two timeouts are never conflated — "never started" and "stopped
  half-way" are different failures and say different things to the user.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from backend.app.config import Settings
from backend.app.rag.llm import LLMService


class _Delta:
    """A chunk shaped like the ones langchain yields: text lives on ``.content``."""

    def __init__(self, content: str) -> None:
        self.content = content


class _ScriptedLLM:
    """Stands in for the langchain client, replaying a timed script.

    Each entry is ``(delay_seconds, content)``: sleep, then yield one chunk
    carrying ``content``. An empty ``content`` models the content-free deltas a
    reasoning model emits before any answer text — they are real chunks, so any
    per-chunk timer would be reset by them.
    """

    def __init__(self, script: list[tuple[float, str]]) -> None:
        self.script = list(script)
        self.closed = False

    def astream(self, _prompt):  # noqa: ANN001 — mirrors langchain's signature
        async def gen():
            try:
                for delay, content in self.script:
                    if delay:
                        await asyncio.sleep(delay)
                    yield _Delta(content)
            finally:
                self.closed = True

        return gen()


def _service(**overrides) -> LLMService:
    """An ``LLMService`` with no client built and no network touched.

    Constructed via ``__new__`` deliberately: ``__init__`` builds a real
    ChatOpenAI plus an httpx connection pool, which these tests replace anyway,
    and leaking one pool per test is worse than skipping the constructor.
    """
    svc = LLMService.__new__(LLMService)
    svc.settings = Settings(
        LLM_PROVIDER="agentrouter",
        AGENTROUTER_API_KEY="sk-test-not-a-real-key",
        AGENTROUTER_BASE_URL="https://agentrouter.example/v1",
        AGENTROUTER_MODEL="claude-opus-5",
        # Cleared because Settings still reads the developer's .env, where a
        # leftover OPENAI_API_BASE would raise an unrelated validation warning.
        OPENAI_API_BASE="",
        **overrides,
    )
    return svc


async def _collect(svc: LLMService) -> tuple[str, float]:
    started = time.monotonic()
    parts = [text async for text in svc.stream_answer("q", "ctx")]
    return "".join(parts), time.monotonic() - started


# --------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("timeout", "retries", "expected"),
    [
        (30, 2, 91.5),  # the shipped-by-accident budget that caused this file
        (25, 1, 50.5),  # the current default
        (25, 0, 25.0),  # no retries: the budget is just the timeout
    ],
)
def test_transport_worst_case_is_the_product(
    timeout: int, retries: int, expected: float
) -> None:
    settings = Settings(
        LLM_TIMEOUT=timeout, LLM_MAX_RETRIES=retries, OPENAI_API_BASE=""
    )
    assert settings.llm_transport_worst_case_seconds == pytest.approx(expected)


def test_startup_reports_the_budget() -> None:
    settings = Settings(LLM_TIMEOUT=25, LLM_MAX_RETRIES=1, OPENAI_API_BASE="")
    assert "50s" in _captured(settings)


def test_config_warns_when_the_ceiling_can_never_fire() -> None:
    """A ceiling above the transport budget is dead configuration, not safety."""
    settings = Settings(
        LLM_TIMEOUT=5,
        LLM_MAX_RETRIES=0,
        LLM_FIRST_TOKEN_TIMEOUT=60,
        OPENAI_API_BASE="",
    )
    assert "can never fire" in _captured(settings)


def test_config_warns_when_the_ceiling_is_disabled() -> None:
    settings = Settings(
        LLM_TIMEOUT=25, LLM_MAX_RETRIES=1, LLM_FIRST_TOKEN_TIMEOUT=0, OPENAI_API_BASE=""
    )
    assert "DISABLED" in _captured(settings)


def _captured(settings: Settings) -> str:
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        settings.log_llm_config()
    return buffer.getvalue()


# --------------------------------------------------------------------------
# The timeout must survive the trip into the OpenAI client
# --------------------------------------------------------------------------

def test_granular_timeout_reaches_the_openai_client() -> None:
    """``describe()`` reads the *built* client, so this catches a silent override.

    langchain forwards ``request_timeout`` to ``AsyncOpenAI``, and the SDK only
    consults the http client's own timeout when it was given none
    (``if not is_given(timeout)``). Passing a bare int would therefore discard
    the connect/read split without any error.
    """
    svc = _service(LLM_TIMEOUT=25, LLM_CONNECT_TIMEOUT=5, LLM_MAX_RETRIES=1)
    svc._llm = svc._build_llm()

    info = svc.describe()
    timeout = info["timeout"]
    assert isinstance(timeout, httpx.Timeout), f"expected a split budget, got {timeout!r}"
    assert timeout.connect == 5.0
    assert timeout.read == 25.0
    assert info["max_retries"] == 1


# --------------------------------------------------------------------------
# The first-token ceiling
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ceiling_fires_when_no_text_ever_arrives() -> None:
    svc = _service(LLM_FIRST_TOKEN_TIMEOUT=0.3)
    svc._llm = _ScriptedLLM([(30.0, "never reached")])

    answer, elapsed = await _collect(svc)

    assert "did not start answering" in answer
    assert elapsed < 3.0, f"deadline did not bound the wait ({elapsed:.1f}s)"


@pytest.mark.asyncio
async def test_content_free_deltas_do_not_rearm_the_ceiling() -> None:
    """The reasoning-model case, and the reason this is a deadline not a gap timer.

    200 empty deltas 20ms apart is four seconds of steady "progress" carrying no
    answer text. A per-chunk timeout sees a healthy stream throughout and never
    fires; a deadline on the whole pre-text phase stops it.
    """
    svc = _service(LLM_FIRST_TOKEN_TIMEOUT=0.3)
    svc._llm = _ScriptedLLM([(0.02, "")] * 200)

    answer, elapsed = await _collect(svc)

    assert "did not start answering" in answer
    assert elapsed < 3.0, f"empty deltas kept the deadline alive ({elapsed:.1f}s)"


@pytest.mark.asyncio
async def test_ceiling_is_released_once_real_text_arrives() -> None:
    """A gap longer than the ceiling, *after* first text, must not truncate."""
    svc = _service(LLM_FIRST_TOKEN_TIMEOUT=0.3)
    svc._llm = _ScriptedLLM([(0.05, "Step 1"), (0.6, " and step 2")])

    answer, _ = await _collect(svc)

    assert answer == "Step 1 and step 2"


@pytest.mark.asyncio
async def test_zero_disables_the_ceiling() -> None:
    svc = _service(LLM_FIRST_TOKEN_TIMEOUT=0)
    svc._llm = _ScriptedLLM([(0.4, "late but fine")])

    answer, _ = await _collect(svc)

    assert answer == "late but fine"


@pytest.mark.asyncio
async def test_normal_stream_passes_through_and_closes_the_iterator() -> None:
    svc = _service(LLM_FIRST_TOKEN_TIMEOUT=30)
    fake = _ScriptedLLM([(0, "Open "), (0, "the "), (0, "portal.")])
    svc._llm = fake

    answer, _ = await _collect(svc)

    assert answer == "Open the portal."
    assert fake.closed, "the httpx connection was left to garbage collection"


# --------------------------------------------------------------------------
# The two timeouts must stay distinguishable
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_stall_after_first_token_reads_as_truncation_not_silence() -> None:
    """``StreamChunkTimeoutError`` subclasses ``TimeoutError``, so one ``except``
    catches both deadlines and has to tell the user which one happened."""

    class _StallingLLM:
        def astream(self, _prompt):  # noqa: ANN001
            async def gen():
                yield _Delta("Step 1: open the portal")
                raise TimeoutError  # what stream_chunk_timeout raises

            return gen()

    svc = _service()
    svc._llm = _StallingLLM()

    answer, _ = await _collect(svc)

    assert answer.startswith("Step 1: open the portal")
    assert "cut off" in answer
    assert "did not start answering" not in answer
