"""Provider tests for the AgentRouter → Anthropic Claude path.

Two layers, because they fail for different reasons and one must not mask the
other:

* **Wiring** (no network) — the ``agentrouter`` branch builds a ChatOpenAI
  client pointed at the configured base URL and model, refuses to build without
  a key, and is treated as a chat model so the system/user message split is
  preserved. These run everywhere, including CI without credentials.
* **Live** — one real call through ``LLMService.generate_answer`` proving the
  configured key actually reaches Claude and comes back with grounded text.
  Skipped when no key is configured.

The live test asserts on returned *content*, never on "no exception was
raised". ``generate_answer`` catches every exception and returns
``_error_message(exc)`` as the answer, so a provider outage produces a
perfectly ordinary-looking string. A test that only checked for exceptions
would pass during a total outage.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from backend.app.config import Settings, get_settings
from backend.app.rag.llm import LLMService

# Content invented for this test — deliberately not in the knowledge base, so an
# answer containing it can only have come from the prompt we sent.
FIXTURE_CONTEXT = """--- Article 1: Resetting Your Zamboni Portal PIN
Category: IT Support
To reset your Zamboni Portal PIN, open https://zamboni.example.edu/reset and
sign in. Click "Reset PIN" in the left menu, enter the 7-digit code sent to your
staff phone, then choose a new PIN. The new PIN takes 12 minutes to propagate.
Source: https://zamboni.example.edu/help/pin
---"""


def _agentrouter_settings(**overrides) -> Settings:
    """A Settings instance pinned to the agentrouter provider.

    Built directly rather than through ``get_settings`` because that is
    ``lru_cache``d on the real environment; these tests must not depend on, or
    disturb, whatever provider the developer has configured in ``.env``.
    """
    values = {
        "LLM_PROVIDER": "agentrouter",
        "AGENTROUTER_API_KEY": "sk-test-not-a-real-key",
        "AGENTROUTER_BASE_URL": "https://agentrouter.example/v1",
        "AGENTROUTER_MODEL": "claude-opus-5",
        # Explicitly cleared: Settings still reads the developer's .env, and a
        # leftover NVIDIA OPENAI_API_BASE there would otherwise show up as a
        # validation warning these tests would attribute to AgentRouter.
        "OPENAI_API_BASE": "",
        **overrides,
    }
    return Settings(**values)


def _service_with(settings: Settings) -> LLMService:
    """Construct LLMService against explicit settings, bypassing the cache."""
    service = LLMService.__new__(LLMService)
    service.settings = settings
    service._llm = service._build_llm()
    return service


# ---------------------------------------------------------------------------
# Wiring — no network
# ---------------------------------------------------------------------------


def test_agentrouter_builds_an_openai_compatible_client() -> None:
    """The provider uses ChatOpenAI against the configured base URL and model.

    AgentRouter exposes Claude through /v1/chat/completions, so the OpenAI
    client is correct here and the Anthropic SDK is not involved.
    """
    from langchain_openai import ChatOpenAI

    service = _service_with(_agentrouter_settings())

    assert isinstance(service._llm, ChatOpenAI)
    assert service._llm.model_name == "claude-opus-5"
    assert str(service._llm.openai_api_base) == "https://agentrouter.example/v1"


def test_agentrouter_sends_the_configured_user_agent() -> None:
    """The user-agent reaches the client's default headers.

    AgentRouter 401s clients it does not recognise, so this header is part of
    the request contract, not cosmetics. Pinning it here means a refactor that
    drops ``default_headers`` fails loudly instead of at runtime.
    """
    service = _service_with(_agentrouter_settings())
    headers = service._llm.default_headers or {}

    assert headers.get("User-Agent") == "claude-cli/1.0.0 (external, cli)"


def test_agentrouter_user_agent_can_be_disabled() -> None:
    """An empty AGENTROUTER_USER_AGENT sends no override."""
    service = _service_with(_agentrouter_settings(AGENTROUTER_USER_AGENT=""))

    assert not (service._llm.default_headers or {}).get("User-Agent")


def test_missing_key_fails_fast_with_a_named_variable() -> None:
    """No key must raise at build time, naming the variable to set."""
    with pytest.raises(RuntimeError, match="AGENTROUTER_API_KEY"):
        _service_with(_agentrouter_settings(AGENTROUTER_API_KEY=""))


def test_placeholder_key_is_rejected() -> None:
    """A copied-from-.env.example key must not reach the network."""
    with pytest.raises(RuntimeError, match="placeholder"):
        _service_with(_agentrouter_settings(AGENTROUTER_API_KEY="your-key-here"))


def test_anthropic_auth_key_is_accepted_as_a_fallback() -> None:
    """Deployments already pointing ANTHROPIC_AUTH_KEY at AgentRouter keep working."""
    settings = Settings(
        LLM_PROVIDER="agentrouter",
        ANTHROPIC_AUTH_KEY="sk-test-fallback",
        AGENTROUTER_BASE_URL="https://agentrouter.example/v1",
    )
    assert settings.agentrouter_api_key == "sk-test-fallback"
    # And it builds, rather than raising for a missing AGENTROUTER_API_KEY.
    assert _service_with(settings)._llm is not None


def test_agentrouter_uses_the_chat_message_split() -> None:
    """System and user content stay separate messages.

    If ``agentrouter`` were missing from ``_use_chat_model``, the prompt would
    silently degrade to one flat concatenated string — the grounding rules and
    the retrieved context would arrive as undifferentiated text.
    """
    service = _service_with(_agentrouter_settings())
    assert service._use_chat_model() is True

    messages = service._build_messages(
        question="How do I reset my PIN?",
        context=FIXTURE_CONTEXT,
        history="No prior conversation.",
        images="No images accompany this answer.",
    )
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "Amref Help Desk Assistant" in messages[0].content
    assert FIXTURE_CONTEXT in messages[1].content


def test_error_message_names_the_agentrouter_model() -> None:
    """A model error reports the model actually configured.

    The diagnostic previously said "see config" for every provider except
    anthropic, which is useless in exactly the case you need it.
    """
    service = _service_with(_agentrouter_settings())

    class NotFoundError(Exception):
        pass

    assert "claude-opus-5" in service._error_message(NotFoundError("nope"))


def test_error_message_explains_an_agentrouter_401() -> None:
    """AgentRouter 401s for two different reasons; the message must say both."""
    service = _service_with(_agentrouter_settings())

    class AuthenticationError(Exception):
        pass

    message = service._error_message(AuthenticationError("401"))
    assert "AGENTROUTER_API_KEY" in message
    assert "user-agent" in message
    # And it must never read as a knowledge-base gap.
    assert "not a gap in the knowledge base" in message


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_validation_reports_missing_key() -> None:
    problems = _agentrouter_settings(AGENTROUTER_API_KEY="").validate_llm_config()
    assert any("AGENTROUTER_API_KEY" in p for p in problems)


def test_validation_reports_a_base_url_missing_v1() -> None:
    problems = _agentrouter_settings(
        AGENTROUTER_BASE_URL="https://agentrouter.org"
    ).validate_llm_config()
    assert any("/v1" in p for p in problems)


def test_validation_is_clean_for_a_good_config() -> None:
    assert _agentrouter_settings().validate_llm_config() == []


def test_validation_never_leaks_the_credential() -> None:
    """Problem text names variables, never values."""
    secret = "sk-super-secret-value"
    blob = repr(
        _agentrouter_settings(
            AGENTROUTER_API_KEY=secret, AGENTROUTER_BASE_URL=""
        ).validate_llm_config()
    )
    assert secret not in blob


# ---------------------------------------------------------------------------
# Live — one real call through the configured provider
# ---------------------------------------------------------------------------


def _live_key() -> str:
    """The real key, only if the environment actually has one configured."""
    settings = get_settings()
    return os.getenv("AGENTROUTER_API_KEY") or settings.agentrouter_api_key or ""


@pytest.mark.asyncio
async def test_live_agentrouter_generates_a_grounded_answer() -> None:
    """A real AgentRouter → Claude call answers from the supplied context.

    This proves the whole path: credential, base URL, user-agent, model name,
    the chat message split, and that the model grounds in what it was given.

    The context is fabricated, so the URL and the 12-minute detail cannot come
    from the model's own knowledge or from the real knowledge base — if they
    appear in the answer, the context genuinely reached Claude.
    """
    if not _live_key():
        pytest.skip("no AgentRouter key configured — live provider test skipped")

    settings = Settings(
        LLM_PROVIDER="agentrouter",
        AGENTROUTER_API_KEY=_live_key(),
        AGENTROUTER_BASE_URL=get_settings().agentrouter_base_url,
        AGENTROUTER_MODEL=get_settings().agentrouter_model,
        LLM_TIMEOUT=120,
        LLM_MAX_RETRIES=1,
    )
    service = _service_with(settings)

    answer = await service.generate_answer(
        question="How do I reset my Zamboni Portal PIN?",
        context=FIXTURE_CONTEXT,
        history="No prior conversation.",
        images="No images accompany this answer.",
    )

    # generate_answer swallows exceptions and returns the error text as the
    # answer, so check for that explicitly before asserting on quality.
    assert "could not generate an answer" not in answer.lower(), (
        f"AgentRouter call failed: {answer}"
    )
    assert len(answer) > 80, f"suspiciously short answer: {answer!r}"
    assert "zamboni.example.edu" in answer.lower(), (
        f"answer is not grounded in the supplied context: {answer!r}"
    )
    assert "12 minutes" in answer, f"specific detail lost: {answer!r}"


@pytest.mark.asyncio
async def test_live_agentrouter_answers_a_short_keyword_query() -> None:
    """A bare keyword must not be declined when the context covers it.

    This is the regression for the reported behaviour: "MFA" and similar
    one-word queries were answered with "the knowledge base didn't retrieve any
    relevant articles" even though relevant material was supplied.
    """
    if not _live_key():
        pytest.skip("no AgentRouter key configured — live provider test skipped")

    settings = Settings(
        LLM_PROVIDER="agentrouter",
        AGENTROUTER_API_KEY=_live_key(),
        AGENTROUTER_BASE_URL=get_settings().agentrouter_base_url,
        AGENTROUTER_MODEL=get_settings().agentrouter_model,
        LLM_TIMEOUT=120,
        LLM_MAX_RETRIES=1,
    )
    service = _service_with(settings)

    answer = await service.generate_answer(
        question="Zamboni PIN",
        context=FIXTURE_CONTEXT,
        history="No prior conversation.",
        images="No images accompany this answer.",
    )

    assert "could not generate an answer" not in answer.lower(), (
        f"AgentRouter call failed: {answer}"
    )
    lowered = answer.lower()
    for phrase in (
        "didn't retrieve",
        "did not retrieve",
        "no relevant articles",
        "isn't covered by the amref",
        "not covered by the amref",
    ):
        assert phrase not in lowered, (
            f"declined a keyword query despite relevant context ({phrase!r}): {answer!r}"
        )
    assert "zamboni.example.edu" in lowered
