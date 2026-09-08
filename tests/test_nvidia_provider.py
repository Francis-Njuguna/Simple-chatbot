"""Live verification for the sole production LLM provider: NVIDIA NIM."""

from __future__ import annotations

import os

import pytest

from backend.app.config import Settings
from backend.app.rag.llm import LLMService

FIXTURE_CONTEXT = """--- Article 1: Resetting your student portal password
Category: Student Portal
To reset your Amref student portal password, open the student portal sign-in
page and select "Forgot password". Enter your Amref university email address
and follow the reset link sent to that address. If the email does not arrive,
check your junk folder and contact the ICT Help Desk.
Source: https://helpdesk.amref.ac.ke/article/student-portal-password
---"""


def _live_key() -> str:
    if os.getenv("RUN_NVIDIA_LIVE") != "1":
        return ""
    return os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY", "")


@pytest.mark.asyncio
@pytest.mark.live
async def test_live_nvidia_generates_a_grounded_answer() -> None:
    """Exercise the real NVIDIA endpoint and verify supplied context is used."""
    key = _live_key()
    if not key:
        pytest.skip("RUN_NVIDIA_LIVE=1 and NVIDIA_API_KEY are required")

    settings = Settings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY=key,
        OPENAI_API_BASE=os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
        OPENAI_MODEL=os.getenv(
            "NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b"
        ),
        LLM_TIMEOUT=45,
        LLM_MAX_RETRIES=1,
    )
    service = LLMService.__new__(LLMService)
    service.settings = settings
    service._llm = service._build_llm()

    try:
        answer = await service.generate_answer(
            question="How do I reset my student portal password?",
            context=FIXTURE_CONTEXT,
            history="No prior conversation.",
            images="No images accompany this answer.",
        )
    finally:
        client = getattr(service._llm, "root_async_client", None)
        if client is not None:
            await client.close()

    lowered = answer.lower()
    assert "could not generate an answer" not in lowered, answer
    assert "student portal" in lowered, answer
    assert "forgot password" in lowered or "reset link" in lowered, answer
