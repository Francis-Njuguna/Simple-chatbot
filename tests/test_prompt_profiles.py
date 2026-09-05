from backend.app.config import Settings
from backend.app.rag.llm import LLMService


def _service(profile: str) -> LLMService:
    service = LLMService.__new__(LLMService)
    service.settings = Settings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
        OPENAI_API_BASE="https://integrate.api.nvidia.com/v1",
        OPENAI_MODEL="nvidia/nemotron-3-super-120b-a12b",
        LLM_PROMPT_PROFILE=profile,
    )
    return service


def test_compact_prompt_preserves_grounding_and_short_query_rules() -> None:
    service = _service("compact")
    messages = service._build_messages(
        "MFA",
        "--- Article: Microsoft Authenticator\nSource: https://kb.example/mfa\nInstall the app and approve the sign-in.",
        "No prior conversation.",
        "No images accompany this answer.",
    )
    prompt = "\n".join(str(message.content) for message in messages)
    assert "Answer from the supplied Retrieved Knowledge Base Context only" in prompt
    assert "Never invent" in prompt
    assert "Bare or vague queries such as MFA" in prompt
    assert "User Question: MFA" in prompt


def test_compact_prompt_is_materially_smaller_than_legacy() -> None:
    context = "Article context " * 100
    args = ("How do I use MFA?", context, "No prior conversation.", "No images accompany this answer.")
    legacy = "\n".join(str(message.content) for message in _service("legacy")._build_messages(*args))
    compact = "\n".join(str(message.content) for message in _service("compact")._build_messages(*args))
    assert len(compact) < len(legacy) * 0.65


def test_compact_profile_keeps_subject_context_for_regression_queries() -> None:
    service = _service("compact")
    for question in (
        "MFA",
        "how do I use MFA",
        "Microsoft Authenticator",
        "how do I log into Moodle",
        "student portal",
        "SMOWL",
        "VAS exams",
    ):
        prompt = "\n".join(str(message.content) for message in service._build_messages(question, "relevant context", "No prior conversation.", "No images accompany this answer."))
        assert question in prompt
        assert "relevant supplied context" in prompt.lower()
