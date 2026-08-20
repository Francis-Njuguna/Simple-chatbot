from backend.app.config import Settings


def _production_settings(**overrides) -> Settings:
    values = {
        "APP_ENV": "production",
        "SECRET_KEY": "a" * 64,
        "CORS_ORIGINS": "https://helpdesk.example.edu",
        "CORS_ALLOW_CREDENTIALS": True,
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key-not-real",
        "OPENAI_API_BASE": "https://integrate.api.nvidia.com/v1",
        "OPENAI_MODEL": "meta/llama-3.1-8b-instruct",
        **overrides,
    }
    return Settings(**values)


def test_production_configuration_accepts_explicit_secure_values() -> None:
    assert _production_settings().validate_production_config() == []


def test_production_configuration_rejects_template_secret() -> None:
    problems = _production_settings(SECRET_KEY="change-me").validate_production_config()
    assert any("SECRET_KEY" in problem for problem in problems)


def test_production_configuration_rejects_credentialed_wildcard_cors() -> None:
    problems = _production_settings(CORS_ORIGINS="*").validate_production_config()
    assert any("CORS_ORIGINS" in problem for problem in problems)


def test_cors_origins_are_parsed_and_trimmed() -> None:
    settings = _production_settings(
        CORS_ORIGINS="https://one.example, https://two.example,"
    )
    assert settings.cors_origin_list == ["https://one.example", "https://two.example"]

