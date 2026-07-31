"""
tests/test_config.py

Unit tests for Settings env-var parsing.

Regression cover for a nightly-canary CI failure: GitHub Actions
substitutes an empty string for `${{ secrets.X }}` when the secret isn't
configured, so the workflow exported SMTP_PORT="". pydantic tried to
parse "" as an int and raised at import time (settings = Settings() runs
at module scope), taking down the whole canary run before it started.

Every test builds its own Settings with _env_file=None so it reads only
the monkeypatched environment, never the developer's real .env.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings

VALID_DB_URL = "postgresql://user:pass@localhost:5432/driftscope"


@pytest.fixture()
def clean_env(monkeypatch):
    """Strip every DriftScope-relevant var so each test starts from scratch."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    return monkeypatch


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_blank_int_env_var_falls_back_to_default(clean_env):
    """The exact CI failure: SMTP_PORT='' must not crash config loading."""
    clean_env.setenv("SMTP_PORT", "")

    assert _settings().smtp_port == 465


def test_whitespace_only_int_env_var_falls_back_to_default(clean_env):
    clean_env.setenv("SMTP_PORT", "   ")

    assert _settings().smtp_port == 465


def test_blank_str_env_var_falls_back_to_non_empty_default(clean_env):
    clean_env.setenv("SMTP_HOST", "")

    assert _settings().smtp_host == "smtp.gmail.com"


def test_blank_float_env_var_falls_back_to_default(clean_env):
    clean_env.setenv("JUDGE_COSINE_THRESHOLD", "")

    assert _settings().judge_cosine_threshold == 0.65


def test_all_blank_secrets_load_cleanly(clean_env):
    """
    Emulates the canary workflow running with none of the SMTP/LLM secrets
    configured — every one of them exported as "".
    """
    for name in ("SMTP_PORT", "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD",
                 "ALERT_EMAIL", "GEMINI_API_KEY", "API_KEY"):
        clean_env.setenv(name, "")

    settings = _settings()

    # Real defaults are restored where one exists...
    assert settings.smtp_port == 465
    assert settings.smtp_host == "smtp.gmail.com"
    # ...and "empty means disabled" sentinels are preserved where the
    # default is already "" (send_configured_email_alert / require_api_key
    # both branch on these being falsy).
    assert settings.alert_email == ""
    assert settings.smtp_user == ""
    assert settings.smtp_password == ""
    assert settings.gemini_api_key == ""
    assert settings.api_key == ""


def test_real_values_still_parse(clean_env):
    """The blank-handling must not interfere with genuinely provided values."""
    clean_env.setenv("SMTP_PORT", "587")
    clean_env.setenv("SMTP_HOST", "smtp.example.com")
    clean_env.setenv("API_KEY", "s3cret")

    settings = _settings()

    assert settings.smtp_port == 587
    assert isinstance(settings.smtp_port, int)
    assert settings.smtp_host == "smtp.example.com"
    assert settings.api_key == "s3cret"


def test_genuinely_invalid_int_still_raises(clean_env):
    """Blank is tolerated; garbage is not — we didn't silence real errors."""
    clean_env.setenv("SMTP_PORT", "not-a-number")

    with pytest.raises(ValidationError, match="int_parsing"):
        _settings()


def test_blank_database_url_fails_loudly(clean_env):
    """
    database_url has no default, so it must not be silently coerced to "" —
    that would surface as a confusing SQLAlchemy error much later instead
    of a clear config error at startup.
    """
    clean_env.setenv("DATABASE_URL", "")

    with pytest.raises(ValidationError, match="string_too_short"):
        _settings()


def test_missing_database_url_fails_loudly(clean_env):
    clean_env.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError, match="[Ff]ield required"):
        _settings()


def test_unknown_env_vars_are_ignored(clean_env):
    """
    .env is shared with docker compose, which needs variables the app
    doesn't define (e.g. DRIFTSCOPE_DB_URL). pydantic-settings forbids
    unknown keys by default, which crashed every entrypoint that imports
    settings as soon as such a variable was added.
    """
    clean_env.setenv("DRIFTSCOPE_DB_URL", "postgresql://compose-only@host/db")
    clean_env.setenv("SOME_OTHER_TOOL_VAR", "whatever")

    settings = _settings()

    assert settings.database_url == VALID_DB_URL


def test_gemini_model_is_configurable(clean_env):
    clean_env.setenv("GEMINI_MODEL", "gemini-9.9-flash")
    assert _settings().gemini_model == "gemini-9.9-flash"


def test_gemini_model_defaults_to_a_pinned_version(clean_env):
    """Not a '-latest' alias: a moving target would manufacture drift."""
    model = _settings().gemini_model
    assert model
    assert "latest" not in model
