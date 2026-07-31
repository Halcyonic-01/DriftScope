"""
app/core/config.py

Central settings — reads from .env file automatically.
Uses pydantic-settings so every variable is type-checked at startup.
"""

from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database ───────────────────────────────────────────────────────
    # min_length=1 so a blank DATABASE_URL (e.g. a CI secret that isn't
    # configured) fails immediately with a clear config error, instead of
    # sailing through as "" and blowing up later inside SQLAlchemy.
    database_url: str = Field(min_length=1)

    # ── LLM ───────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    # Pin a concrete model, not a "-latest" alias. DriftScope exists to
    # detect when a provider silently changes the model behind a fixed
    # name; an alias that intentionally moves would manufacture drift
    # signals and make the canary meaningless.
    # Google retires model IDs for new API keys — if calls start 404ing,
    # set GEMINI_MODEL to a current one (list them with genai.list_models).
    gemini_model: str = "gemini-3.6-flash"
    gemini_max_output_tokens: int = 4096
    gemini_retry_max_output_tokens: int = 8192
    gemini_request_timeout_seconds: int = 60
    # Rate-limit retries. A single eval case is a burst of calls (one
    # generation plus one judge call per rule), so free-tier limits are
    # hit in bursts no fixed inter-case delay can prevent. Retrying with
    # the server-supplied backoff is what actually makes runs survive.
    gemini_max_retries: int = 4
    gemini_max_retry_wait_seconds: int = 90
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_request_timeout_seconds: int = 120
    judge_cosine_threshold: float = 0.65
    composite_cosine_weight: float = 0.6
    composite_judge_weight: float = 0.4

    # ── App ───────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── API Auth ──────────────────────────────────────────────────────
    # Shared-secret key required via the X-API-Key header on protected
    # routes. Leave empty (the default) to disable auth entirely — that's
    # the local/dev default so nothing breaks if you don't set it.
    api_key: str = ""

    # ── Email Alerts ──────────────────────────────────────────────────
    alert_email: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_password: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # .env is shared with docker compose, which reads the same file and
        # needs variables the app itself doesn't define (DRIFTSCOPE_DB_URL,
        # for one). pydantic-settings forbids unknown keys by default, so
        # without this a compose-only variable crashes every entrypoint that
        # imports settings — API, scripts, and tests alike.
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _blank_env_vars_fall_back_to_defaults(cls, data: Any) -> Any:
        """
        Treat a blank env var as "not set" for any field that has a default.

        CI runners and orchestrators routinely define a variable as an empty
        string when the underlying secret is missing — GitHub Actions does
        exactly this for `SMTP_PORT: ${{ secrets.SMTP_PORT }}` when
        SMTP_PORT isn't configured. Without this, pydantic tries to parse
        "" as an int and the whole app fails at import time, because
        `settings = Settings()` runs at module scope.

        Fields whose default is already "" (api_key, gemini_api_key, and
        the smtp_user/smtp_password credentials) are unaffected — they end
        up as "" either way, so the "empty means disabled" sentinel that
        send_configured_email_alert() and require_api_key() rely on still
        behaves identically.

        Required fields with no default (database_url) are deliberately left
        alone, so a blank DATABASE_URL still fails loudly rather than being
        quietly swallowed — see the min_length=1 constraint on that field.
        """
        if not isinstance(data, dict):
            return data

        optional_fields = {
            name.lower()
            for name, field in cls.model_fields.items()
            if not field.is_required()
        }

        return {
            key: value
            for key, value in data.items()
            if not (
                isinstance(value, str)
                and not value.strip()
                and str(key).lower() in optional_fields
            )
        }


# Single shared instance — import this everywhere
settings = Settings()
