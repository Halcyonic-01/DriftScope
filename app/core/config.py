"""
app/core/config.py

Central settings — reads from .env file automatically.
Uses pydantic-settings so every variable is type-checked at startup.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database ───────────────────────────────────────────────────────
    database_url: str

    # ── LLM ───────────────────────────────────────────────────────────
    gemini_api_key: str = ""

    # ── App ───────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

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
    )


# Single shared instance — import this everywhere
settings = Settings()
