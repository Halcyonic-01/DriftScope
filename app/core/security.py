"""
app/core/security.py

Simple shared-secret API key check.

Set API_KEY in .env to require every protected request to send it back
via the X-API-Key header. If API_KEY is left empty (the default), auth
is disabled entirely — that keeps local development and the existing
test suite working with no extra setup.

/health and /metrics are intentionally NOT protected by this — they're
infrastructure endpoints (liveness probes, Prometheus scraping) that
conventionally sit outside application-level auth.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency — raises 401 if API_KEY is set and the caller didn't send it."""
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Provide it via the X-API-Key header.",
        )
