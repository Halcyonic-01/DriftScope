"""
tests/test_security.py

Unit tests for the require_api_key dependency in isolation — no DB, no
HTTP client, just the dependency function against a monkeypatched setting.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.core.security import require_api_key


def test_auth_disabled_when_api_key_unset(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    require_api_key(x_api_key=None)  # should not raise


def test_rejects_missing_key_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")

    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key=None)

    assert exc_info.value.status_code == 401


def test_rejects_wrong_key_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")

    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key="wrong-key")

    assert exc_info.value.status_code == 401


def test_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")
    require_api_key(x_api_key="secret123")  # should not raise
