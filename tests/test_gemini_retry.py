"""
tests/test_gemini_retry.py

Unit tests for Gemini rate-limit retry.

A full monitor run failed 20/20 cases on 429s even with 16s between
cases. The reason is that a case is not one call: it's a generation call
plus one judge call per rule, fired back-to-back. No inter-case delay
prevents that burst, so the client has to retry using the delay the API
supplies.

genai is stubbed entirely — these run offline with no API key.
"""

from __future__ import annotations

import sys
import types

import pytest
from google.api_core import exceptions as google_exceptions

from app.core.config import settings


@pytest.fixture()
def client(monkeypatch):
    """A GeminiClient whose underlying SDK model is a controllable stub."""
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_max_retries", 4)

    from app.core.llm import gemini_client as gc

    monkeypatch.setattr(gc.genai, "configure", lambda **kw: None)
    monkeypatch.setattr(gc.genai, "GenerativeModel", lambda **kw: object())
    # Never actually sleep in tests.
    monkeypatch.setattr(gc.time, "sleep", lambda s: None)

    return gc.GeminiClient()


def _ok_response(text="hello"):
    part = types.SimpleNamespace(text=text)
    content = types.SimpleNamespace(parts=[part])
    candidate = types.SimpleNamespace(
        content=content, finish_reason=types.SimpleNamespace(name="STOP")
    )
    return types.SimpleNamespace(
        text=text, candidates=[candidate], usage_metadata=None
    )


def _rate_limited(seconds=None):
    exc = google_exceptions.ResourceExhausted("429 quota exceeded")
    if seconds is not None:
        exc.retry_delay = types.SimpleNamespace(seconds=seconds)
    return exc


def test_retries_then_succeeds(client, monkeypatch):
    calls = {"n": 0}

    def flaky(prompt, generation_config):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _rate_limited(seconds=1)
        return _ok_response("recovered")

    monkeypatch.setattr(client, "_generate_with_truncation_retry", flaky)

    result = client.complete("hi")

    assert result.text == "recovered"
    assert calls["n"] == 3  # two failures, then success


def test_honours_server_supplied_retry_delay(client, monkeypatch):
    waits = []
    from app.core.llm import gemini_client as gc

    monkeypatch.setattr(gc.time, "sleep", lambda s: waits.append(s))

    calls = {"n": 0}

    def flaky(prompt, generation_config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rate_limited(seconds=7)
        return _ok_response()

    monkeypatch.setattr(client, "_generate_with_truncation_retry", flaky)
    client.complete("hi")

    assert waits == [7]


def test_falls_back_to_backoff_without_retry_delay(client, monkeypatch):
    waits = []
    from app.core.llm import gemini_client as gc

    monkeypatch.setattr(gc.time, "sleep", lambda s: waits.append(s))

    def always_limited(prompt, generation_config):
        raise _rate_limited(seconds=None)

    monkeypatch.setattr(client, "_generate_with_truncation_retry", always_limited)

    with pytest.raises(Exception):
        client.complete("hi")

    # Exponential, and never unbounded.
    assert waits == [2, 4, 8]
    assert all(w <= settings.gemini_max_retry_wait_seconds for w in waits)


def test_gives_up_after_max_retries(client, monkeypatch):
    from app.core.llm.base import LLMProviderError

    calls = {"n": 0}

    def always_limited(prompt, generation_config):
        calls["n"] += 1
        raise _rate_limited(seconds=1)

    monkeypatch.setattr(client, "_generate_with_truncation_retry", always_limited)

    with pytest.raises(LLMProviderError, match="still exhausted"):
        client.complete("hi")

    assert calls["n"] == settings.gemini_max_retries  # bounded, not infinite


def test_retry_wait_is_capped(client, monkeypatch):
    waits = []
    from app.core.llm import gemini_client as gc

    monkeypatch.setattr(gc.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(settings, "gemini_max_retry_wait_seconds", 5)

    calls = {"n": 0}

    def flaky(prompt, generation_config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rate_limited(seconds=3600)  # absurd server delay
        return _ok_response()

    monkeypatch.setattr(client, "_generate_with_truncation_retry", flaky)
    client.complete("hi")

    assert waits == [5]  # capped, so a run can't hang for an hour


def test_non_rate_limit_errors_are_not_retried(client, monkeypatch):
    """A retired model 404 must fail fast, not burn four attempts."""
    from app.core.llm.base import LLMProviderError

    calls = {"n": 0}

    def not_found(prompt, generation_config):
        calls["n"] += 1
        raise google_exceptions.NotFound("404 model gone")

    monkeypatch.setattr(client, "_generate_with_truncation_retry", not_found)

    with pytest.raises(LLMProviderError, match="unavailable for this API key"):
        client.complete("hi")

    assert calls["n"] == 1
