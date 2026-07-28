"""
tests/test_ollama_client.py

Unit tests for OllamaClient's HTTP error handling and settings-driven
configuration. httpx.Client is monkeypatched with a small fake so these
run instantly with no real Ollama server or network access.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import settings
from app.core.llm import ollama_client as ollama_module
from app.core.llm.base import LLMProviderError


class _FakeHTTPResponse:
    def __init__(self, json_data: dict | None = None, status_code: int = 200, text: str = ""):
        self._json = json_data or {}
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://fake-ollama/api/generate")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> dict:
        return self._json


class _FakeGetResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


class _FakeHTTPXClient:
    def __init__(self, post_response=None, post_exception=None, get_response=None, get_exception=None):
        self._post_response = post_response
        self._post_exception = post_exception
        self._get_response = get_response
        self._get_exception = get_exception

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def post(self, url, json=None):
        if self._post_exception:
            raise self._post_exception
        return self._post_response

    def get(self, url):
        if self._get_exception:
            raise self._get_exception
        return self._get_response


def _patch_client(monkeypatch, fake_client) -> None:
    monkeypatch.setattr(ollama_module.httpx, "Client", lambda timeout=None: fake_client)


def test_default_base_url_and_model_come_from_settings():
    client = ollama_module.OllamaClient()
    assert client._base_url == settings.ollama_base_url.rstrip("/")
    assert client._model == settings.ollama_model


def test_explicit_base_url_and_model_override_settings():
    client = ollama_module.OllamaClient(base_url="http://custom:11434/", model="mistral")
    assert client._base_url == "http://custom:11434"
    assert client._model == "mistral"


def test_complete_returns_response_text(monkeypatch):
    fake = _FakeHTTPXClient(post_response=_FakeHTTPResponse({"response": "Hello there"}))
    _patch_client(monkeypatch, fake)

    client = ollama_module.OllamaClient(base_url="http://fake-ollama")
    result = client.complete("hi")

    assert result.text == "Hello there"
    assert result.provider == "ollama"
    assert result.model == client._model


def test_complete_raises_provider_error_on_connect_failure(monkeypatch):
    fake = _FakeHTTPXClient(post_exception=httpx.ConnectError("boom"))
    _patch_client(monkeypatch, fake)

    client = ollama_module.OllamaClient(base_url="http://fake-ollama")
    with pytest.raises(LLMProviderError, match="Could not connect"):
        client.complete("hi")


def test_complete_raises_provider_error_on_timeout(monkeypatch):
    fake = _FakeHTTPXClient(post_exception=httpx.TimeoutException("timed out"))
    _patch_client(monkeypatch, fake)

    client = ollama_module.OllamaClient(base_url="http://fake-ollama")
    with pytest.raises(LLMProviderError, match="timed out"):
        client.complete("hi")


def test_complete_raises_provider_error_on_bad_http_status(monkeypatch):
    fake = _FakeHTTPXClient(post_response=_FakeHTTPResponse(status_code=500, text="server exploded"))
    _patch_client(monkeypatch, fake)

    client = ollama_module.OllamaClient(base_url="http://fake-ollama")
    with pytest.raises(LLMProviderError, match="HTTP 500"):
        client.complete("hi")


def test_complete_raises_provider_error_on_empty_completion(monkeypatch):
    fake = _FakeHTTPXClient(post_response=_FakeHTTPResponse({"response": "   "}))
    _patch_client(monkeypatch, fake)

    client = ollama_module.OllamaClient(base_url="http://fake-ollama")
    with pytest.raises(LLMProviderError, match="empty completion"):
        client.complete("hi")


def test_health_check_true_when_server_responds_200(monkeypatch):
    fake = _FakeHTTPXClient(get_response=_FakeGetResponse(status_code=200))
    _patch_client(monkeypatch, fake)

    client = ollama_module.OllamaClient(base_url="http://fake-ollama")
    assert client.health_check() is True


def test_health_check_false_when_server_unreachable(monkeypatch):
    fake = _FakeHTTPXClient(get_exception=httpx.ConnectError("boom"))
    _patch_client(monkeypatch, fake)

    client = ollama_module.OllamaClient(base_url="http://fake-ollama")
    assert client.health_check() is False
