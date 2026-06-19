"""
tests/test_llm_factory.py

Smoke tests for the LLM client factory.
We ONLY test with MockLLMClient — no real API calls in unit tests.
"""

import pytest

from app.core.llm.factory import get_client, list_providers
from app.core.llm.base import LLMClient, LLMResponse
from app.core.llm.mock_client import MockLLMClient


def test_factory_returns_llm_client():
    """get_client() should return an object that is an LLMClient."""
    client = get_client("mock")
    assert isinstance(client, LLMClient)


def test_factory_mock_provider_returns_mock_client():
    """get_client('mock') should specifically return a MockLLMClient."""
    client = get_client("mock")
    assert isinstance(client, MockLLMClient)


def test_mock_client_complete_returns_llm_response():
    """MockLLMClient.complete() should return a proper LLMResponse dataclass."""
    client = get_client("mock")
    response = client.complete("What is 2 + 2?")

    assert isinstance(response, LLMResponse)
    assert isinstance(response.text, str)
    assert len(response.text) > 0
    assert response.provider == "mock"
    assert response.model == "mock-model"


def test_mock_client_fixed_response():
    """MockLLMClient should return exactly the text we configure it with."""
    expected = "The answer is 42."
    client = MockLLMClient(fixed_response=expected)
    response = client.complete("What is the answer?")

    assert response.text == expected


def test_mock_client_tracks_call_count():
    """MockLLMClient.call_count tracks how many times complete() was called."""
    client = MockLLMClient()
    assert client.call_count == 0

    client.complete("First call")
    assert client.call_count == 1

    client.complete("Second call")
    client.complete("Third call")
    assert client.call_count == 3


def test_factory_lists_all_providers():
    """list_providers() should include gemini, ollama, and mock."""
    providers = list_providers()
    assert "gemini" in providers
    assert "ollama" in providers
    assert "mock" in providers


def test_factory_raises_on_unknown_provider():
    """get_client() should raise ValueError for unrecognised providers."""
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_client("gpt-99-turbo-ultra")


def test_mock_health_check_always_true():
    """MockLLMClient.health_check() should always return True."""
    client = MockLLMClient()
    assert client.health_check() is True
