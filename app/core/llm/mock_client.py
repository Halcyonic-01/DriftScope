"""
app/core/llm/mock_client.py

A fake LLM client used ONLY in tests.

WHY a mock?
  Real LLM calls cost money (or time). When running unit tests we don't
  want to hit the real Gemini API for every test run. Instead we use a
  MockLLMClient that immediately returns a pre-configured response.

  This makes tests:
    - Fast     (no network round-trip)
    - Free     (no API credits consumed)
    - Reliable (no flakiness from network errors)
"""

from __future__ import annotations

from app.core.llm.base import LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    """
    Fake LLM client that returns a fixed response.
    Perfect for unit and integration tests.

    Usage in tests:
        client = MockLLMClient(fixed_response="The sky is blue.")
        resp = client.complete("What color is the sky?")
        assert resp.text == "The sky is blue."
    """

    def __init__(
        self,
        fixed_response: str = "This is a mock LLM response.",
        fixed_judge_response: str = '{"pass": true, "reason": "Mock judge passed."}',
    ) -> None:
        self._fixed_response = fixed_response
        self._fixed_judge_response = fixed_judge_response
        self._call_count = 0   # track how many times complete() was called

    def complete(self, prompt: str, response_mime_type: str | None = None) -> LLMResponse:
        self._call_count += 1
        text = (
            self._fixed_judge_response
            if response_mime_type == "application/json"
            else self._fixed_response
        )
        return LLMResponse(
            text=text,
            provider="mock",
            model="mock-model",
            tokens_used=42,   # arbitrary
        )

    def health_check(self) -> bool:
        return True   # always healthy in tests

    @property
    def call_count(self) -> int:
        """How many times complete() has been called — useful for assertions."""
        return self._call_count
