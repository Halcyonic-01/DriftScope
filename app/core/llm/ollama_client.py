"""
app/core/llm/ollama_client.py

Ollama LLM client for running local models (e.g. Llama 3, Mistral).
This is the free local fallback — no API key needed, runs on your machine.

HOW OLLAMA WORKS:
  Ollama runs a local HTTP server (default: http://localhost:11434).
  We use httpx to send REST API calls to it.
  
  To use this client:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama3
    3. Set LLM_PROVIDER=ollama in your .env
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import settings
from app.core.llm.base import LLMClient, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """
    LLM client backed by a locally running Ollama server.

    No API key required — all inference happens on your own machine.
    Requires Ollama to be installed and running: https://ollama.ai

    base_url/model default to settings.ollama_base_url/settings.ollama_model
    (configurable via OLLAMA_BASE_URL / OLLAMA_MODEL in .env) so switching
    models doesn't require editing source code.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_model
        logger.info("OllamaClient initialised → %s, model: %s", self._base_url, self._model)

    def complete(self, prompt: str, response_mime_type: str | None = None) -> LLMResponse:
        """
        Send a prompt to the local Ollama server.
        We use the /api/generate endpoint with stream=False.
        """
        logger.debug("Ollama prompt (first 100 chars): %s...", prompt[:100])

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,        # get full response at once, not streamed
            "options": {
                "temperature": 0.0,  # deterministic
            },
        }

        try:
            # httpx is a modern HTTP client — similar to requests but async-capable
            with httpx.Client(timeout=settings.ollama_request_timeout_seconds) as client:
                response = client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()  # raises if HTTP status >= 400
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                f"Could not connect to Ollama at {self._base_url}. "
                "Is `ollama serve` running?",
                provider="ollama",
                status_code=503,
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMProviderError(
                f"Ollama request timed out after {settings.ollama_request_timeout_seconds}s.",
                provider="ollama",
                status_code=504,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}",
                provider="ollama",
                status_code=502,
            ) from exc

        data = response.json()
        text = data.get("response", "").strip()

        if not text:
            raise LLMProviderError(
                "Ollama returned an empty completion.",
                provider="ollama",
                status_code=502,
            )

        logger.debug("Ollama response: %s...", text[:80])

        return LLMResponse(
            text=text,
            provider="ollama",
            model=self._model,
            tokens_used=None,   # Ollama doesn't always report token counts
        )

    def health_check(self) -> bool:
        """Check if the Ollama server is up and responding."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False
