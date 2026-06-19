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

import logging

import httpx

from app.core.llm.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Default Ollama server location (same machine)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"   # change to any model you have pulled


class OllamaClient(LLMClient):
    """
    LLM client backed by a locally running Ollama server.
    
    No API key required — all inference happens on your own machine.
    Requires Ollama to be installed and running: https://ollama.ai
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        logger.info("OllamaClient initialised → %s, model: %s", base_url, model)

    def complete(self, prompt: str) -> LLMResponse:
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

        # httpx is a modern HTTP client — similar to requests but async-capable
        with httpx.Client(timeout=120.0) as client:  # 2 min timeout
            response = client.post(
                f"{self._base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()  # raises if HTTP status >= 400

        data = response.json()
        text = data.get("response", "").strip()

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
