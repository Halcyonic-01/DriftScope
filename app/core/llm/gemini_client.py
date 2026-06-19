"""
app/core/llm/gemini_client.py

Gemini LLM client using Google's generativeai SDK.
We use gemini-2.0-flash — fast, free tier, good quality.

HOW GEMINI WORKS:
  1. We configure the SDK with our API key from .env
  2. We create a GenerativeModel pointing to "gemini-2.0-flash"
  3. We call model.generate_content(prompt) to get a response
  4. We wrap it in our standard LLMResponse and return it
"""

import logging

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.core.config import settings
from app.core.llm.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# The model we use — gemini-2.0-flash is fast and free
GEMINI_MODEL = "gemini-2.0-flash"


class GeminiClient(LLMClient):
    """
    LLM client backed by Google Gemini 2.0 Flash.
    
    Requires GEMINI_API_KEY to be set in your .env file.
    Get a free key at: https://aistudio.google.com
    """

    def __init__(self) -> None:
        if not settings.gemini_api_key or settings.gemini_api_key.startswith("AIza..."):
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file. "
                "Get a free key at https://aistudio.google.com"
            )
        # Configure the SDK globally (one-time setup)
        genai.configure(api_key=settings.gemini_api_key)

        # Create the model instance
        self._model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=GenerationConfig(
                temperature=0.0,    # 0.0 = deterministic, no randomness
                                    # Important for consistent evals!
                max_output_tokens=1024,
            ),
        )
        logger.info("GeminiClient initialised with model: %s", GEMINI_MODEL)

    def complete(self, prompt: str) -> LLMResponse:
        """
        Send a prompt to Gemini and return the response.

        temperature=0.0 means the same prompt will always return the same
        response — crucial for reproducibility in evaluations.
        """
        logger.debug("Gemini prompt (first 100 chars): %s...", prompt[:100])

        response = self._model.generate_content(prompt)

        # Extract text from the response object
        text = response.text.strip()

        # Extract token usage if available
        tokens_used = None
        if response.usage_metadata:
            tokens_used = (
                response.usage_metadata.prompt_token_count
                + response.usage_metadata.candidates_token_count
            )

        logger.debug("Gemini response (%d tokens): %s...", tokens_used or 0, text[:80])

        return LLMResponse(
            text=text,
            provider="gemini",
            model=GEMINI_MODEL,
            tokens_used=tokens_used,
        )

    def health_check(self) -> bool:
        """Ping Gemini with a minimal prompt to verify connectivity."""
        try:
            resp = self._model.generate_content("Reply with the word OK only.")
            return "ok" in resp.text.lower()
        except Exception as exc:
            logger.warning("Gemini health check failed: %s", exc)
            return False
