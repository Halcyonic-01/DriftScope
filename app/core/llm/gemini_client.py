"""
app/core/llm/gemini_client.py

Gemini LLM client using Google's generativeai SDK.
We use gemini-2.5-flash — fast, available, and good for judge calls.

HOW GEMINI WORKS:
  1. We configure the SDK with our API key from .env
  2. We create a GenerativeModel pointing to "gemini-2.5-flash"
  3. We call model.generate_content(prompt) to get a response
  4. We wrap it in our standard LLMResponse and return it
"""

from __future__ import annotations

import logging

from google.api_core import exceptions as google_exceptions
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.core.config import settings
from app.core.llm.base import LLMClient, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)

# The model we use — stable Gemini 2.5 Flash model code
GEMINI_MODEL = "gemini-2.5-flash"


class GeminiClient(LLMClient):
    """
    LLM client backed by Google Gemini 2.5 Flash.
    
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
        self._model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        logger.info("GeminiClient initialised with model: %s", GEMINI_MODEL)

    def complete(
        self,
        prompt: str,
        response_mime_type: str | None = None,
    ) -> LLMResponse:
        """
        Send a prompt to Gemini and return the response.

        temperature=0.0 means the same prompt will always return the same
        response — crucial for reproducibility in evaluations.
        """
        logger.debug("Gemini prompt (first 100 chars): %s...", prompt[:100])

        max_output_tokens = (
            min(settings.gemini_max_output_tokens, 2048)
            if response_mime_type == "application/json"
            else settings.gemini_max_output_tokens
        )
        generation_config = {
            "temperature": 0.0,    # deterministic for reproducible evals
            "max_output_tokens": max_output_tokens,
        }
        if response_mime_type:
            generation_config["response_mime_type"] = response_mime_type

        try:
            response = self._generate_with_truncation_retry(
                prompt=prompt,
                generation_config=generation_config,
            )
        except google_exceptions.ResourceExhausted as exc:
            raise LLMProviderError(
                (
                    "Gemini quota or rate limit was exhausted for this API key/project. "
                    "Use provider='mock' for local testing, wait for quota reset, or "
                    "enable/request quota for the Gemini API project."
                ),
                provider="gemini",
                status_code=503,
                retry_after_seconds=_retry_after_seconds(exc),
            ) from exc
        except google_exceptions.GoogleAPIError as exc:
            raise LLMProviderError(
                f"Gemini API request failed: {exc}",
                provider="gemini",
                status_code=502,
            ) from exc

        finish_reason = _finish_reason(response)
        text = _extract_text(response)

        if finish_reason in {"MAX_TOKENS", "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"}:
            raise LLMProviderError(
                f"Gemini returned no usable completion. finish_reason={finish_reason}",
                provider="gemini",
                status_code=502,
            )

        if not text:
            raise LLMProviderError(
                f"Gemini returned an empty completion. finish_reason={finish_reason or 'unknown'}",
                provider="gemini",
                status_code=502,
            )

        # Extract token usage if available
        tokens_used = None
        if response.usage_metadata:
            tokens_used = (
                response.usage_metadata.prompt_token_count
                + response.usage_metadata.candidates_token_count
            )

        logger.debug(
            "Gemini response (%d tokens, finish=%s): %s...",
            tokens_used or 0,
            finish_reason or "unknown",
            text[:80],
        )

        return LLMResponse(
            text=text,
            provider="gemini",
            model=GEMINI_MODEL,
            tokens_used=tokens_used,
        )

    def _generate_with_truncation_retry(
        self,
        prompt: str,
        generation_config: dict,
    ):
        response = self._model.generate_content(
            prompt,
            generation_config=GenerationConfig(**generation_config),
        )
        finish_reason = _finish_reason(response)

        if (
            finish_reason == "MAX_TOKENS"
            and settings.gemini_retry_max_output_tokens > generation_config["max_output_tokens"]
        ):
            retry_config = {
                **generation_config,
                "max_output_tokens": settings.gemini_retry_max_output_tokens,
            }
            logger.warning(
                "Gemini response hit MAX_TOKENS at %s tokens; retrying with %s.",
                generation_config["max_output_tokens"],
                retry_config["max_output_tokens"],
            )
            response = self._model.generate_content(
                prompt,
                generation_config=GenerationConfig(**retry_config),
            )

        return response

    def health_check(self) -> bool:
        """Ping Gemini with a minimal prompt to verify connectivity."""
        try:
            resp = self._model.generate_content("Reply with the word OK only.")
            return "ok" in resp.text.lower()
        except Exception as exc:
            logger.warning("Gemini health check failed: %s", exc)
            return False


def _retry_after_seconds(exc: Exception) -> int | None:
    retry_delay = getattr(exc, "retry_delay", None)
    if retry_delay is None:
        return None
    seconds = getattr(retry_delay, "seconds", None)
    return int(seconds) if seconds is not None else None


def _extract_text(response) -> str:
    try:
        return response.text.strip()
    except Exception:
        pass

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "".join(parts).strip()


def _finish_reason(response) -> str | None:
    candidates = getattr(response, "candidates", []) or []
    if not candidates:
        return None

    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        return None

    name = getattr(finish_reason, "name", None)
    if name:
        return str(name)

    value = getattr(finish_reason, "value", finish_reason)
    known_values = {
        1: "STOP",
        2: "MAX_TOKENS",
        3: "SAFETY",
        4: "RECITATION",
        5: "OTHER",
        6: "BLOCKLIST",
        7: "PROHIBITED_CONTENT",
    }
    return known_values.get(value, str(value))
