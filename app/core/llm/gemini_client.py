"""
app/core/llm/gemini_client.py

Gemini LLM client using Google's generativeai SDK.
The model is configurable via GEMINI_MODEL (see app/core/config.py).

HOW GEMINI WORKS:
  1. We configure the SDK with our API key from .env
  2. We create a GenerativeModel pointing to settings.gemini_model
  3. We call model.generate_content(prompt) to get a response
  4. We wrap it in our standard LLMResponse and return it
"""

from __future__ import annotations

import logging
import time

from google.api_core import exceptions as google_exceptions
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.core.config import settings
from app.core.instrumentation import track_llm_call
from app.core.llm.base import LLMClient, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """
    LLM client backed by Google Gemini.

    Requires GEMINI_API_KEY to be set in your .env file.
    Get a free key at: https://aistudio.google.com

    The model comes from settings.gemini_model (GEMINI_MODEL in .env), so
    a retired model ID can be swapped without a code change.
    """

    def __init__(self, model: str | None = None) -> None:
        if not settings.gemini_api_key or settings.gemini_api_key.startswith("AIza..."):
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file. "
                "Get a free key at https://aistudio.google.com"
            )
        # Configure the SDK globally (one-time setup)
        genai.configure(api_key=settings.gemini_api_key)

        self._model_name = model or settings.gemini_model
        self._model = genai.GenerativeModel(model_name=self._model_name)
        logger.info("GeminiClient initialised with model: %s", self._model_name)

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
            response = self._generate_with_rate_limit_retry(
                prompt=prompt,
                generation_config=generation_config,
            )
        except google_exceptions.ResourceExhausted as exc:
            wait = _retry_after_seconds(exc)
            raise LLMProviderError(
                (
                    "Gemini rate limit or quota still exhausted after "
                    f"{settings.gemini_max_retries} attempts. Free-tier limits are "
                    "per-minute, and one eval case is a burst of calls (a generation "
                    "plus one judge call per rule). Reduce the golden-case set, raise "
                    "--delay, or use provider='mock' for local testing."
                ),
                provider="gemini",
                status_code=503,
                retry_after_seconds=wait,
            ) from exc
        except google_exceptions.NotFound as exc:
            # Google retires model IDs for new API keys. The raw 404 doesn't
            # say where the model name came from, which makes this slow to
            # diagnose — point straight at the setting to change.
            raise LLMProviderError(
                f"Gemini model '{self._model_name}' is unavailable for this API key: {exc}. "
                "Set GEMINI_MODEL in your environment to a current model. "
                "Note that a model can be returned by list_models() and still "
                "reject generateContent, so verify with a real call.",
                provider="gemini",
                status_code=502,
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
            model=self._model_name,
            tokens_used=tokens_used,
        )

    def _generate_with_rate_limit_retry(
        self,
        prompt: str,
        generation_config: dict,
    ):
        """
        Retry on 429 using the delay the API itself supplies.

        Spacing calls with a fixed sleep cannot solve this: a single case
        issues a generation call plus one judge call per rule, all
        back-to-back, so the burst breaches a per-minute limit no matter
        how far apart the cases are. Google returns a retry_delay on the
        429 — honouring it is what actually lets a run complete.
        """
        last_exc: Exception | None = None

        for attempt in range(1, settings.gemini_max_retries + 1):
            try:
                with track_llm_call("gemini"):
                    return self._generate_with_truncation_retry(
                        prompt=prompt,
                        generation_config=generation_config,
                    )
            except google_exceptions.ResourceExhausted as exc:
                last_exc = exc
                if attempt >= settings.gemini_max_retries:
                    break

                # Prefer the server's own retry_delay; fall back to
                # exponential backoff when it isn't supplied.
                wait = _retry_after_seconds(exc) or (2 ** attempt)
                wait = min(wait, settings.gemini_max_retry_wait_seconds)
                logger.warning(
                    "Gemini rate limited (attempt %d/%d). Waiting %ss before retry.",
                    attempt, settings.gemini_max_retries, wait,
                )
                time.sleep(wait)

        assert last_exc is not None
        raise last_exc

    def _generate_with_truncation_retry(
        self,
        prompt: str,
        generation_config: dict,
    ):
        response = self._model.generate_content(
            prompt,
            generation_config=GenerationConfig(**generation_config),
            request_options={"timeout": settings.gemini_request_timeout_seconds},
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
                request_options={"timeout": settings.gemini_request_timeout_seconds},
            )

        return response

    def health_check(self) -> bool:
        """Ping Gemini with a minimal prompt to verify connectivity."""
        try:
            resp = self._model.generate_content(
                "Reply with the word OK only.",
                request_options={"timeout": settings.gemini_request_timeout_seconds},
            )
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
