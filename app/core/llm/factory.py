"""
app/core/llm/factory.py

The Factory — the single place that decides which LLM client to build.

FACTORY PATTERN explained:
  Instead of doing this everywhere in your code:
    
    if provider == "gemini":
        client = GeminiClient()
    elif provider == "ollama":
        client = OllamaClient()

  You do this ONCE here, and everywhere else just calls:
    
    client = get_client("gemini")
    response = client.complete(prompt)

  Benefits:
    1. Adding a new LLM (e.g. OpenAI) = edit one file only
    2. Tests can request "mock" without changing business logic
    3. Provider selection can come from .env, making it configurable
"""

import logging

from app.core.llm.base import LLMClient

logger = logging.getLogger(__name__)

# Registry of all available providers
# Key = provider name string, Value = callable that builds the client
_REGISTRY: dict[str, type] = {}


def _register() -> None:
    """
    Populate the registry lazily.
    We import inside the function to avoid circular imports and
    to skip importing heavy dependencies (like google-generativeai)
    unless that provider is actually requested.
    """
    global _REGISTRY
    if _REGISTRY:
        return  # already registered

    from app.core.llm.gemini_client import GeminiClient
    from app.core.llm.ollama_client import OllamaClient
    from app.core.llm.mock_client import MockLLMClient

    _REGISTRY = {
        "gemini": GeminiClient,
        "ollama": OllamaClient,
        "mock":   MockLLMClient,
    }


def get_client(provider: str = "gemini") -> LLMClient:
    """
    Factory function — returns the appropriate LLM client instance.

    Args:
        provider: One of "gemini", "ollama", "mock".
                  Defaults to "gemini".

    Returns:
        A fully initialised LLMClient ready to call .complete()

    Raises:
        ValueError: If the provider name is not recognised.

    Examples:
        client = get_client("gemini")
        response = client.complete("Explain gravity in one sentence.")
        print(response.text)
    """
    _register()

    provider = provider.lower().strip()

    if provider not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Available providers: {available}"
        )

    logger.info("Creating LLM client for provider: %s", provider)
    return _REGISTRY[provider]()


def list_providers() -> list[str]:
    """Return all registered provider names."""
    _register()
    return list(_REGISTRY.keys())
