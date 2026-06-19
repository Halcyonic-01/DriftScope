"""
app/core/llm/__init__.py

LLM client package.
Import get_client() from here to get the right LLM for a given provider.
"""

from app.core.llm.factory import get_client
from app.core.llm.base import LLMClient, LLMResponse

__all__ = ["get_client", "LLMClient", "LLMResponse"]
