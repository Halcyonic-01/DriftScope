"""
app/core/llm/base.py

Abstract base class for all LLM clients.

WHY an abstract base class?
Think of it like a contract. Every LLM client MUST implement:
  - complete(prompt) → LLMResponse

This means any code that calls .complete() doesn't care whether
it's talking to Gemini, Ollama, or any other LLM.
That's the power of abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """
    Standardized response object from any LLM.
    We always return this regardless of which LLM was used.

    Attributes:
        text:     The raw text the LLM returned.
        provider: Which LLM generated this (e.g. "gemini", "ollama").
        model:    The specific model name (e.g. "gemini-2.0-flash").
        tokens_used: Total tokens consumed (prompt + completion). None if unknown.
    """
    text: str
    provider: str
    model: str
    tokens_used: Optional[int] = None


class LLMClient(ABC):
    """
    Abstract base class — all LLM clients must inherit from this.
    
    ABC = Abstract Base Class
    @abstractmethod = subclass MUST implement this method or Python
                      raises a TypeError at runtime (a safety net).
    """

    @abstractmethod
    def complete(self, prompt: str) -> LLMResponse:
        """
        Send a prompt to the LLM and return its response.

        Args:
            prompt: The full prompt string to send.

        Returns:
            LLMResponse with the text and metadata.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify the LLM service is reachable and responding.
        Returns True if healthy, False otherwise.
        """
        ...
