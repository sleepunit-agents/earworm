"""LLM provider abstraction for Voice interpretation.

Supports Anthropic (Claude) and Ollama (local models).
Provider selection via EARWORM_LLM_PROVIDER env var.
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx


class LLMProvider(Protocol):
    """Protocol for LLM providers that Voice can use."""

    def generate(self, system: str, prompt: str) -> str:
        """Generate a text response from the LLM."""
        ...


class AnthropicProvider:
    """Anthropic Claude API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable required for Anthropic provider"
            )

    def generate(self, system: str, prompt: str) -> str:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 2048,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]


class OllamaProvider:
    """Ollama local model provider."""

    def __init__(
        self,
        host: str | None = None,
        model: str = "llama3.1:8b",
    ):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model

    def generate(self, system: str, prompt: str) -> str:
        response = httpx.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["response"]


def get_provider(
    provider_name: str | None = None,
    **kwargs,
) -> LLMProvider:
    """Get an LLM provider by name or from EARWORM_LLM_PROVIDER env var.

    Args:
        provider_name: "anthropic" or "ollama". Defaults to env var.
        **kwargs: Passed to the provider constructor.
    """
    name = provider_name or os.environ.get("EARWORM_LLM_PROVIDER", "ollama")

    if name == "anthropic":
        return AnthropicProvider(**kwargs)
    elif name == "ollama":
        return OllamaProvider(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {name}. Use 'anthropic' or 'ollama'.")
