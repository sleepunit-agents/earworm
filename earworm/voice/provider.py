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


class OllamaModelNotFoundError(ValueError):
    """Raised when the requested Ollama model is not installed."""

    def __init__(self, model: str, available: list[str], host: str):
        self.model = model
        self.available = available
        self.host = host
        available_str = ", ".join(available) if available else "(none)"
        super().__init__(
            f"Ollama model '{model}' not found on {host}. "
            f"Available models: {available_str}. "
            f"Install it with: ollama pull {model}"
        )


class OllamaConnectionError(ConnectionError):
    """Raised when we can't reach the Ollama server."""

    def __init__(self, host: str, cause: Exception):
        self.host = host
        super().__init__(f"Cannot connect to Ollama at {host}: {cause}")


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


_OLLAMA_DEFAULT_MODEL = "llama3.1:8b"


class OllamaProvider:
    """Ollama local model provider.

    Model resolution order:
    1. Explicit `model` parameter (from CLI --model flag)
    2. EARWORM_OLLAMA_MODEL environment variable
    3. Built-in default (llama3.1:8b)
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
    ):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = (
            model
            or os.environ.get("EARWORM_OLLAMA_MODEL")
            or _OLLAMA_DEFAULT_MODEL
        )
        self._validated = False

    def list_models(self) -> list[str]:
        """Query Ollama for installed models."""
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise OllamaConnectionError(self.host, e) from e
        except httpx.HTTPStatusError as e:
            raise OllamaConnectionError(self.host, e) from e

        data = response.json()
        return [m["name"] for m in data.get("models", [])]

    def validate_model(self) -> None:
        """Check that the configured model is installed on the Ollama server.

        Raises OllamaModelNotFoundError if not found.
        Raises OllamaConnectionError if the server is unreachable.
        """
        if self._validated:
            return

        available = self.list_models()
        # Ollama model names can be "qwen3:14b" or "qwen3:14b" — match with
        # and without the :latest tag
        normalized = set()
        for name in available:
            normalized.add(name)
            if ":" in name:
                normalized.add(name.split(":")[0])
            else:
                normalized.add(f"{name}:latest")

        if self.model not in normalized:
            raise OllamaModelNotFoundError(self.model, available, self.host)

        self._validated = True

    def generate(self, system: str, prompt: str) -> str:
        self.validate_model()

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
