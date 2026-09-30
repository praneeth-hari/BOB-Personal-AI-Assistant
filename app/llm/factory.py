"""
app/llm/factory.py
──────────────────
Selects the LLM backend from the LLM_PROVIDER environment variable.

  LLM_PROVIDER=nvidia  (default) → NvidiaClient (NVIDIA NIM API, requires NVIDIA_API_KEY)
  LLM_PROVIDER=ollama            → OllamaClient (local alternative, no API key)

Selection is deterministic: a misconfigured provider raises
LLMConfigurationError instead of silently falling back to another one.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from app.llm.errors import LLMConfigurationError

DEFAULT_PROVIDER = "nvidia"
SUPPORTED_PROVIDERS = ("nvidia", "ollama")


def get_provider_name(provider: Optional[str] = None) -> str:
    """Return the normalised provider name (explicit argument, env var, or default)."""
    raw = provider if provider is not None else os.environ.get("LLM_PROVIDER", "")
    return (raw or DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER


def create_llm_client(provider: Optional[str] = None) -> Any:
    """
    Build the LLM client for the configured provider.

    :param provider: Override LLM_PROVIDER (mainly for tests).
    :raises LLMConfigurationError: unknown provider, or NVIDIA without an API key.
    """
    name = get_provider_name(provider)

    if name == "ollama":
        from app.llm.ollama_client import OllamaClient
        return OllamaClient()

    if name == "nvidia":
        from app.llm.nvidia_client import NvidiaClient
        return NvidiaClient()  # raises NvidiaConfigurationError if the key is missing

    raise LLMConfigurationError(
        f"Unknown LLM_PROVIDER {name!r}. Supported values: {', '.join(SUPPORTED_PROVIDERS)}."
    )
