"""
app/llm/errors.py
─────────────────
Provider-independent LLM error types.
"""

from __future__ import annotations


class LLMConfigurationError(Exception):
    """Raised when the LLM provider is misconfigured (unknown provider, missing API key, …)."""
