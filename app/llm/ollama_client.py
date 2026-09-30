"""
app/llm/ollama_client.py
────────────────────────
Thin abstraction over the local Ollama HTTP API.

Environment variables (all optional):
  OLLAMA_HOST   – base URL of the Ollama server   (default: http://localhost:11434)
  OLLAMA_MODEL  – model tag to use                (default: qwen2.5-coder:7b)
  OLLAMA_TIMEOUT – request timeout in seconds     (default: 120)

Only the /api/chat endpoint is used.  No streaming – we wait for the full
response so the rest of BOB stays simple for now.

Two chat methods are provided:
  chat()            – simple text-in / text-out (used for plain conversation).
  chat_with_tools() – sends a tools list; returns the raw message dict which
                      may contain tool_calls for the agentic loop.
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any

import requests

logger = logging.getLogger(__name__)

# ─── Public exception types ───────────────────────────────────────────────────

class OllamaError(Exception):
    """Base class for all Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached."""


class OllamaResponseError(OllamaError):
    """Raised when the Ollama server returns an unexpected response."""


# ─── Client ───────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Minimal client for the local Ollama server.

    Usage::

        client = OllamaClient()
        response = client.chat(messages=[{"role": "user", "content": "Hi!"}])
        print(response)   # → "Hello! How can I help you today?"
    """

    DEFAULT_HOST: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "qwen2.5-coder:7b"
    DEFAULT_TIMEOUT: int = 120

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.timeout = timeout or int(os.environ.get("OLLAMA_TIMEOUT", self.DEFAULT_TIMEOUT))

        self._chat_url = f"{self.host}/api/chat"
        logger.debug("OllamaClient initialised: host=%s model=%s", self.host, self.model)

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str | None = None,
        options: Dict[str, Any] | None = None,
    ) -> str:
        """
        Send a conversation to Ollama and return the assistant reply as a string.

        :param messages:  List of ``{"role": ..., "content": ...}`` dicts.
        :param model:     Override the instance-level model for this call only.
        :param options:   Ollama model options (temperature, top_p, …).
        :returns:         The text content of the assistant message.
        :raises OllamaConnectionError: Server unreachable.
        :raises OllamaResponseError:   Unexpected/malformed response.
        """
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options

        logger.debug("POST %s  model=%s  messages=%d", self._chat_url, payload["model"], len(messages))

        try:
            resp = requests.post(self._chat_url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running (`ollama serve`)."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                f"Request to Ollama timed out after {self.timeout}s."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(f"Ollama request failed: {exc}") from exc

        if not resp.ok:
            raise OllamaResponseError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
            content: str = data["message"]["content"]
        except (ValueError, KeyError) as exc:
            raise OllamaResponseError(
                f"Unexpected Ollama response format: {resp.text[:200]}"
            ) from exc

        logger.debug("Ollama replied (%d chars)", len(content))
        return content

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        model: str | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Send a conversation to Ollama with a tools list and return the raw
        assistant message dict.

        When the model decides to call a tool, the returned dict looks like::

            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "...", "arguments": {...}}}
                ]
            }

        When the model gives a plain text answer (no tool call), the returned
        dict looks like::

            {"role": "assistant", "content": "Here is the answer…"}

        :param messages: Conversation history.
        :param tools:    List of tool schemas (OpenAI / Ollama format).
        :param model:    Override the instance-level model for this call only.
        :param options:  Ollama model options.
        :returns:        The ``message`` dict from the Ollama response.
        :raises OllamaConnectionError: Server unreachable.
        :raises OllamaResponseError:   Unexpected/malformed response.
        """
        payload: Dict[str, Any] = {
            "model":   model or self.model,
            "messages": messages,
            "tools":   tools,
            "stream":  False,
        }
        if options:
            payload["options"] = options

        logger.debug(
            "POST %s (with_tools) model=%s messages=%d tools=%d",
            self._chat_url, payload["model"], len(messages), len(tools),
        )

        try:
            resp = requests.post(self._chat_url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running (`ollama serve`)."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                f"Request to Ollama timed out after {self.timeout}s."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(f"Ollama request failed: {exc}") from exc

        if not resp.ok:
            raise OllamaResponseError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
            message: Dict[str, Any] = data["message"]
        except (ValueError, KeyError) as exc:
            raise OllamaResponseError(
                f"Unexpected Ollama response format: {resp.text[:200]}"
            ) from exc

        tool_calls = message.get("tool_calls") or []
        logger.debug(
            "Ollama replied: content=%d chars, tool_calls=%d",
            len(message.get("content") or ""),
            len(tool_calls),
        )
        return message

    def is_available(self) -> bool:
        """
        Quick health-check: return True if the Ollama server responds.
        Does NOT raise – safe to call during startup.
        """
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.ok
        except requests.exceptions.RequestException:
            return False

    def list_local_models(self) -> List[str]:
        """
        Return a list of locally available model tags.
        Returns an empty list if Ollama is unreachable.
        """
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            if not resp.ok:
                return []
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except (requests.exceptions.RequestException, ValueError, KeyError):
            return []

    def __repr__(self) -> str:  # pragma: no cover
        return f"OllamaClient(host={self.host!r}, model={self.model!r})"
