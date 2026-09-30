"""
app/llm/nvidia_client.py
────────────────────────
NVIDIA NIM backend for BOB, using NVIDIA's hosted OpenAI-compatible API
(POST https://integrate.api.nvidia.com/v1/chat/completions).

This is NVIDIA's hosted API endpoint for development and testing; access,
quotas and rate limits are determined by NVIDIA and your NVIDIA account.

Exposes the same interface as OllamaClient so JarvisAssistant needs no
provider-specific logic:

  chat()            – text-in / text-out.
  chat_with_tools() – sends tool definitions; returns an Ollama-style message
                      dict that may contain ``tool_calls`` for the agentic loop.

Environment variables:
  NVIDIA_API_KEY  – required. Never logged or included in error messages.
  NVIDIA_MODEL    – model name              (default: openai/gpt-oss-20b)
  NVIDIA_TIMEOUT  – request timeout seconds (default: 120)

SECURITY MODEL
──────────────
• The model only receives *function* tool definitions taken from BOB's
  ToolRegistry schemas. It returns tool-call requests; BOB's agentic loop
  validates them and executes the tools through the ToolRegistry.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

from app.llm.errors import LLMConfigurationError

logger = logging.getLogger(__name__)


# ─── Public exception types ───────────────────────────────────────────────────

class NvidiaError(Exception):
    """Base class for all NVIDIA client errors."""


class NvidiaConfigurationError(NvidiaError, LLMConfigurationError):
    """Raised when the NVIDIA client is misconfigured (e.g. missing API key)."""


class NvidiaConnectionError(NvidiaError):
    """Raised when the NVIDIA API cannot be reached or the request times out."""


class NvidiaResponseError(NvidiaError):
    """Raised when the NVIDIA API returns an error or an unusable response."""


class NvidiaAuthenticationError(NvidiaResponseError):
    """Raised when NVIDIA rejects the API key (HTTP 401/403)."""


class NvidiaAccessError(NvidiaResponseError):
    """Raised for payment/credit/access problems (HTTP 402)."""


class NvidiaRateLimitError(NvidiaResponseError):
    """Raised when the NVIDIA rate limit is exceeded (HTTP 429)."""


# Ollama-style option names → chat-completions request fields.
_OPTION_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "num_predict": "max_tokens",
    "max_tokens": "max_tokens",
}


# ─── Schema / history conversion (pure functions) ─────────────────────────────

def convert_tool_schemas(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalise ToolRegistry schemas into OpenAI chat-completions function tools::

        {"type": "function", "function": {"name", "description", "parameters"}}

    The registry already uses this format; entries are validated and passed
    through so the registry stays the single source of truth.
    """
    converted: List[Dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            logger.warning("Skipping tool schema without a name.")
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict) or not params:
            params = {"type": "object", "properties": {}}
        converted.append({
            "type": "function",
            "function": {"name": name, "description": fn.get("description") or "", "parameters": params},
        })
    return converted


def _arguments_json(raw: Any) -> str:
    if isinstance(raw, str):
        return raw or "{}"
    return json.dumps(raw if isinstance(raw, dict) else {}, default=str)


def convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert BOB conversation history into chat-completions messages.

    BOB's tool messages carry no id, so each is paired, in order, with the
    tool calls of the preceding assistant message and given its tool_call_id.
    """
    converted: List[Dict[str, Any]] = []
    pending_ids: List[str] = []
    generated = 0

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "tool":
            output = content if isinstance(content, str) else json.dumps(content, default=str)
            if pending_ids:
                converted.append({"role": "tool", "tool_call_id": pending_ids.pop(0), "content": output})
            else:
                # Orphan result (no matching call): keep the information as plain context.
                converted.append({"role": "user", "content": f"Tool result: {output}"})
            continue

        if role == "assistant":
            pending_ids = []
            calls: List[Dict[str, Any]] = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = str(fn.get("name") or "").strip()
                if not name:
                    continue
                call_id = str(tc.get("id") or "").strip()
                if not call_id:
                    generated += 1
                    call_id = f"call_bob_{generated}"
                calls.append({"id": call_id, "type": "function",
                              "function": {"name": name, "arguments": _arguments_json(fn.get("arguments"))}})
                pending_ids.append(call_id)
            if calls:
                converted.append({"role": "assistant", "content": str(content or ""), "tool_calls": calls})
            elif content:
                converted.append({"role": "assistant", "content": str(content)})
            continue

        if role in ("system", "user") and content:
            converted.append({"role": role, "content": str(content)})

    return converted


def normalize_response(data: Any) -> Dict[str, Any]:
    """
    Convert a chat-completions response body into an Ollama-style message dict::

        {"role": "assistant", "content": "...",
         "tool_calls": [{"id": "...", "function": {"name": "...", "arguments": {...}}}]}

    ``tool_calls`` is omitted when the model made no tool calls. Any separate
    ``reasoning_content`` is never part of the reply.

    :raises NvidiaResponseError: malformed, truncated or empty responses.
    """
    if not isinstance(data, dict):
        raise NvidiaResponseError("Unexpected NVIDIA response format.")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise NvidiaResponseError("Unexpected NVIDIA response format (no choices).")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise NvidiaResponseError("Unexpected NVIDIA response format (no message).")

    text = message.get("content") if isinstance(message.get("content"), str) else ""
    tool_calls: List[Dict[str, Any]] = []
    malformed = 0

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict):
            malformed += 1
            continue
        name = str(fn.get("name") or "").strip()
        call_id = str(tc.get("id") or "").strip()
        raw_args = fn.get("arguments")
        try:
            if isinstance(raw_args, str):
                args = json.loads(raw_args) if raw_args.strip() else {}
            else:
                args = raw_args if raw_args is not None else {}
        except json.JSONDecodeError:
            args = None
        if not name or not call_id or not isinstance(args, dict):
            malformed += 1
            logger.warning("Ignoring malformed NVIDIA tool call (name=%r).", name)
            continue
        tool_calls.append({"id": call_id, "function": {"name": name, "arguments": args}})

    if tool_calls:
        return {"role": "assistant", "content": text, "tool_calls": tool_calls}
    if malformed:
        raise NvidiaResponseError(
            "The NVIDIA model produced a malformed tool call. Please try rephrasing your request."
        )
    if not text.strip():
        reason = choice.get("finish_reason")
        hint = " (the max_tokens limit was reached)" if reason == "length" else ""
        raise NvidiaResponseError(f"NVIDIA returned an empty response (finish_reason={reason}){hint}.")
    return {"role": "assistant", "content": text}


# ─── Client ───────────────────────────────────────────────────────────────────

class NvidiaClient:
    """
    NVIDIA NIM client with the same interface as OllamaClient.

    Usage::

        client = NvidiaClient()                     # reads NVIDIA_API_KEY
        reply = client.chat([{"role": "user", "content": "Hi!"}])
    """

    DEFAULT_MODEL: str = "openai/gpt-oss-20b"
    # GLM-5.3 Flash remains available as a secondary model selectable via
    # NVIDIA_MODEL=z-ai/glm-5.3-flash (e.g. for future Laya routing).
    SECONDARY_MODEL: str = "z-ai/glm-5.3-flash"
    DEFAULT_TIMEOUT: int = 120
    # NVIDIA's API defaults max_tokens to 1024; reasoning models can use most of
    # that before answering, so request a larger budget unless overridden.
    DEFAULT_MAX_TOKENS: int = 4096
    API_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"

    # Shown in the CLI banner next to "Host".
    host: str = "NVIDIA NIM API (integrate.api.nvidia.com)"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        """
        :param api_key: NVIDIA API key (default: NVIDIA_API_KEY env var).
        :param model:   Model name (default: NVIDIA_MODEL env var or openai/gpt-oss-20b).
        :param timeout: Request timeout in seconds (default: NVIDIA_TIMEOUT or 120).
        :raises NvidiaConfigurationError: API key missing.
        """
        key = api_key if api_key is not None else os.environ.get("NVIDIA_API_KEY", "")
        key = (key or "").strip()
        if not key:
            raise NvidiaConfigurationError(
                "NVIDIA_API_KEY is required when LLM_PROVIDER=nvidia (the default provider). "
                "Set it in your environment or .env file, or set LLM_PROVIDER=ollama "
                "to use local Ollama instead."
            )
        self.__api_key = key
        self.model = model or os.environ.get("NVIDIA_MODEL") or self.DEFAULT_MODEL
        try:
            self.timeout = int(timeout or os.environ.get("NVIDIA_TIMEOUT") or self.DEFAULT_TIMEOUT)
        except ValueError:
            self.timeout = self.DEFAULT_TIMEOUT
        logger.debug("NvidiaClient initialised: model=%s", self.model)

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str | None = None,
        options: Dict[str, Any] | None = None,
    ) -> str:
        """
        Send a conversation and return the reply text.

        :raises NvidiaConnectionError: API unreachable / timed out.
        :raises NvidiaResponseError:   API error or unusable response.
        """
        message = self._generate(messages, tools=None, model=model, options=options)
        return message.get("content") or ""

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        model: str | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Send a conversation with function tools and return an Ollama-style
        assistant message dict (see ``normalize_response``).

        The model never executes tools itself: returned ``tool_calls`` are run
        by BOB's agentic loop through the ToolRegistry.

        :raises NvidiaConnectionError: API unreachable / timed out.
        :raises NvidiaResponseError:   API error or unusable response.
        """
        return self._generate(messages, tools=tools, model=model, options=options)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        model: Optional[str],
        options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        model_name = model or self.model
        converted = convert_messages(messages)
        if not converted:
            raise NvidiaResponseError("No user message to send to NVIDIA.")

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": converted,
            "max_tokens": self.DEFAULT_MAX_TOKENS,
            "stream": False,
        }
        function_tools = convert_tool_schemas(tools or [])
        if function_tools:
            payload["tools"] = function_tools
        for key, value in (options or {}).items():
            field = _OPTION_MAP.get(key)
            if field:
                payload[field] = value

        logger.debug("POST %s model=%s messages=%d tools=%d",
                     self.API_URL, model_name, len(converted), len(function_tools))
        data = self._post(payload, model_name)
        message = normalize_response(data)
        logger.debug(
            "NVIDIA replied: content=%d chars, tool_calls=%d",
            len(message.get("content") or ""), len(message.get("tool_calls") or []),
        )
        return message

    def _post(self, payload: Dict[str, Any], model_name: str) -> Any:
        """POST to the chat-completions endpoint, translating failures into Nvidia* errors."""
        headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            resp = requests.post(self.API_URL, json=payload, headers=headers, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise NvidiaConnectionError(f"Request to NVIDIA timed out after {self.timeout}s.") from None
        except requests.exceptions.RequestException as exc:
            logger.debug("NVIDIA network error: %s", self._redact(repr(exc)))
            raise NvidiaConnectionError(
                "Cannot connect to the NVIDIA API. Check your internet connection."
            ) from None

        if not resp.ok:
            raise self._http_error(resp, model_name)
        try:
            return resp.json()
        except ValueError:
            raise NvidiaResponseError("NVIDIA returned a non-JSON response.") from None

    def _http_error(self, resp: Any, model_name: str) -> NvidiaError:
        code = resp.status_code
        detail = self._error_detail(resp)
        logger.debug("NVIDIA HTTP %s: %s", code, detail)

        if code in (401, 403):
            return NvidiaAuthenticationError(
                f"NVIDIA rejected the API key (HTTP {code}). Check NVIDIA_API_KEY."
            )
        if code == 402:
            return NvidiaAccessError(
                f"NVIDIA API access or credits problem (HTTP 402): {detail}. "
                "Check your NVIDIA account."
            )
        if code == 429:
            return NvidiaRateLimitError(
                "NVIDIA rate limit exceeded (HTTP 429). Wait a moment and try again."
            )
        if code == 404:
            return NvidiaResponseError(
                f"NVIDIA model {model_name!r} or endpoint not found (HTTP 404). Check NVIDIA_MODEL."
            )
        if code >= 500:
            return NvidiaResponseError(f"NVIDIA server error (HTTP {code}): {detail}")
        return NvidiaResponseError(f"NVIDIA API error (HTTP {code}): {detail}")

    def _error_detail(self, resp: Any) -> str:
        try:
            body = resp.json()
        except ValueError:
            body = None
        detail: Any = None
        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, list):   # FastAPI-style validation errors (HTTP 422)
                detail = "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
            if not detail:
                err = body.get("error")
                detail = err.get("message") if isinstance(err, dict) else err
            detail = detail or body.get("message") or body.get("title")
        if not detail:
            detail = getattr(resp, "text", "") or "unknown error"
        return self._redact(str(detail))[:300]

    def _redact(self, text: str) -> str:
        """Remove the API key (and any bearer token) from text."""
        redacted = text.replace(self.__api_key, "***") if self.__api_key else text
        return re.sub(r"(Bearer\s+)\S+", r"\1***", redacted)

    def __repr__(self) -> str:  # pragma: no cover
        return f"NvidiaClient(model={self.model!r})"
