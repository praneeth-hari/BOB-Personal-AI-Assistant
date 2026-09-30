"""
tests/test_assistant_fallback.py
─────────────────────────────────
Unit tests for JarvisAssistant NVIDIA → Ollama fallback logic.
No real NVIDIA or Ollama connections are made.
"""
from __future__ import annotations

import pytest

from app.core.assistant import JarvisAssistant
from app.llm.nvidia_client import (
    NvidiaAccessError,
    NvidiaAuthenticationError,
    NvidiaConnectionError,
    NvidiaRateLimitError,
)


# ── Mock helpers ──────────────────────────────────────────────────────────────

class MockSuccessLLM:
    """Succeeds on every call; records the model= kwarg."""

    def __init__(self, reply: str = "primary OK") -> None:
        self.reply = reply
        self.model_kwargs: list = []

    def chat(self, messages, *, model=None, **kwargs):
        self.model_kwargs.append(model)
        return self.reply


class MockRaisingLLM:
    """Always raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def chat(self, messages, **kwargs):
        raise self._exc


class MockFallbackWithAvailable:
    """Fallback mock that exposes is_available(); availability is togglable."""

    def __init__(self, *, available: bool = True, reply: str = "fallback OK") -> None:
        self._available = available
        self.reply = reply
        self.model_kwargs: list = []

    def is_available(self) -> bool:
        return self._available

    def chat(self, messages, *, model=None, **kwargs):
        self.model_kwargs.append(model)
        return self.reply


class MockFallbackRaising:
    """Fallback that itself raises."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def chat(self, messages, **kwargs):
        raise self._exc


# ── NVIDIA success — fallback never used ─────────────────────────────────────

class TestFallbackNoop:
    def test_nvidia_success_fallback_not_called(self):
        primary = MockSuccessLLM(reply="primary response")
        fallback = MockFallbackWithAvailable()
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        reply = bob.chat("Hello")
        assert reply == "primary response"
        assert fallback.model_kwargs == []  # fallback never touched

    def test_no_fallback_configured_success(self):
        primary = MockSuccessLLM(reply="no fallback needed")
        bob = JarvisAssistant(llm_client=primary)
        assert bob.chat("Hi") == "no fallback needed"


# ── Transient NVIDIA errors trigger fallback ──────────────────────────────────

class TestFallbackTriggers:
    def test_connection_error_triggers_fallback(self):
        primary = MockRaisingLLM(NvidiaConnectionError("timeout"))
        fallback = MockSuccessLLM(reply="ollama reply")
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        assert bob.chat("Hello") == "ollama reply"

    def test_rate_limit_triggers_fallback(self):
        primary = MockRaisingLLM(NvidiaRateLimitError("429"))
        fallback = MockSuccessLLM(reply="ollama 429 reply")
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        assert bob.chat("Hello") == "ollama 429 reply"

    def test_connection_error_no_fallback_configured_propagates(self):
        primary = MockRaisingLLM(NvidiaConnectionError("no fallback"))
        bob = JarvisAssistant(llm_client=primary)
        with pytest.raises(NvidiaConnectionError, match="no fallback"):
            bob.chat("Hello")


# ── Permanent NVIDIA errors do NOT trigger fallback ───────────────────────────

class TestFallbackNotTriggered:
    def test_auth_error_does_not_fallback(self):
        primary = MockRaisingLLM(NvidiaAuthenticationError("401"))
        fallback = MockSuccessLLM()
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        with pytest.raises(NvidiaAuthenticationError):
            bob.chat("Hello")
        assert fallback.model_kwargs == []  # fallback never used

    def test_access_error_does_not_fallback(self):
        primary = MockRaisingLLM(NvidiaAccessError("402"))
        fallback = MockSuccessLLM()
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        with pytest.raises(NvidiaAccessError):
            bob.chat("Hello")
        assert fallback.model_kwargs == []


# ── Ollama availability check ─────────────────────────────────────────────────

class TestOllamaAvailability:
    def test_ollama_unavailable_raises_original_nvidia_error(self):
        primary = MockRaisingLLM(NvidiaConnectionError("original timeout"))
        fallback = MockFallbackWithAvailable(available=False)
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        with pytest.raises(NvidiaConnectionError, match="original timeout"):
            bob.chat("Hello")

    def test_fallback_without_is_available_is_used_unconditionally(self):
        """Fallback with no is_available() method is used without checking."""
        primary = MockRaisingLLM(NvidiaConnectionError("timeout"))
        fallback = MockSuccessLLM(reply="no health check needed")
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        assert bob.chat("Hi") == "no health check needed"


# ── Fallback itself fails ─────────────────────────────────────────────────────

class TestFallbackFailure:
    def test_fallback_failure_raises_original_nvidia_error(self):
        primary = MockRaisingLLM(NvidiaConnectionError("root cause"))
        fallback = MockFallbackRaising(RuntimeError("ollama also broke"))
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        with pytest.raises(NvidiaConnectionError, match="root cause"):
            bob.chat("Hello")

    def test_fallback_failure_leaves_clean_history(self):
        primary = MockRaisingLLM(NvidiaConnectionError("timeout"))
        fallback = MockFallbackRaising(RuntimeError("ollama dead"))
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        with pytest.raises(NvidiaConnectionError):
            bob.chat("Hello")
        # History must be clean — only system prompt
        assert len(bob.history) == 1
        assert bob.history[0]["role"] == "system"


# ── History correctness ───────────────────────────────────────────────────────

class TestHistoryCorrectness:
    def test_no_duplicate_user_message_after_fallback(self):
        primary = MockRaisingLLM(NvidiaConnectionError("timeout"))
        fallback = MockSuccessLLM(reply="ok")
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        bob.chat("unique message")
        user_msgs = [m for m in bob.history if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "unique message"

    def test_successful_fallback_reply_stored_in_history(self):
        primary = MockRaisingLLM(NvidiaConnectionError("timeout"))
        fallback = MockSuccessLLM(reply="fallback answer")
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        bob.chat("test")
        history = bob.history
        # system + user + assistant = 3
        assert len(history) == 3
        assert history[2]["role"] == "assistant"
        assert history[2]["content"] == "fallback answer"

    def test_nvidia_error_no_fallback_leaves_clean_history(self):
        primary = MockRaisingLLM(NvidiaConnectionError("boom"))
        bob = JarvisAssistant(llm_client=primary)
        with pytest.raises(NvidiaConnectionError):
            bob.chat("Hello")
        assert len(bob.history) == 1  # system prompt only


# ── Model not forwarded to fallback ──────────────────────────────────────────

class TestModelNotPassedToFallback:
    def test_fallback_called_with_model_none(self):
        """NVIDIA Laya model ID must never be forwarded to the fallback LLM."""
        primary = MockRaisingLLM(NvidiaConnectionError("timeout"))
        fallback = MockSuccessLLM(reply="ok")
        bob = JarvisAssistant(llm_client=primary, fallback_llm=fallback)
        bob.chat("route me")
        assert fallback.model_kwargs == [None]


# ── Existing callers without fallback_llm still work ─────────────────────────

class TestBackwardCompatibility:
    def test_no_fallback_kwarg_success(self):
        primary = MockSuccessLLM(reply="backward compat")
        bob = JarvisAssistant(llm_client=primary)
        assert bob.chat("Hi") == "backward compat"

    def test_no_fallback_kwarg_error_propagates(self):
        primary = MockRaisingLLM(NvidiaConnectionError("no fallback configured"))
        bob = JarvisAssistant(llm_client=primary)
        with pytest.raises(NvidiaConnectionError):
            bob.chat("Hi")
