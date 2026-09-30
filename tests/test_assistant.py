"""
tests/test_assistant.py
───────────────────────
Unit tests for JarvisAssistant.

All tests use a mock LLM client – Ollama is NOT required.
"""

from __future__ import annotations

import pytest

from app.core.assistant import JarvisAssistant


# ─── Minimal mock LLM client ─────────────────────────────────────────────────

class MockLLMClient:
    """Always returns a fixed reply; records every call for assertion."""

    def __init__(self, reply: str = "Mock reply.") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        # Store a shallow copy so post-mutation of the live history
        # does not retroactively change what we recorded.
        self.calls.append(list(messages))
        return self.reply


class FailingLLMClient:
    """Always raises – used to test error propagation."""

    def chat(self, messages, **kwargs):
        raise RuntimeError("LLM exploded")


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestAssistantInit:
    def test_default_system_prompt_is_set(self):
        assistant = JarvisAssistant(llm_client=MockLLMClient())
        assert assistant.system_prompt  # non-empty
        assert "BOB" in assistant.system_prompt

    def test_default_system_prompt_is_provider_neutral(self, monkeypatch):
        monkeypatch.delenv("BOB_SYSTEM_PROMPT", raising=False)
        monkeypatch.delenv("JARVIS_SYSTEM_PROMPT", raising=False)
        prompt = JarvisAssistant(llm_client=MockLLMClient()).system_prompt.lower()
        # Must not claim BOB always runs locally on Ollama…
        assert "entirely on the user's machine" not in prompt
        assert "using ollama" not in prompt
        assert "local ai assistant" not in prompt
        # …and must acknowledge both supported providers.
        assert "ollama" in prompt and "nvidia" in prompt
        assert "configured" in prompt

    def test_custom_system_prompt(self):
        custom = "You are a pirate."
        assistant = JarvisAssistant(llm_client=MockLLMClient(), system_prompt=custom)
        assert assistant.system_prompt == custom

    def test_history_starts_with_system_message(self):
        assistant = JarvisAssistant(llm_client=MockLLMClient())
        assert len(assistant.history) == 1
        assert assistant.history[0]["role"] == "system"

    def test_system_prompt_env_var(self, monkeypatch):
        monkeypatch.setenv("JARVIS_SYSTEM_PROMPT", "You are a test bot.")
        assistant = JarvisAssistant(llm_client=MockLLMClient())
        assert assistant.system_prompt == "You are a test bot."

    def test_system_prompt_bob_env_var(self, monkeypatch):
        monkeypatch.setenv("BOB_SYSTEM_PROMPT", "You are BOB custom.")
        assistant = JarvisAssistant(llm_client=MockLLMClient())
        assert assistant.system_prompt == "You are BOB custom."

    def test_explicit_prompt_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("JARVIS_SYSTEM_PROMPT", "Env prompt.")
        assistant = JarvisAssistant(
            llm_client=MockLLMClient(), system_prompt="Explicit prompt."
        )
        assert assistant.system_prompt == "Explicit prompt."


class TestConversationHistory:
    def test_chat_appends_user_and_assistant_turns(self):
        mock = MockLLMClient(reply="Hello!")
        assistant = JarvisAssistant(llm_client=mock)
        assistant.chat("Hi there")

        history = assistant.history
        # system + user + assistant = 3
        assert len(history) == 3
        assert history[1]["role"] == "user"
        assert history[1]["content"] == "Hi there"
        assert history[2]["role"] == "assistant"
        assert history[2]["content"] == "Hello!"

    def test_multiple_turns_accumulate(self):
        mock = MockLLMClient()
        assistant = JarvisAssistant(llm_client=mock)
        assistant.chat("Turn 1")
        assistant.chat("Turn 2")
        assistant.chat("Turn 3")

        # system + 3 user + 3 assistant = 7
        assert len(assistant.history) == 7

    def test_full_history_is_sent_to_llm(self):
        mock = MockLLMClient()
        assistant = JarvisAssistant(llm_client=mock)
        assistant.chat("First message")
        assistant.chat("Second message")

        # First call: system + user1 = 2 messages
        assert len(mock.calls[0]) == 2
        # Second call: system + user1 + assistant1 + user2 = 4 messages
        assert len(mock.calls[1]) == 4

    def test_reset_clears_history_but_keeps_system_prompt(self):
        mock = MockLLMClient()
        assistant = JarvisAssistant(llm_client=mock)
        assistant.chat("Message before reset")
        assert len(assistant.history) > 1

        assistant.reset()
        assert len(assistant.history) == 1
        assert assistant.history[0]["role"] == "system"

    def test_history_property_returns_copy(self):
        assistant = JarvisAssistant(llm_client=MockLLMClient())
        h1 = assistant.history
        h1.append({"role": "user", "content": "injected"})
        # Internal history should not be affected
        assert len(assistant.history) == 1

    def test_blank_message_returns_placeholder(self):
        mock = MockLLMClient()
        assistant = JarvisAssistant(llm_client=mock)
        reply = assistant.chat("   ")
        assert reply  # non-empty
        assert mock.calls == []  # LLM was NOT called

    def test_whitespace_is_stripped_from_user_message(self):
        mock = MockLLMClient()
        assistant = JarvisAssistant(llm_client=mock)
        assistant.chat("  hello  ")
        # The user message is the last entry in calls[0] (system + user).
        # We cross-check against history to be explicit about role.
        user_entry = next(m for m in mock.calls[0] if m["role"] == "user")
        assert user_entry["content"] == "hello"


class TestErrorPropagation:
    def test_llm_error_propagates_out_of_chat(self):
        assistant = JarvisAssistant(llm_client=FailingLLMClient())
        with pytest.raises(RuntimeError, match="LLM exploded"):
            assistant.chat("Trigger error")

    def test_failed_chat_does_not_add_assistant_turn(self):
        assistant = JarvisAssistant(llm_client=FailingLLMClient())
        with pytest.raises(RuntimeError):
            assistant.chat("Trigger error")
        # chat() removes the optimistically-added user message on failure,
        # so history contains only the system prompt.
        assert len(assistant.history) == 1
        assert assistant.history[0]["role"] == "system"
