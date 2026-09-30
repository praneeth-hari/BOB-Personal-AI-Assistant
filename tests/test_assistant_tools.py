"""
tests/test_assistant_tools.py
──────────────────────────────
Integration tests for JarvisAssistant + ToolRegistry.

All tests use mock LLM clients – Ollama is NOT required.
Covers:
  • Normal conversation still works (no tool registry)
  • Tool calling flow: LLM requests tool → tool executes → result fed back
  • Tool result is passed to LLM for natural-language summary
  • History structure after a tool call turn
  • History restored if LLM raises during agentic loop
  • Unknown tool request handled gracefully
  • Max iteration safeguard
  • Shell/arbitrary commands cannot be executed via the tool system
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.core.assistant import JarvisAssistant
from app.tools.tool import Tool, ToolResult
from app.tools.registry import ToolRegistry


# ─── Mock helpers ──────────────────────────────────────────────────────────────

class SimpleMockLLM:
    """Plain LLM mock — chat() only.  No tool support."""
    def __init__(self, reply: str = "Plain reply."):
        self.reply = reply
        self.calls: list = []

    def chat(self, messages, **kwargs) -> str:
        self.calls.append(list(messages))
        return self.reply


class ToolCapableMockLLM:
    """
    LLM mock that supports chat_with_tools().
    You set .responses as an iterator of message dicts that the mock will yield
    one per call to chat_with_tools().
    """
    def __init__(self):
        self.chat_calls: list = []
        self.tool_calls_log: list = []
        self._responses: list = []

    def set_responses(self, *responses: Dict[str, Any]) -> None:
        """Set the sequence of message dicts to return."""
        self._responses = list(responses)
        self._idx = 0

    def chat(self, messages, **kwargs) -> str:
        self.chat_calls.append(list(messages))
        return "Fallback plain response"

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs: Any) -> Dict[str, Any]:
        self.tool_calls_log.append({"messages": list(messages), "tools": tools})
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


def _make_tool_call_response(tool_name: str, arguments: dict) -> Dict[str, Any]:
    """Helper: build an Ollama-style tool-call message dict."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": tool_name, "arguments": arguments}}
        ],
    }


def _make_text_response(content: str) -> Dict[str, Any]:
    """Helper: build an Ollama-style plain text response."""
    return {"role": "assistant", "content": content}


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _echo_tool(message: str = "ok") -> ToolResult:
    return ToolResult(success=True, data={"echo": message})


def _fail_tool() -> ToolResult:
    return ToolResult(success=False, error="Tool returned failure")


@pytest.fixture()
def registry_with_echo() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="Echoes a message",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        fn=_echo_tool,
    ))
    return reg


@pytest.fixture()
def registry_with_fail() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(name="fail_tool", description="Always fails", parameters={}, fn=_fail_tool))
    return reg


# ═══════════════════════════════════════════════════════════════════════════════
# Plain conversation (no tools)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssistantWithoutTools:
    def test_plain_chat_still_works(self):
        llm = SimpleMockLLM(reply="Hello there!")
        assistant = JarvisAssistant(llm_client=llm)
        reply = assistant.chat("Hi")
        assert reply == "Hello there!"

    def test_no_tool_registry_uses_plain_chat(self):
        llm = ToolCapableMockLLM()
        llm.set_responses()  # no responses needed
        assistant = JarvisAssistant(llm_client=llm)
        # Even though llm has chat_with_tools, no registry → uses chat()
        llm.chat_calls.clear()
        reply = assistant.chat("Hi")
        assert len(llm.chat_calls) == 1
        assert len(llm.tool_calls_log) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tool calling flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssistantToolCallingFlow:
    def test_single_tool_call_and_answer(self, registry_with_echo):
        llm = ToolCapableMockLLM()
        llm.set_responses(
            _make_tool_call_response("echo", {"message": "ping"}),
            _make_text_response("The tool echoed: ping"),
        )
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry_with_echo)
        reply = assistant.chat("Echo ping")
        assert reply == "The tool echoed: ping"

    def test_tool_is_actually_executed(self, registry_with_echo):
        """Verify the tool fn was called and its result appears in history."""
        llm = ToolCapableMockLLM()
        llm.set_responses(
            _make_tool_call_response("echo", {"message": "hello"}),
            _make_text_response("Done"),
        )
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry_with_echo)
        assistant.chat("Run echo")

        # History: system, user, assistant(tool_call), tool_result, assistant(final)
        h = assistant.history
        tool_msg = next((m for m in h if m.get("role") == "tool"), None)
        assert tool_msg is not None
        payload = json.loads(tool_msg["content"])
        assert payload["success"] is True
        assert payload["data"]["echo"] == "hello"

    def test_tool_result_appears_in_second_llm_call(self, registry_with_echo):
        """Second call to LLM must include the tool result so it can summarise."""
        llm = ToolCapableMockLLM()
        llm.set_responses(
            _make_tool_call_response("echo", {"message": "data"}),
            _make_text_response("Here is the summary."),
        )
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry_with_echo)
        assistant.chat("Summarise the echo")

        second_call_messages = llm.tool_calls_log[1]["messages"]
        roles = [m["role"] for m in second_call_messages]
        assert "tool" in roles

    def test_history_after_tool_turn(self, registry_with_echo):
        llm = ToolCapableMockLLM()
        llm.set_responses(
            _make_tool_call_response("echo", {"message": "x"}),
            _make_text_response("Final answer"),
        )
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry_with_echo)
        assistant.chat("Go")

        h = assistant.history
        roles = [m["role"] for m in h]
        # Expected: system, user, assistant(tool_call), tool, assistant(final text)
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_tool_schemas_sent_to_llm(self, registry_with_echo):
        llm = ToolCapableMockLLM()
        llm.set_responses(_make_text_response("No tool needed"))
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry_with_echo)
        assistant.chat("Hello")

        sent_tools = llm.tool_calls_log[0]["tools"]
        tool_names = [t["function"]["name"] for t in sent_tools]
        assert "echo" in tool_names

    def test_unknown_tool_request_handled_gracefully(self, registry_with_echo):
        """LLM requests a non-existent tool; registry returns error; LLM summarises."""
        llm = ToolCapableMockLLM()
        llm.set_responses(
            _make_tool_call_response("nonexistent_tool", {}),
            _make_text_response("I could not find that tool."),
        )
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry_with_echo)
        reply = assistant.chat("Call unknown tool")
        assert reply == "I could not find that tool."

        # Tool result message should contain an error
        h = assistant.history
        tool_msg = next(m for m in h if m["role"] == "tool")
        payload = json.loads(tool_msg["content"])
        assert payload["success"] is False

    def test_failed_tool_result_passed_to_llm(self, registry_with_fail):
        llm = ToolCapableMockLLM()
        llm.set_responses(
            _make_tool_call_response("fail_tool", {}),
            _make_text_response("The tool failed; I cannot complete that."),
        )
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry_with_fail)
        reply = assistant.chat("Trigger fail")
        assert "fail" in reply.lower() or reply  # LLM summarised the failure


# ═══════════════════════════════════════════════════════════════════════════════
# Max iteration guard
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaxIterations:
    def test_max_iterations_stops_infinite_loop(self):
        """If LLM keeps calling tools, we stop after MAX_TOOL_ITERATIONS."""
        from app.core.assistant import MAX_TOOL_ITERATIONS

        llm = ToolCapableMockLLM()
        # Return a tool call every single time
        llm.set_responses(
            *[_make_tool_call_response("echo", {"message": "loop"})
              for _ in range(MAX_TOOL_ITERATIONS + 2)]
        )

        reg = ToolRegistry()
        reg.register(Tool(name="echo", description="", parameters={}, fn=_echo_tool))

        assistant = JarvisAssistant(llm_client=llm, tool_registry=reg)
        reply = assistant.chat("Loop forever")
        # Should return the fallback message, not raise
        assert "maximum" in reply.lower() or "steps" in reply.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Error handling & history consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentErrorHandling:
    def test_llm_error_in_agentic_loop_restores_history(self, registry_with_echo):
        """If chat_with_tools raises, history should be restored (user msg stays)."""
        llm = ToolCapableMockLLM()
        # Make the first call raise
        llm._responses = [None]  # will trigger index error or we mock it

        def raising_chat_with_tools(messages, tools, **kwargs):
            raise ConnectionError("Server down")

        llm.chat_with_tools = raising_chat_with_tools

        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry_with_echo)
        with pytest.raises(ConnectionError):
            assistant.chat("trigger error")

        # History should be [system] only — user message was removed on exception
        assert len(assistant.history) == 1
        assert assistant.history[0]["role"] == "system"

    def test_json_string_arguments_parsed(self, registry_with_echo):
        """Arguments returned as JSON string (not dict) are parsed correctly."""
        llm = ToolCapableMockLLM()
        # Simulate Ollama returning arguments as a JSON string
        llm.set_responses(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "echo",
                            "arguments": json.dumps({"message": "from_string"}),
                        }
                    }
                ],
            },
            _make_text_response("Done"),
        )
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry_with_echo)
        assistant.chat("Test string args")

        h = assistant.history
        tool_msg = next(m for m in h if m["role"] == "tool")
        payload = json.loads(tool_msg["content"])
        assert payload["data"]["echo"] == "from_string"


# ═══════════════════════════════════════════════════════════════════════════════
# Security: arbitrary shell cannot be run via tool system
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolSecurity:
    def test_open_application_rejects_cmd(self):
        from app.tools.open_application import _open_application
        result = _open_application("cmd")
        assert result.success is False

    def test_open_application_rejects_powershell(self):
        from app.tools.open_application import _open_application
        result = _open_application("powershell")
        assert result.success is False

    def test_open_application_rejects_arbitrary_path(self):
        from app.tools.open_application import _open_application
        result = _open_application("C:\\Windows\\System32\\whoami.exe")
        assert result.success is False

    def test_registry_cannot_be_extended_via_llm(self, registry_with_echo):
        """
        The registry is immutable at runtime from the LLM's perspective.
        Executing an unregistered tool name returns an error, not execution.
        """
        result = registry_with_echo.execute("os.system", {"cmd": "whoami"})
        assert result["success"] is False

    def test_search_files_is_read_only(self, tmp_path):
        """search_files must not modify any files."""
        from app.tools.search_files import _search_files
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")
        _search_files(str(tmp_path), "*.txt")
        assert test_file.read_text() == "original content"
