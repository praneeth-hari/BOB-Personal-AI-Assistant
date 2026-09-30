"""
tests/test_text_tool_fallback.py
──────────────────────────────────
Tests for the _parse_text_tool_call fallback in JarvisAssistant.

qwen2.5-coder:7b (Ollama 0.34.x) emits tool calls as plain JSON text in
the content field rather than as structured tool_calls. This fallback
ensures BOB works regardless.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from app.core.assistant import JarvisAssistant
from app.tools.tool import Tool, ToolResult
from app.tools.registry import ToolRegistry


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _echo_tool(message: str = "ok") -> ToolResult:
    return ToolResult(success=True, data={"echo": message})


class ToolCapableMockLLM:
    def __init__(self):
        self.tool_calls_log: list = []
        self._responses: list = []
        self._idx = 0

    def set_responses(self, *responses):
        self._responses = list(responses)
        self._idx = 0

    def chat(self, messages, **kwargs) -> str:
        return "Fallback"

    def chat_with_tools(self, messages, tools, **kwargs) -> Dict[str, Any]:
        self.tool_calls_log.append({"messages": list(messages), "tools": tools})
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


@pytest.fixture()
def echo_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo", description="Echoes",
        parameters={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
        fn=_echo_tool,
    ))
    return reg


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_text_tool_call unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseTextToolCall:
    def test_pattern_a_no_args(self):
        content = '{"name": "get_system_info", "arguments": {}}'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_system_info"
        assert result[0]["function"]["arguments"] == {}

    def test_pattern_a_with_args(self):
        content = '{"name": "open_application", "arguments": {"application": "notepad"}}'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert result[0]["function"]["arguments"]["application"] == "notepad"

    def test_pattern_a_null_arguments(self):
        content = '{"name": "get_system_info", "arguments": null}'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert len(result) == 1
        assert result[0]["function"]["arguments"] == {}

    def test_pattern_b_wrapped_list(self):
        content = '{"tool_calls": [{"name": "search_files", "arguments": {"directory": "D:\\\\", "query": "*.py"}}]}'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "search_files"

    def test_plain_text_no_match(self):
        assert JarvisAssistant._parse_text_tool_call("Your RAM usage is 65%.") == []

    def test_empty_string_no_match(self):
        assert JarvisAssistant._parse_text_tool_call("") == []

    def test_whitespace_only_no_match(self):
        assert JarvisAssistant._parse_text_tool_call("   ") == []

    def test_invalid_json_no_match(self):
        assert JarvisAssistant._parse_text_tool_call("{not valid json}") == []

    def test_unrelated_json_no_match(self):
        assert JarvisAssistant._parse_text_tool_call('{"key": "value", "other": 123}') == []

    def test_json_array_no_match(self):
        assert JarvisAssistant._parse_text_tool_call('[{"name": "x"}]') == []

    def test_json_inside_code_fence(self):
        content = '```json\n{"name": "get_system_info", "arguments": {}}\n```'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_system_info"
        assert result[0]["function"]["arguments"] == {}

    def test_json_inside_plain_code_fence(self):
        content = '```\n{"name": "get_system_info", "arguments": {}}\n```'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_system_info"

    def test_json_with_surrounding_prose(self):
        content = 'I will check your system stats: {"name": "get_system_info", "arguments": {}} Thanks!'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_system_info"

    def test_json_in_xml_tags(self):
        content = '<tool_call>\n{"name": "get_system_info", "arguments": {}}\n</tool_call>'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_system_info"

    def test_known_tool_omitted_arguments(self):
        content = '{"name": "get_system_info"}'
        result = JarvisAssistant._parse_text_tool_call(content, known_tools={"get_system_info"})
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_system_info"
        assert result[0]["function"]["arguments"] == {}

    def test_unknown_tool_omitted_arguments_rejected(self):
        content = '{"name": "Alice"}'
        result = JarvisAssistant._parse_text_tool_call(content, known_tools={"get_system_info"})
        assert result == []

    def test_parameters_key_alias(self):
        content = '{"name": "open_application", "parameters": {"application": "notepad"}}'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert len(result) == 1
        assert result[0]["function"]["arguments"]["application"] == "notepad"

    def test_stringified_json_arguments(self):
        content = '{"name": "open_application", "arguments": "{\\"application\\": \\"notepad\\"}"}'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert len(result) == 1
        assert result[0]["function"]["arguments"]["application"] == "notepad"

    def test_direct_list_of_tool_calls(self):
        content = '[{"name": "get_system_info", "arguments": {}}]'
        result = JarvisAssistant._parse_text_tool_call(content)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_system_info"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: text-mode triggers actual tool execution
# ═══════════════════════════════════════════════════════════════════════════════

class TestTextModeFallbackIntegration:
    def test_text_mode_tool_is_executed(self, echo_registry):
        """Simulate qwen2.5-coder:7b: tool call in content, no tool_calls key."""
        llm = ToolCapableMockLLM()
        llm.set_responses(
            {
                "role": "assistant",
                "content": '{"name": "echo", "arguments": {"message": "text_mode"}}',
            },
            {"role": "assistant", "content": "Done via text-mode fallback."},
        )
        assistant = JarvisAssistant(llm_client=llm, tool_registry=echo_registry)
        reply = assistant.chat("Test")
        assert reply == "Done via text-mode fallback."

        h = assistant.history
        tool_msg = next((m for m in h if m["role"] == "tool"), None)
        assert tool_msg is not None
        payload = json.loads(tool_msg["content"])
        assert payload["success"] is True
        assert payload["data"]["echo"] == "text_mode"

    def test_text_mode_tool_result_in_history(self, echo_registry):
        """History must contain the tool turn even in text-mode path."""
        llm = ToolCapableMockLLM()
        llm.set_responses(
            {"role": "assistant", "content": '{"name": "echo", "arguments": {"message": "hi"}}'},
            {"role": "assistant", "content": "Final"},
        )
        assistant = JarvisAssistant(llm_client=llm, tool_registry=echo_registry)
        assistant.chat("Go")
        roles = [m["role"] for m in assistant.history]
        assert "tool" in roles

    def test_regular_text_not_treated_as_tool_call(self, echo_registry):
        """Normal assistant reply must NOT be parsed as a tool call."""
        llm = ToolCapableMockLLM()
        llm.set_responses(
            {"role": "assistant", "content": "I cannot help with that."},
        )
        assistant = JarvisAssistant(llm_client=llm, tool_registry=echo_registry)
        reply = assistant.chat("Something")
        assert reply == "I cannot help with that."
        assert not any(m["role"] == "tool" for m in assistant.history)

    def test_ram_query_code_fence_regression(self):
        """
        Regression test: When model outputs tool call wrapped in a markdown code fence
        ```json\n{"name": "get_system_info", "arguments": {}}\n```
        it must execute get_system_info and return the natural-language answer, NOT raw JSON.
        """
        from app.tools.registry import build_default_registry

        llm = ToolCapableMockLLM()
        llm.set_responses(
            {
                "role": "assistant",
                "content": '```json\n{"name": "get_system_info", "arguments": {}}\n```',
            },
            {
                "role": "assistant",
                "content": "Your system is using 84.2% RAM.",
            },
        )
        registry = build_default_registry()
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry)

        reply = assistant.chat("what is ram usage")

        # Must return the natural language answer, NOT raw JSON
        assert reply == "Your system is using 84.2% RAM."
        assert "get_system_info" not in reply

        # Tool must have executed and result added to history
        tool_turn = next((m for m in assistant.history if m["role"] == "tool"), None)
        assert tool_turn is not None
        result = json.loads(tool_turn["content"])
        assert result["success"] is True
        assert "ram" in result["data"]

    def test_ram_query_with_surrounding_prose_regression(self):
        """Regression test: Tool call embedded in surrounding prose executes."""
        from app.tools.registry import build_default_registry

        llm = ToolCapableMockLLM()
        llm.set_responses(
            {
                "role": "assistant",
                "content": 'Let me check that for you: {"name": "get_system_info", "arguments": {}}',
            },
            {
                "role": "assistant",
                "content": "Current RAM usage is 8 GB out of 16 GB.",
            },
        )
        registry = build_default_registry()
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry)

        reply = assistant.chat("what is ram usage")
        assert reply == "Current RAM usage is 8 GB out of 16 GB."
        tool_turn = next((m for m in assistant.history if m["role"] == "tool"), None)
        assert tool_turn is not None
        result = json.loads(tool_turn["content"])
        assert result["success"] is True

    def test_close_notepad_code_fence_integration(self):
        """
        Integration test: 'close notepad' emitted as a code-fenced tool call
        results in close_application executing and returning a natural language answer,
        not a raw JSON response.
        """
        from unittest.mock import patch, MagicMock
        from app.tools.registry import build_default_registry

        llm = ToolCapableMockLLM()
        llm.set_responses(
            {
                "role": "assistant",
                "content": '```json\n{"name": "close_application", "arguments": {"application": "notepad"}}\n```',
            },
            {
                "role": "assistant",
                "content": "Notepad has been closed.",
            },
        )
        registry = build_default_registry()
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry)

        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_proc.info = {"name": "notepad.exe", "pid": 9999}

        with patch("app.tools.close_application.psutil.process_iter", return_value=[mock_proc]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([mock_proc], [])):
            reply = assistant.chat("close notepad")

        # Must return natural language answer, NOT raw JSON
        assert reply == "Notepad has been closed."
        assert "close_application" not in reply

        # Tool must have executed and result recorded in history
        tool_turn = next((m for m in assistant.history if m["role"] == "tool"), None)
        assert tool_turn is not None
        result = json.loads(tool_turn["content"])
        assert result["success"] is True
        assert result["data"]["application"] == "notepad"
        assert result["data"]["status"] == "closed"
        mock_proc.terminate.assert_called_once()

    def test_close_notepad_native_tool_call_integration(self):
        """
        Integration test: 'close notepad' with native Ollama tool_calls structure
        results in close_application executing and returning natural language answer.
        """
        from unittest.mock import patch, MagicMock
        from app.tools.registry import build_default_registry

        llm = ToolCapableMockLLM()
        llm.set_responses(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "close_application",
                            "arguments": {"application": "notepad"},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "I have closed Notepad for you.",
            },
        )
        registry = build_default_registry()
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry)

        mock_proc = MagicMock()
        mock_proc.pid = 8888
        mock_proc.info = {"name": "notepad.exe", "pid": 8888}

        with patch("app.tools.close_application.psutil.process_iter", return_value=[mock_proc]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([mock_proc], [])):
            reply = assistant.chat("close notepad")

        assert reply == "I have closed Notepad for you."
        tool_turn = next((m for m in assistant.history if m["role"] == "tool"), None)
        assert tool_turn is not None
        result = json.loads(tool_turn["content"])
        assert result["success"] is True
        mock_proc.terminate.assert_called_once()

