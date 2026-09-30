"""
tests/test_nvidia_client.py
───────────────────────────
Unit tests for the NVIDIA NIM LLM provider and the provider factory.

The HTTP layer (requests.post) is always mocked — no API key or internet
connection is required and no real NVIDIA requests are made.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.core.assistant import JarvisAssistant
from app.llm.errors import LLMConfigurationError
from app.llm.factory import create_llm_client, get_provider_name
from app.llm.nvidia_client import (
    NvidiaAccessError,
    NvidiaAuthenticationError,
    NvidiaClient,
    NvidiaConfigurationError,
    NvidiaConnectionError,
    NvidiaRateLimitError,
    NvidiaResponseError,
    convert_messages,
    convert_tool_schemas,
    normalize_response,
)
from app.llm.ollama_client import OllamaClient
from app.tools.registry import ToolRegistry, build_default_registry
from app.tools.tool import Tool, ToolResult

FAKE_KEY = "nvapi-FAKE-test-key-1234567890abcdef"
POST = "app.llm.nvidia_client.requests.post"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _body_text(text: str, finish: str = "stop", reasoning: str | None = None) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"id": "chatcmpl-1", "object": "chat.completion", "model": "z-ai/glm-5.3-flash",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def _tc(name: str, args: Any = "{}", call_id: str = "call_1") -> Dict[str, Any]:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def _body_calls(*calls: Dict[str, Any], content: Any = None) -> Dict[str, Any]:
    return {"id": "chatcmpl-2", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": content, "tool_calls": list(calls)}}]}


def _resp(body: Any = None, status: int = 200, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.text = text or (json.dumps(body) if body is not None else "")
    if body is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = body
    return resp


class Recorder:
    """side_effect for requests.post that records calls and returns scripted responses."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self.responses.pop(0)

    @property
    def payloads(self) -> List[Dict[str, Any]]:
        return [c["json"] for c in self.calls]


def _client(**kwargs: Any) -> NvidiaClient:
    return NvidiaClient(api_key=FAKE_KEY, **kwargs)


USER = [{"role": "system", "content": "You are BOB."}, {"role": "user", "content": "Hi"}]

BATTERY_SCHEMA = {"type": "function", "function": {
    "name": "get_battery_status", "description": "Battery info.",
    "parameters": {"type": "object", "properties": {}, "required": []}}}
SEARCH_SCHEMA = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                   "required": ["query"]}}}


# ═══════════════════════════════════════════════════════════════════════════════
# Initialisation & configuration
# ═══════════════════════════════════════════════════════════════════════════════

class TestInit:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_MODEL", raising=False)
        monkeypatch.delenv("NVIDIA_TIMEOUT", raising=False)
        client = _client()
        assert client.model == "openai/gpt-oss-20b"
        assert client.timeout == 120
        assert client.API_URL == "https://integrate.api.nvidia.com/v1/chat/completions"

    def test_glm_available_as_secondary_model(self, monkeypatch):
        """GLM-5.3 Flash can still be selected explicitly and is exposed as SECONDARY_MODEL."""
        monkeypatch.setenv("NVIDIA_MODEL", "z-ai/glm-5.3-flash")
        client = _client()
        assert client.model == "z-ai/glm-5.3-flash"
        assert NvidiaClient.SECONDARY_MODEL == "z-ai/glm-5.3-flash"

    def test_both_model_ids_accepted(self, monkeypatch):
        """Both the new default and the legacy GLM model are accepted by the client."""
        for model_id in ("openai/gpt-oss-20b", "z-ai/glm-5.3-flash"):
            monkeypatch.setenv("NVIDIA_MODEL", model_id)
            assert _client().model == model_id

    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        assert NvidiaClient().model

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_missing_key(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        else:
            monkeypatch.setenv("NVIDIA_API_KEY", value)
        with pytest.raises(NvidiaConfigurationError, match="NVIDIA_API_KEY is required") as info:
            NvidiaClient()
        assert isinstance(info.value, LLMConfigurationError)
        assert "LLM_PROVIDER=ollama" in str(info.value)

    def test_model_from_env_and_override(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_MODEL", "z-ai/glm-5.3")
        assert _client().model == "z-ai/glm-5.3"
        assert _client(model="custom/model").model == "custom/model"

    def test_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_TIMEOUT", "30")
        assert _client().timeout == 30
        monkeypatch.setenv("NVIDIA_TIMEOUT", "nope")
        assert _client().timeout == 120

    def test_banner_attributes_and_interface(self):
        client = _client()
        assert client.host
        assert callable(client.chat) and callable(client.chat_with_tools)


# ═══════════════════════════════════════════════════════════════════════════════
# Request shape
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequest:
    def test_endpoint_auth_and_payload(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_MODEL", raising=False)
        rec = Recorder(_resp(_body_text("hello")))
        with patch(POST, side_effect=rec):
            assert _client(timeout=45).chat(USER) == "hello"
        call = rec.calls[0]
        assert call["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
        assert call["timeout"] == 45
        assert call["json"] == {"model": "openai/gpt-oss-20b", "messages": USER,
                                "max_tokens": 4096, "stream": False}

    def test_tools_sent_in_openai_function_format(self):
        rec = Recorder(_resp(_body_text("ok")))
        with patch(POST, side_effect=rec):
            _client().chat_with_tools(USER, [SEARCH_SCHEMA])
        assert rec.payloads[0]["tools"] == [SEARCH_SCHEMA]

    def test_all_eleven_bob_tools_sent_as_functions(self):
        rec = Recorder(_resp(_body_text("ok")))
        schemas = build_default_registry().get_tool_schemas()
        with patch(POST, side_effect=rec):
            _client().chat_with_tools(USER, schemas)
        tools = rec.payloads[0]["tools"]
        assert len(tools) == 11
        assert {t["type"] for t in tools} == {"function"}
        assert [t["function"]["name"] for t in tools] == [s["function"]["name"] for s in schemas]

    def test_plain_chat_sends_no_tools(self):
        rec = Recorder(_resp(_body_text("ok")))
        with patch(POST, side_effect=rec):
            _client().chat(USER)
        assert "tools" not in rec.payloads[0]

    def test_options_mapped(self):
        rec = Recorder(_resp(_body_text("ok")))
        with patch(POST, side_effect=rec):
            _client().chat(USER, options={"temperature": 0.2, "num_predict": 50, "unknown": 1})
        payload = rec.payloads[0]
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 50
        assert "unknown" not in payload

    def test_model_override_per_call(self):
        rec = Recorder(_resp(_body_text("ok")))
        with patch(POST, side_effect=rec):
            _client().chat(USER, model="z-ai/glm-5.3")
        assert rec.payloads[0]["model"] == "z-ai/glm-5.3"

    def test_no_messages_raises_without_request(self):
        with patch(POST) as post:
            with pytest.raises(NvidiaResponseError):
                _client().chat([])
        post.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Conversion
# ═══════════════════════════════════════════════════════════════════════════════

class TestConversion:
    def test_no_arg_tool_gets_empty_object_schema(self):
        tool = convert_tool_schemas([{"type": "function", "function": {"name": "t", "description": "d"}}])[0]
        assert tool["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_invalid_tool_entries_skipped(self):
        assert convert_tool_schemas([{"type": "function", "function": {"description": "x"}}, "junk", None]) == []

    def test_plain_history(self):
        history = USER + [{"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Bye"}]
        assert convert_messages(history) == history

    def test_tool_turn_with_parallel_calls(self):
        history = USER + [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_a", "function": {"name": "get_battery_status", "arguments": {}}},
                {"id": "call_b", "function": {"name": "web_search", "arguments": {"query": "python"}}}]},
            {"role": "tool", "content": '{"success": true}'},
            {"role": "tool", "content": '{"success": false}'},
        ]
        assert convert_messages(history)[2:] == [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "get_battery_status", "arguments": "{}"}},
                {"id": "call_b", "type": "function",
                 "function": {"name": "web_search", "arguments": '{"query": "python"}'}}]},
            {"role": "tool", "tool_call_id": "call_a", "content": '{"success": true}'},
            {"role": "tool", "tool_call_id": "call_b", "content": '{"success": false}'},
        ]

    def test_calls_without_ids_get_unique_generated_ids(self):
        history = USER + [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "a", "arguments": "{}"}}, {"function": {"name": "b", "arguments": {}}}]},
            {"role": "tool", "content": "{}"}, {"role": "tool", "content": "{}"},
        ]
        converted = convert_messages(history)
        call_ids = [c["id"] for c in converted[2]["tool_calls"]]
        assert len(set(call_ids)) == 2
        assert [m["tool_call_id"] for m in converted[3:]] == call_ids

    def test_orphan_tool_result_kept_as_context(self):
        assert convert_messages(USER + [{"role": "tool", "content": "{}"}])[-1] == {
            "role": "user", "content": "Tool result: {}"}

    def test_empty_messages_skipped(self):
        assert convert_messages([{"role": "user", "content": "x"}, {"role": "assistant", "content": ""}]) == [
            {"role": "user", "content": "x"}]


# ═══════════════════════════════════════════════════════════════════════════════
# Response normalisation
# ═══════════════════════════════════════════════════════════════════════════════

class TestResponse:
    def test_text(self):
        assert normalize_response(_body_text("Hello.")) == {"role": "assistant", "content": "Hello."}

    def test_reasoning_content_is_not_part_of_reply(self):
        assert normalize_response(_body_text("Answer.", reasoning="secret thinking"))["content"] == "Answer."

    def test_tool_call(self):
        assert normalize_response(_body_calls(_tc("get_battery_status"))) == {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "get_battery_status", "arguments": {}}}]}

    def test_tool_call_arguments_parsed(self):
        msg = normalize_response(_body_calls(_tc("web_search", {"query": "latest python", "max_results": 3})))
        assert msg["tool_calls"][0]["function"]["arguments"] == {"query": "latest python", "max_results": 3}

    def test_dict_arguments_accepted(self):
        call = {"id": "c1", "type": "function", "function": {"name": "t", "arguments": {"a": 1}}}
        assert normalize_response(_body_calls(call))["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_multiple_tool_calls(self):
        msg = normalize_response(_body_calls(_tc("get_battery_status", call_id="c1"),
                                             _tc("web_search", {"query": "x"}, call_id="c2")))
        assert [(c["id"], c["function"]["name"]) for c in msg["tool_calls"]] == [
            ("c1", "get_battery_status"), ("c2", "web_search")]

    def test_text_alongside_tool_call(self):
        msg = normalize_response(_body_calls(_tc("get_battery_status"), content="Let me check."))
        assert msg["content"] == "Let me check." and len(msg["tool_calls"]) == 1

    @pytest.mark.parametrize("call", [
        {"id": "c1", "type": "function", "function": {"name": "", "arguments": "{}"}},
        {"id": "", "type": "function", "function": {"name": "t", "arguments": "{}"}},
        {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{not json"}},
        {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "[1, 2]"}},
        {"id": "c1", "type": "function"},
        "junk",
    ])
    def test_malformed_tool_call(self, call):
        with pytest.raises(NvidiaResponseError, match="malformed tool call"):
            normalize_response(_body_calls(call))

    def test_malformed_call_skipped_when_valid_call_present(self):
        msg = normalize_response(_body_calls(
            {"id": "bad", "type": "function", "function": {"name": "t", "arguments": "{oops"}},
            _tc("get_battery_status", call_id="good")))
        assert [c["id"] for c in msg["tool_calls"]] == ["good"]

    @pytest.mark.parametrize("content", [None, "", "   "])
    def test_empty_response(self, content):
        with pytest.raises(NvidiaResponseError, match="empty response"):
            normalize_response(_body_text(content))

    def test_truncated_empty_response_mentions_max_tokens(self):
        with pytest.raises(NvidiaResponseError, match="max_tokens"):
            normalize_response(_body_text("", finish="length"))

    @pytest.mark.parametrize("body", [None, [], "text", {}, {"choices": []}, {"choices": ["x"]},
                                      {"choices": [{"message": None}]}])
    def test_unexpected_format(self, body):
        with pytest.raises(NvidiaResponseError, match="Unexpected NVIDIA response"):
            normalize_response(body)


# ═══════════════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrors:
    @pytest.mark.parametrize("code", [401, 403])
    def test_authentication(self, code):
        with patch(POST, return_value=_resp({"detail": "Authentication failed"}, code)):
            with pytest.raises(NvidiaAuthenticationError, match="NVIDIA_API_KEY"):
                _client().chat(USER)

    def test_payment_required(self):
        with patch(POST, return_value=_resp({"detail": "Account has insufficient credits"}, 402)):
            with pytest.raises(NvidiaAccessError, match="HTTP 402.*insufficient credits"):
                _client().chat(USER)

    def test_rate_limit(self):
        with patch(POST, return_value=_resp({"detail": "Too many requests"}, 429)):
            with pytest.raises(NvidiaRateLimitError, match="HTTP 429"):
                _client().chat(USER)

    def test_model_not_found(self):
        with patch(POST, return_value=_resp({"detail": "Function not found"}, 404)):
            with pytest.raises(NvidiaResponseError, match="NVIDIA_MODEL"):
                _client().chat(USER)

    @pytest.mark.parametrize("code", [500, 502, 503])
    def test_server_error(self, code):
        with patch(POST, return_value=_resp({"detail": "overloaded"}, code)):
            with pytest.raises(NvidiaResponseError, match=f"server error \\(HTTP {code}\\): overloaded"):
                _client().chat(USER)

    def test_validation_error_detail_list(self):
        body = {"detail": [{"loc": ["body", "messages"], "msg": "field required", "type": "missing"}]}
        with patch(POST, return_value=_resp(body, 422)):
            with pytest.raises(NvidiaResponseError, match="HTTP 422.*field required"):
                _client().chat_with_tools(USER, [BATTERY_SCHEMA])

    @pytest.mark.parametrize("body", [{"error": {"message": "bad tools"}}, {"error": "bad tools"},
                                      {"message": "bad tools"}, {"title": "bad tools"}])
    def test_other_error_body_shapes(self, body):
        with patch(POST, return_value=_resp(body, 400)):
            with pytest.raises(NvidiaResponseError, match="bad tools"):
                _client().chat(USER)

    def test_non_json_error_body(self):
        with patch(POST, return_value=_resp(None, 502, text="<html>Bad gateway</html>")):
            with pytest.raises(NvidiaResponseError, match="Bad gateway"):
                _client().chat(USER)

    def test_malformed_json_success_body(self):
        with patch(POST, return_value=_resp(None, 200, text="{not json")):
            with pytest.raises(NvidiaResponseError, match="non-JSON"):
                _client().chat(USER)

    @pytest.mark.parametrize("exc", [requests.exceptions.ReadTimeout("slow"),
                                     requests.exceptions.ConnectTimeout("slow")])
    def test_timeout(self, exc):
        with patch(POST, side_effect=exc):
            with pytest.raises(NvidiaConnectionError, match="timed out after 120s"):
                _client().chat(USER)

    @pytest.mark.parametrize("exc", [requests.exceptions.ConnectionError("refused"),
                                     requests.exceptions.RequestException("boom")])
    def test_network_failure(self, exc):
        with patch(POST, side_effect=exc):
            with pytest.raises(NvidiaConnectionError, match="Cannot connect"):
                _client().chat_with_tools(USER, [BATTERY_SCHEMA])

    def test_assistant_history_restored_on_error(self):
        bob = JarvisAssistant(llm_client=_client(), tool_registry=build_default_registry(), system_prompt="sys")
        with patch(POST, side_effect=requests.exceptions.ConnectionError("x")):
            with pytest.raises(NvidiaConnectionError):
                bob.chat("hello")
        assert bob.history == [{"role": "system", "content": "sys"}]


# ═══════════════════════════════════════════════════════════════════════════════
# API key leakage
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoKeyLeakage:
    def test_key_not_in_repr_or_public_attributes(self):
        client = _client()
        assert FAKE_KEY not in repr(client) and FAKE_KEY not in str(client)
        public = {k: v for k, v in vars(client).items() if not k.startswith("_")}
        assert FAKE_KEY not in public.values()

    @pytest.mark.parametrize("code", [400, 401, 402, 500])
    def test_key_redacted_from_errors_and_logs(self, caplog, code):
        body = {"detail": f"bad key {FAKE_KEY} / Bearer {FAKE_KEY}"}
        with caplog.at_level(logging.DEBUG, logger="app.llm.nvidia_client"):
            with patch(POST, return_value=_resp(body, code)):
                with pytest.raises(NvidiaResponseError) as info:
                    _client().chat(USER)
        assert FAKE_KEY not in str(info.value)
        assert FAKE_KEY not in caplog.text
        assert info.value.__cause__ is None

    def test_key_redacted_from_network_error_logs(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="app.llm.nvidia_client"):
            with patch(POST, side_effect=requests.exceptions.ConnectionError(f"Bearer {FAKE_KEY} failed")):
                with pytest.raises(NvidiaConnectionError) as info:
                    _client().chat(USER)
        assert FAKE_KEY not in str(info.value) and FAKE_KEY not in caplog.text
        assert info.value.__cause__ is None and info.value.__suppress_context__

    def test_missing_key_error_contains_no_secret(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "")
        with pytest.raises(NvidiaConfigurationError) as info:
            NvidiaClient()
        assert "nvapi-" not in str(info.value)

    def test_key_only_in_authorization_header(self):
        rec = Recorder(_resp(_body_calls(_tc("echo", {"text": "hi"}))), _resp(_body_text("done")))
        registry = ToolRegistry()
        registry.register(Tool(name="echo", description="echo", parameters={
            "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            fn=lambda text: ToolResult(success=True, data=text)))
        bob = JarvisAssistant(llm_client=_client(), tool_registry=registry)
        with patch(POST, side_effect=rec):
            bob.chat("say hi")
        assert FAKE_KEY not in json.dumps(bob.history, default=str)
        for call in rec.calls:
            assert FAKE_KEY not in call["url"]
            assert FAKE_KEY not in json.dumps(call["json"], default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end with the real agentic loop (NVIDIA mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgenticLoop:
    def _registry(self, executed: List[Dict[str, Any]]) -> ToolRegistry:
        reg = ToolRegistry()

        def battery() -> ToolResult:
            executed.append({"tool": "get_battery_status"})
            return ToolResult(success=True, data={"percent": 77})

        def search(query: str, max_results: int = 5) -> ToolResult:
            executed.append({"tool": "web_search", "query": query})
            return ToolResult(success=True, data=[{"title": "Python 3.99"}])

        reg.register(Tool(name="get_battery_status", description="Battery info.",
                          parameters=BATTERY_SCHEMA["function"]["parameters"], fn=battery))
        reg.register(Tool(name="web_search", description="Search the web.",
                          parameters=SEARCH_SCHEMA["function"]["parameters"], fn=search))
        return reg

    def test_tool_executed_by_registry_then_final_answer(self):
        executed: List[Dict[str, Any]] = []
        rec = Recorder(_resp(_body_calls(_tc("get_battery_status", call_id="call_x"))),
                       _resp(_body_text("Your battery is at 77%.")))
        bob = JarvisAssistant(llm_client=_client(), tool_registry=self._registry(executed), system_prompt="sys")
        with patch(POST, side_effect=rec):
            assert bob.chat("What is my battery status?") == "Your battery is at 77%."

        assert executed == [{"tool": "get_battery_status"}]
        assert rec.payloads[1]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is my battery status?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_x", "type": "function", "function": {"name": "get_battery_status", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_x",
             "content": json.dumps({"success": True, "data": {"percent": 77}, "error": None})},
        ]
        assert "tools" in rec.payloads[1]      # tools stay available on the follow-up call

    def test_multiple_tool_calls(self):
        executed: List[Dict[str, Any]] = []
        rec = Recorder(_resp(_body_calls(_tc("get_battery_status", call_id="c1"),
                                         _tc("web_search", {"query": "python"}, call_id="c2"))),
                       _resp(_body_text("Done.")))
        bob = JarvisAssistant(llm_client=_client(), tool_registry=self._registry(executed))
        with patch(POST, side_effect=rec):
            assert bob.chat("battery and python?") == "Done."
        assert [e["tool"] for e in executed] == ["get_battery_status", "web_search"]
        assert [m["tool_call_id"] for m in rec.payloads[1]["messages"] if m["role"] == "tool"] == ["c1", "c2"]

    def test_unknown_tool_not_executed(self):
        executed: List[Dict[str, Any]] = []
        rec = Recorder(_resp(_body_calls(_tc("run_shell", {"cmd": "del *"}))), _resp(_body_text("I can't.")))
        bob = JarvisAssistant(llm_client=_client(), tool_registry=self._registry(executed))
        with patch(POST, side_effect=rec):
            assert bob.chat("delete everything") == "I can't."
        assert executed == []
        result = json.loads(rec.payloads[1]["messages"][-1]["content"])
        assert result["success"] is False and "Unknown tool" in result["error"]

    def test_invalid_arguments_reported_back(self):
        executed: List[Dict[str, Any]] = []
        rec = Recorder(_resp(_body_calls(_tc("web_search", {"bogus": 1}))), _resp(_body_text("Sorry.")))
        bob = JarvisAssistant(llm_client=_client(), tool_registry=self._registry(executed))
        with patch(POST, side_effect=rec):
            assert bob.chat("search") == "Sorry."
        assert executed == []
        result = json.loads(rec.payloads[1]["messages"][-1]["content"])
        assert result["success"] is False and "Invalid arguments" in result["error"]

    def test_history_kept_across_turns(self):
        executed: List[Dict[str, Any]] = []
        rec = Recorder(_resp(_body_text("Hello!")), _resp(_body_calls(_tc("get_battery_status"))),
                       _resp(_body_text("77%.")))
        bob = JarvisAssistant(llm_client=_client(), tool_registry=self._registry(executed), system_prompt="sys")
        with patch(POST, side_effect=rec):
            bob.chat("Hello BOB.")
            bob.chat("Battery?")
        assert rec.payloads[2]["messages"][:4] == [
            {"role": "system", "content": "sys"}, {"role": "user", "content": "Hello BOB."},
            {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Battery?"}]


# ═══════════════════════════════════════════════════════════════════════════════
# Provider factory
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactory:
    def test_default_is_nvidia(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        monkeypatch.delenv("NVIDIA_MODEL", raising=False)
        assert get_provider_name() == "nvidia"
        client = create_llm_client()
        assert isinstance(client, NvidiaClient)
        assert client.model == "openai/gpt-oss-20b"

    def test_blank_provider_is_nvidia(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "  ")
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        assert isinstance(create_llm_client(), NvidiaClient)

    def test_nvidia_selected_with_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "NVIDIA")
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        monkeypatch.setenv("NVIDIA_MODEL", "z-ai/glm-5.3")
        client = create_llm_client()
        assert isinstance(client, NvidiaClient) and client.model == "z-ai/glm-5.3"

    def test_ollama_selected_without_nvidia_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        assert isinstance(create_llm_client(), OllamaClient)

    def test_explicit_provider_argument(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        assert isinstance(create_llm_client("nvidia"), NvidiaClient)

    def test_missing_key_errors_and_does_not_fall_back(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with patch("app.llm.ollama_client.OllamaClient") as ollama:
            with pytest.raises(LLMConfigurationError, match="NVIDIA_API_KEY is required"):
                create_llm_client()
            ollama.assert_not_called()

    @pytest.mark.parametrize("provider", ["grok", "gemini", "openai", "nim"])
    def test_unknown_provider(self, monkeypatch, provider):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        with pytest.raises(LLMConfigurationError, match="Unknown LLM_PROVIDER") as info:
            create_llm_client()
        assert "nvidia" in str(info.value) and "ollama" in str(info.value)

    def test_run_cli_uses_factory(self):
        from app.interfaces import cli
        with patch("app.llm.factory.create_llm_client", return_value=MagicMock()) as factory, \
             patch("app.core.laya_router.build_router_from_env", return_value=None), \
             patch.object(cli.CLISession, "run"), \
             patch.object(cli, "_configure_stdout_utf8"):
            cli.run_cli()
        factory.assert_called_once_with()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI error display
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLIErrorDisplay:
    def test_nvidia_errors_shown_without_crashing(self):
        from app.interfaces.cli import CLISession

        assistant = MagicMock()
        assistant._llm = SimpleNamespace(model="z-ai/glm-5.3-flash", host="NVIDIA NIM API")
        assistant.tool_registry = None
        assistant.chat.side_effect = [NvidiaRateLimitError("NVIDIA rate limit"), NvidiaConnectionError("offline"),
                                      NvidiaAccessError("credits"), NvidiaAuthenticationError("bad key"), "ok"]
        session = CLISession(assistant=assistant, voice_reply_enabled=False)
        with patch("builtins.input", side_effect=["a", "b", "c", "d", "e", "exit"]), \
             patch("builtins.print") as mock_print:
            session.run()
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        for expected in ("[Response error] NVIDIA rate limit", "[Connection error] offline",
                         "[Response error] credits", "[Response error] bad key"):
            assert expected in printed
        assert assistant.chat.call_count == 5
