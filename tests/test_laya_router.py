"""
tests/test_laya_router.py
─────────────────────────
Tests for the Laya request router and its integration with the BOB assistant.

Most tests use a fake Laya Router (no GPU needed). The TestLiveLaya class runs
the real installed Laya checkpoint on CUDA and is skipped when laya or CUDA is
unavailable.
"""

from __future__ import annotations

import ast
import inspect
import json
import statistics
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

import app.core.laya_router as laya_router_module
from app.core.assistant import JarvisAssistant
from app.core.laya_router import (
    MODEL_GLM,
    MODEL_GPT_OSS,
    ROUTE_CATEGORIES,
    ROUTE_CRITERIA,
    ROUTE_HINTS,
    UNKNOWN,
    LayaRouter,
    RouteDecision,
    _model_for_category,
    build_router_from_env,
)
from app.tools.registry import ToolRegistry
from app.tools.tool import Tool, ToolResult


# ─── Fakes ────────────────────────────────────────────────────────────────────

class FakeAgent:
    def __init__(self, device: str) -> None:
        self.device = device


class FakeLaya:
    """Stands in for laya.Router."""

    def __init__(self, choice: Any = "device_status", agent_device: str = "cuda",
                 predict_error: Optional[Exception] = None, result: Any = None) -> None:
        self.choice = choice
        self.agent_device = agent_device
        self.predict_error = predict_error
        self.result = result
        self.init_kwargs: Dict[str, Any] = {}
        self.load_calls: List[str] = []
        self.unload_calls: List[str] = []
        self.predict_calls: List[Dict[str, Any]] = []
        self.loaded: List[str] = []

    def __call__(self, **kwargs: Any) -> "FakeLaya":   # acts as the factory
        self.init_kwargs = kwargs
        return self

    def load(self, name: str) -> FakeAgent:
        self.load_calls.append(name)
        self.loaded = [name]
        return FakeAgent(self.agent_device)

    def unload(self, name: str) -> None:
        self.unload_calls.append(name)
        self.loaded = []

    def predict(self, state: Any, questions: Dict[str, Any], model: Optional[str] = None, **kw: Any) -> Any:
        self.predict_calls.append({"state": state, "questions": questions, "model": model})
        if self.predict_error is not None and len(self.predict_calls) > 1:   # warm-up succeeds
            raise self.predict_error
        if self.result is not None and len(self.predict_calls) > 1:
            return self.result
        return {"answers": {"route": {"type": "choice", "choice": self.choice, "confidence": 0.42}}}

    @property
    def user_predictions(self) -> List[Dict[str, Any]]:
        return self.predict_calls[1:]   # first call is the warm-up


def _router(fake: FakeLaya, **kwargs: Any) -> LayaRouter:
    return LayaRouter(router_factory=fake, **kwargs)


class StaticRouter:
    """A RequestRouter that always returns the same decision."""

    def __init__(self, decision: Any) -> None:
        self.decision = decision
        self.calls: List[str] = []

    def route(self, text: str) -> Any:
        self.calls.append(text)
        return self.decision


class ScriptedLLM:
    """LLM client returning scripted chat_with_tools messages and recording inputs."""

    def __init__(self, *messages: Dict[str, Any]) -> None:
        self._messages = list(messages)
        self.tool_calls_inputs: List[List[Dict[str, Any]]] = []
        self.chat_inputs: List[List[Dict[str, Any]]] = []

    def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        self.chat_inputs.append([dict(m) for m in messages])
        return "plain reply"

    def chat_with_tools(self, messages: List[Dict[str, Any]], tools: List[Any], **kwargs: Any) -> Dict[str, Any]:
        self.tool_calls_inputs.append([dict(m) for m in messages])
        return self._messages.pop(0)


def _registry(executed: List[str]) -> ToolRegistry:
    reg = ToolRegistry()

    def battery() -> ToolResult:
        executed.append("get_battery_status")
        return ToolResult(success=True, data={"percent": 77})

    reg.register(Tool(name="get_battery_status", description="Battery info.",
                      parameters={"type": "object", "properties": {}, "required": []}, fn=battery))
    return reg


def _tool_call(name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args or {}}}]}


def _text(content: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": content}


def _hints(messages: List[Dict[str, Any]]) -> List[str]:
    return [m["content"] for m in messages if m["role"] == "system" and m["content"] in ROUTE_HINTS.values()]


# ═══════════════════════════════════════════════════════════════════════════════
# Categories & decisions
# ═══════════════════════════════════════════════════════════════════════════════

class TestCategories:
    def test_small_explicit_category_set(self):
        assert set(ROUTE_CATEGORIES) == {
            "chitchat", "device_status", "local_action", "file_search",
            "web_lookup", "general_knowledge", "unknown",
        }

    def test_unknown_is_not_offered_to_laya(self):
        assert UNKNOWN not in ROUTE_CRITERIA

    def test_every_offered_category_has_a_hint(self):
        assert set(ROUTE_HINTS) == set(ROUTE_CRITERIA)

    def test_unknown_has_no_hint(self):
        assert RouteDecision(category=UNKNOWN).hint is None

    def test_hints_are_provider_neutral(self):
        for hint in ROUTE_HINTS.values():
            assert "nvidia" not in hint.lower()
            assert "ollama" not in hint.lower()
            assert "can be wrong" in hint


# ═══════════════════════════════════════════════════════════════════════════════
# LayaRouter: initialisation & device
# ═══════════════════════════════════════════════════════════════════════════════

class TestInitialization:
    def test_construction_is_lazy(self):
        fake = FakeLaya()
        router = _router(fake)
        assert not router.is_ready
        assert fake.load_calls == []

    def test_requests_cuda_and_single_resident_model(self):
        fake = FakeLaya()
        assert _router(fake).load() is True
        assert fake.init_kwargs == {"device": "cuda", "max_loaded": 1}
        assert fake.load_calls == ["english"]

    def test_loads_once_and_warms_up(self):
        fake = FakeLaya()
        router = _router(fake)
        assert router.load() and router.load()
        router.route("a")
        router.route("b")
        assert fake.load_calls == ["english"]
        assert len(fake.predict_calls) == 3          # 1 warm-up + 2 routes
        assert router.load_seconds is not None

    def test_route_loads_lazily_once(self):
        fake = FakeLaya()
        router = _router(fake)
        router.route("hello")
        router.route("again")
        assert fake.load_calls == ["english"]

    def test_reports_actual_cuda_device(self):
        router = _router(FakeLaya(agent_device="cuda"))
        router.load()
        assert router.device == "cuda"
        assert router.loaded_models == ["english"]

    def test_cpu_fallback_disables_routing(self):
        fake = FakeLaya(agent_device="cpu")
        router = _router(fake)
        assert router.load() is False
        assert "instead of CUDA" in router.failure_reason
        assert fake.unload_calls == ["english"]
        decision = router.route("What is my battery status?")
        assert decision.category == UNKNOWN
        assert fake.predict_calls == []

    def test_cpu_allowed_explicitly(self):
        router = _router(FakeLaya(agent_device="cpu"), allow_cpu=True)
        assert router.load() is True
        assert router.device == "cpu"

    def test_explicit_model_on_every_prediction(self):
        fake = FakeLaya()
        router = _router(fake)
        router.route("Mein Konto wurde zweimal belastet")   # non-English: must not auto-route
        assert {c["model"] for c in fake.predict_calls} == {"english"}

    def test_build_from_env_default_enabled(self, monkeypatch):
        monkeypatch.delenv("BOB_LAYA_ROUTER", raising=False)
        router = build_router_from_env()
        assert isinstance(router, LayaRouter)
        assert not router.is_ready        # not loaded by construction

    @pytest.mark.parametrize("value", ["false", "0", "no", "OFF"])
    def test_build_from_env_disabled(self, monkeypatch, value):
        monkeypatch.setenv("BOB_LAYA_ROUTER", value)
        assert build_router_from_env() is None


# ═══════════════════════════════════════════════════════════════════════════════
# LayaRouter: routing & fallbacks
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouting:
    @pytest.mark.parametrize("category", list(ROUTE_CRITERIA))
    def test_known_categories(self, category):
        decision = _router(FakeLaya(choice=category)).route("some request")
        assert decision.category == category
        assert decision.raw_choice == category
        assert decision.hint == ROUTE_HINTS[category]
        assert decision.error is None
        assert decision.latency_ms >= 0

    def test_request_text_and_single_choice_question(self):
        fake = FakeLaya()
        _router(fake).route("  What is my battery status?  ")
        call = fake.user_predictions[0]
        assert call["state"] == {"request": "What is my battery status?"}
        assert list(call["questions"]) == ["route"]
        assert call["questions"]["route"]["type"] == "choice"
        assert call["questions"]["route"]["criteria"] == ROUTE_CRITERIA

    def test_low_confidence_is_not_thresholded(self):
        fake = FakeLaya(result={"answers": {"route": {"choice": "web_lookup", "confidence": 0.01}}})
        decision = _router(fake).route("latest python?")
        assert decision.category == "web_lookup"
        assert decision.confidence == 0.01

    @pytest.mark.parametrize("choice", ["banana", "unknown", "", None, 42, ["chitchat"]])
    def test_invalid_choice_falls_back_to_unknown(self, choice):
        decision = _router(FakeLaya(choice=choice)).route("hello")
        assert decision.category == UNKNOWN
        assert decision.hint is None
        assert "unexpected Laya choice" in decision.error

    @pytest.mark.parametrize("result", [{}, {"answers": {}}, {"answers": None}, "garbage", None])
    def test_malformed_result_falls_back(self, result):
        fake = FakeLaya()
        fake.result = result if result is not None else {"answers": {"other": {}}}
        decision = _router(fake).route("hello")
        assert decision.category == UNKNOWN
        assert decision.hint is None

    def test_prediction_error_falls_back_but_router_stays_available(self):
        fake = FakeLaya(predict_error=RuntimeError("CUDA error: device-side assert"))
        router = _router(fake)
        decision = router.route("hello")
        assert decision.category == UNKNOWN
        assert "CUDA error" in decision.error
        assert router.is_ready

    def test_load_failure_falls_back_and_is_not_retried(self):
        factory = MagicMock(side_effect=RuntimeError("out of memory"))
        router = LayaRouter(router_factory=factory)
        for _ in range(3):
            decision = router.route("hello")
            assert decision.category == UNKNOWN
            assert "out of memory" in decision.error
        assert factory.call_count == 1

    def test_laya_not_installed(self):
        router = LayaRouter()
        with patch.dict(sys.modules, {"laya": None}):
            assert router.load() is False
        assert "laya" in router.failure_reason
        assert router.route("hi").category == UNKNOWN

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_request_is_not_sent_to_laya(self, text):
        fake = FakeLaya()
        router = _router(fake)
        assert router.route(text).category == UNKNOWN
        assert fake.load_calls == []


# ═══════════════════════════════════════════════════════════════════════════════
# Laya never executes tools
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoToolExecutionByLaya:
    def test_router_module_has_no_tool_access(self):
        tree = ast.parse(inspect.getsource(laya_router_module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert not any(name.startswith("app.") for name in imported), imported
        assert not imported & {"subprocess", "os.system", "shutil"}, imported

    def test_route_decision_carries_no_executable(self):
        decision = _router(FakeLaya(choice="local_action")).route("Open Notepad.")
        for value in vars(decision).values():
            assert not callable(value)

    def test_tool_route_without_llm_tool_call_executes_nothing(self):
        executed: List[str] = []
        registry = _registry(executed)
        registry.execute = MagicMock(wraps=registry.execute)
        llm = ScriptedLLM(_text("I won't do anything."))
        bob = JarvisAssistant(llm_client=llm, tool_registry=registry,
                              router=_router(FakeLaya(choice="local_action")))
        assert bob.chat("Open Notepad.") == "I won't do anything."
        registry.execute.assert_not_called()
        assert executed == []


# ═══════════════════════════════════════════════════════════════════════════════
# Assistant integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssistantIntegration:
    def test_hint_inserted_before_current_user_message(self):
        llm = ScriptedLLM(_text("It is 77%."))
        bob = JarvisAssistant(llm_client=llm, tool_registry=_registry([]), system_prompt="sys",
                              router=_router(FakeLaya(choice="device_status")))
        bob.chat("What is my battery status?")
        sent = llm.tool_calls_inputs[0]
        assert [m["role"] for m in sent] == ["system", "system", "user"]
        assert sent[1]["content"] == ROUTE_HINTS["device_status"]
        assert sent[2]["content"] == "What is my battery status?"

    def test_hint_not_stored_in_history(self):
        llm = ScriptedLLM(_text("Hi!"))
        bob = JarvisAssistant(llm_client=llm, tool_registry=_registry([]), system_prompt="sys",
                              router=_router(FakeLaya(choice="chitchat")))
        bob.chat("Hello BOB.")
        assert bob.history == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hello BOB."},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_routed_once_per_user_message_across_tool_iterations(self):
        router = StaticRouter(RouteDecision(category="device_status"))
        executed: List[str] = []
        llm = ScriptedLLM(_tool_call("get_battery_status"), _text("77%."))
        bob = JarvisAssistant(llm_client=llm, tool_registry=_registry(executed), router=router)

        assert bob.chat("Battery?") == "77%."
        assert router.calls == ["Battery?"]              # not re-run for the tool result turn
        assert executed == ["get_battery_status"]
        # The same hint is present on both LLM calls of this turn, before the user message.
        for sent in llm.tool_calls_inputs:
            assert _hints(sent) == [ROUTE_HINTS["device_status"]]
            hint_index = next(i for i, m in enumerate(sent) if m["content"] == ROUTE_HINTS["device_status"])
            assert sent[hint_index + 1] == {"role": "user", "content": "Battery?"}

    def test_old_hint_does_not_leak_into_next_turn(self):
        router = StaticRouter(RouteDecision(category="web_lookup"))
        llm = ScriptedLLM(_text("a"), _text("b"))
        bob = JarvisAssistant(llm_client=llm, tool_registry=_registry([]), router=router)
        bob.chat("latest python?")
        router.decision = RouteDecision(category=UNKNOWN)
        bob.chat("hmm")
        assert _hints(llm.tool_calls_inputs[1]) == []

    def test_unknown_route_sends_exactly_the_existing_messages(self):
        llm_routed, llm_plain = ScriptedLLM(_text("x")), ScriptedLLM(_text("x"))
        routed = JarvisAssistant(llm_client=llm_routed, tool_registry=_registry([]), system_prompt="sys",
                                 router=_router(FakeLaya(choice="banana")))
        plain = JarvisAssistant(llm_client=llm_plain, tool_registry=_registry([]), system_prompt="sys")
        routed.chat("hello")
        plain.chat("hello")
        assert llm_routed.tool_calls_inputs == llm_plain.tool_calls_inputs
        assert routed.last_route.category == UNKNOWN

    def test_router_exception_falls_back_to_existing_behaviour(self):
        class Exploding:
            def route(self, text: str) -> Any:
                raise RuntimeError("boom")

        llm = ScriptedLLM(_text("still works"))
        bob = JarvisAssistant(llm_client=llm, tool_registry=_registry([]), router=Exploding())
        assert bob.chat("hello") == "still works"
        assert _hints(llm.tool_calls_inputs[0]) == []
        assert bob.last_route is None

    @pytest.mark.parametrize("decision", [None, object(), RouteDecision(category=UNKNOWN), "chitchat"])
    def test_garbage_decisions_mean_no_hint(self, decision):
        llm = ScriptedLLM(_text("ok"))
        bob = JarvisAssistant(llm_client=llm, tool_registry=_registry([]), router=StaticRouter(decision))
        assert bob.chat("hi") == "ok"
        assert _hints(llm.tool_calls_inputs[0]) == []

    def test_plain_chat_path_gets_hint(self):
        llm = ScriptedLLM()
        bob = JarvisAssistant(llm_client=llm, router=StaticRouter(RouteDecision(category="general_knowledge")))
        assert bob.chat("What is the capital of France?") == "plain reply"
        assert _hints(llm.chat_inputs[0]) == [ROUTE_HINTS["general_knowledge"]]

    def test_empty_message_is_not_routed(self):
        router = StaticRouter(RouteDecision(category="chitchat"))
        bob = JarvisAssistant(llm_client=ScriptedLLM(), router=router)
        bob.chat("   ")
        assert router.calls == []

    def test_no_router_is_unchanged(self):
        llm = ScriptedLLM(_text("x"))
        bob = JarvisAssistant(llm_client=llm, tool_registry=_registry([]), system_prompt="sys")
        bob.chat("hello")
        assert llm.tool_calls_inputs[0] == [{"role": "system", "content": "sys"},
                                             {"role": "user", "content": "hello"}]
        assert bob.router is None and bob.last_route is None

    def test_llm_error_restores_history(self):
        class FailingLLM(ScriptedLLM):
            def chat_with_tools(self, messages, tools, **kwargs):
                raise ConnectionError("down")

        bob = JarvisAssistant(llm_client=FailingLLM(), tool_registry=_registry([]), system_prompt="sys",
                              router=StaticRouter(RouteDecision(category="chitchat")))
        with pytest.raises(ConnectionError):
            bob.chat("hello")
        assert bob.history == [{"role": "system", "content": "sys"}]


# ═══════════════════════════════════════════════════════════════════════════════
# Existing tool validation is unchanged with a router attached
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolValidationUnchanged:
    def test_unknown_tool_rejected_by_registry(self):
        executed: List[str] = []
        llm = ScriptedLLM(_tool_call("run_shell", {"cmd": "del *"}), _text("Can't do that."))
        bob = JarvisAssistant(llm_client=llm, tool_registry=_registry(executed),
                              router=StaticRouter(RouteDecision(category="local_action")))
        assert bob.chat("delete everything") == "Can't do that."
        assert executed == []
        result = json.loads(llm.tool_calls_inputs[1][-1]["content"])
        assert result["success"] is False and "Unknown tool" in result["error"]

    def test_invalid_arguments_rejected_by_registry(self):
        executed: List[str] = []
        llm = ScriptedLLM(_tool_call("get_battery_status", {"bogus": 1}), _text("Sorry."))
        bob = JarvisAssistant(llm_client=llm, tool_registry=_registry(executed),
                              router=StaticRouter(RouteDecision(category="device_status")))
        bob.chat("battery?")
        assert executed == []
        result = json.loads(llm.tool_calls_inputs[1][-1]["content"])
        assert result["success"] is False and "Invalid arguments" in result["error"]

    def test_tools_still_run_only_through_registry_execute(self):
        executed: List[str] = []
        registry = _registry(executed)
        registry.execute = MagicMock(wraps=registry.execute)
        llm = ScriptedLLM(_tool_call("get_battery_status"), _text("77%."))
        bob = JarvisAssistant(llm_client=llm, tool_registry=registry,
                              router=StaticRouter(RouteDecision(category="chitchat")))  # wrong route
        assert bob.chat("battery?") == "77%."
        registry.execute.assert_called_once_with("get_battery_status", {})
        assert executed == ["get_battery_status"]

    def test_all_tools_still_offered_regardless_of_route(self):
        seen_tools: List[Any] = []

        class ToolsSpy(ScriptedLLM):
            def chat_with_tools(self, messages, tools, **kwargs):
                seen_tools.append([t["function"]["name"] for t in tools])
                return super().chat_with_tools(messages, tools, **kwargs)

        from app.tools.registry import build_default_registry
        registry = build_default_registry()
        bob = JarvisAssistant(llm_client=ToolsSpy(_text("hi")), tool_registry=registry,
                              router=StaticRouter(RouteDecision(category="chitchat")))
        bob.chat("Hello BOB.")
        assert seen_tools == [registry.list_tools()]


# ═══════════════════════════════════════════════════════════════════════════════
# Both providers still work with the router
# ═══════════════════════════════════════════════════════════════════════════════

class TestProviders:
    def test_ollama_provider_with_router(self):
        from app.llm.ollama_client import OllamaClient

        replies = [
            {"message": {"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": "get_battery_status", "arguments": {}}}]}},
            {"message": {"role": "assistant", "content": "Battery at 77%."}},
        ]
        payloads: List[Dict[str, Any]] = []

        def fake_post(url, json=None, timeout=None):
            payloads.append(json)
            resp = MagicMock(ok=True)
            resp.json.return_value = replies[len(payloads) - 1]
            return resp

        executed: List[str] = []
        bob = JarvisAssistant(llm_client=OllamaClient(host="http://localhost:11434", model="qwen2.5-coder:7b"),
                              tool_registry=_registry(executed), system_prompt="sys",
                              router=_router(FakeLaya(choice="device_status")))
        with patch("app.llm.ollama_client.requests.post", side_effect=fake_post):
            assert bob.chat("What is my battery status?") == "Battery at 77%."

        assert executed == ["get_battery_status"]
        assert payloads[0]["model"] == "qwen2.5-coder:7b"
        assert payloads[0]["tools"][0]["function"]["name"] == "get_battery_status"
        assert _hints(payloads[0]["messages"]) == [ROUTE_HINTS["device_status"]]
        assert payloads[1]["messages"][-1]["role"] == "tool"

    def test_nvidia_provider_with_router(self):
        from app.llm.nvidia_client import NvidiaClient

        replies = [
            {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_battery_status", "arguments": "{}"}}]}}]},
            {"choices": [{"index": 0, "finish_reason": "stop",
                          "message": {"role": "assistant", "content": "Battery at 77%."}}]},
        ]
        payloads: List[Dict[str, Any]] = []

        def fake_post(url, json=None, headers=None, timeout=None):
            payloads.append(json)
            resp = MagicMock(ok=True, status_code=200)
            resp.json.return_value = replies[len(payloads) - 1]
            return resp

        executed: List[str] = []
        bob = JarvisAssistant(llm_client=NvidiaClient(api_key="nvapi-FAKE-router-test"),
                              tool_registry=_registry(executed), system_prompt="sys",
                              router=_router(FakeLaya(choice="device_status")))
        with patch("app.llm.nvidia_client.requests.post", side_effect=fake_post):
            assert bob.chat("What is my battery status?") == "Battery at 77%."

        assert executed == ["get_battery_status"]
        assert payloads[0]["model"] == MODEL_GPT_OSS  # device_status now routes to GPT-OSS
        # The transient hint is a system message just before the user message.
        assert payloads[0]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": ROUTE_HINTS["device_status"]},
            {"role": "user", "content": "What is my battery status?"},
        ]
        assert payloads[1]["messages"][-1] == {
            "role": "tool", "tool_call_id": "call_1",
            "content": json.dumps({"success": True, "data": {"percent": 77}, "error": None}),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Model-level routing policy
# ═══════════════════════════════════════════════════════════════════════════════

FAKE_KEY = "nvapi-FAKE-routing-policy-key-1234"
_POST = "app.llm.nvidia_client.requests.post"


def _mock_resp(text: str) -> MagicMock:
    body = {"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}]}
    m = MagicMock()
    m.ok = True
    m.status_code = 200
    m.json.return_value = body
    return m



class TestModelSelection:
    """
    Verify the Laya → model routing policy.

    All tests are deterministic (no real API calls, no GPU).
    """

    # ── _model_for_category() ─────────────────────────────────────────────────

    @pytest.mark.parametrize("category,expected", [
        ("chitchat",          MODEL_GPT_OSS),
        ("general_knowledge", MODEL_GPT_OSS),
        ("device_status",     MODEL_GPT_OSS),
        ("local_action",      MODEL_GPT_OSS),
        ("file_search",       MODEL_GPT_OSS),
        ("web_lookup",        MODEL_GPT_OSS),
        ("unknown",           None),
    ])
    def test_category_to_model_mapping(self, category, expected):
        assert _model_for_category(category) == expected

    def test_unknown_category_produces_no_override(self):
        assert _model_for_category(UNKNOWN) is None

    # ── RouteDecision carries selected_model ──────────────────────────────────

    @pytest.mark.parametrize("choice,expected_model", [
        ("chitchat",          MODEL_GPT_OSS),
        ("general_knowledge", MODEL_GPT_OSS),
        ("device_status",     MODEL_GPT_OSS),
        ("local_action",      MODEL_GPT_OSS),
        ("file_search",       MODEL_GPT_OSS),
        ("web_lookup",        MODEL_GPT_OSS),
    ])
    def test_router_sets_selected_model(self, choice, expected_model):
        decision = _router(FakeLaya(choice=choice)).route("any request")
        assert decision.selected_model == expected_model
        assert decision.category == choice

    def test_router_failure_has_no_selected_model(self):
        """Unknown/invalid Laya output must not set a model override."""
        decision = _router(FakeLaya(choice="banana")).route("anything")
        assert decision.category == UNKNOWN
        assert decision.selected_model is None

    def test_unknown_decision_has_no_selected_model(self):
        assert RouteDecision(category=UNKNOWN).selected_model is None

    # ── Named examples from the routing policy ────────────────────────────────

    @pytest.mark.parametrize("category,request_text,expected_model", [
        # All standard categories → GPT-OSS (benchmark: GLM ~78-80 s queue latency)
        ("chitchat",          "Hello BOB",                            MODEL_GPT_OSS),
        ("chitchat",          "How are you?",                         MODEL_GPT_OSS),
        ("general_knowledge", "Explain what RAG is",                  MODEL_GPT_OSS),
        ("device_status",     "What's my battery status?",            MODEL_GPT_OSS),
        ("local_action",      "Open VS Code",                         MODEL_GPT_OSS),
        ("file_search",       "Search my computer for report.pdf",    MODEL_GPT_OSS),
        ("local_action",      "Set volume to 30",                     MODEL_GPT_OSS),
        ("web_lookup",        "Search the web for NVIDIA news",       MODEL_GPT_OSS),
    ])
    def test_policy_examples(self, category, request_text, expected_model):
        decision = _router(FakeLaya(choice=category)).route(request_text)
        assert decision.selected_model == expected_model

    # ── Laya never executes tools ─────────────────────────────────────────────

    def test_selected_model_is_a_string_not_callable(self):
        for model in (MODEL_GPT_OSS, MODEL_GLM):
            assert isinstance(model, str) and not callable(model)

    def test_route_decision_has_no_tool_access(self):
        d = RouteDecision(category="device_status", selected_model=MODEL_GLM)
        for attr in ("execute", "run", "tool_registry", "registry"):
            assert not hasattr(d, attr)

    def test_laya_module_has_no_tool_imports(self):
        """laya_router must not import ToolRegistry or any tool module."""
        import app.core.laya_router as mod
        assert not hasattr(mod, "ToolRegistry")
        assert not hasattr(mod, "build_default_registry")

    # ── Both model IDs are valid and distinct ─────────────────────────────────

    def test_model_ids_are_nonempty_and_distinct(self):
        assert isinstance(MODEL_GPT_OSS, str) and MODEL_GPT_OSS
        assert isinstance(MODEL_GLM, str) and MODEL_GLM
        assert MODEL_GPT_OSS != MODEL_GLM

    def test_nvidia_client_accepts_both_model_ids(self):
        from app.llm.nvidia_client import NvidiaClient
        assert NvidiaClient(api_key=FAKE_KEY, model=MODEL_GPT_OSS).model == MODEL_GPT_OSS
        assert NvidiaClient(api_key=FAKE_KEY, model=MODEL_GLM).model == MODEL_GLM

    # ── Assistant passes the routed model to the LLM ─────────────────────────

    def test_assistant_passes_gpt_oss_for_chitchat(self):
        """chitchat → GPT-OSS must appear in the LLM payload."""
        from app.llm.nvidia_client import NvidiaClient
        client = NvidiaClient(api_key=FAKE_KEY)
        bob = JarvisAssistant(llm_client=client, router=_router(FakeLaya(choice="chitchat")))
        with patch(_POST, return_value=_mock_resp("Hi!")) as mock_post:
            bob.chat("Hello BOB")
        assert mock_post.call_args[1]["json"]["model"] == MODEL_GPT_OSS

    def test_assistant_passes_gpt_oss_for_device_status(self):
        """device_status → GPT-OSS must appear in the LLM payload."""
        from app.llm.nvidia_client import NvidiaClient
        client = NvidiaClient(api_key=FAKE_KEY)
        bob = JarvisAssistant(llm_client=client, router=_router(FakeLaya(choice="device_status")))
        with patch(_POST, return_value=_mock_resp("Battery at 80%.")) as mock_post:
            bob.chat("What is my battery status?")
        assert mock_post.call_args[1]["json"]["model"] == MODEL_GPT_OSS

    def test_assistant_passes_gpt_oss_for_tool_agentic_loop(self):
        """GPT-OSS model must be used in both the first and follow-up LLM calls within the tool loop."""
        from app.llm.nvidia_client import NvidiaClient

        def _tool_call_body():
            return {"choices": [{"index": 0, "finish_reason": "tool_calls",
                                  "message": {"role": "assistant", "content": "",
                                              "tool_calls": [{"id": "c1", "type": "function",
                                                              "function": {"name": "get_battery_status",
                                                                           "arguments": "{}"}}]}}]}

        payloads: List[Dict] = []

        def fake_post(url, json=None, **kw):
            payloads.append(json)
            if len(payloads) == 1:
                return _mock_resp_raw(_tool_call_body())
            return _mock_resp("Battery OK.")

        def _mock_resp_raw(body):
            m = MagicMock(); m.ok = True; m.status_code = 200; m.json.return_value = body; return m

        reg = ToolRegistry()
        reg.register(Tool(name="get_battery_status", description="battery",
                          parameters={"type": "object", "properties": {}},
                          fn=lambda: ToolResult(success=True, data={"percent": 90})))

        client = NvidiaClient(api_key=FAKE_KEY)
        bob = JarvisAssistant(llm_client=client, tool_registry=reg,
                              router=_router(FakeLaya(choice="device_status")))
        with patch(_POST, side_effect=fake_post):
            bob.chat("Battery?")

        assert len(payloads) == 2
        assert payloads[0]["model"] == MODEL_GPT_OSS, "first call must use GPT-OSS"
        assert payloads[1]["model"] == MODEL_GPT_OSS, "follow-up call must use GPT-OSS"

    # ── Ollama backend: Laya model routing is suppressed ──────────────────────

    def test_ollama_backend_ignores_laya_model_routing(self):
        """When LLM is OllamaClient, selected_model must never be forwarded."""
        from app.llm.ollama_client import OllamaClient

        client = OllamaClient(model="qwen2.5-coder:7b")
        bob = JarvisAssistant(llm_client=client, router=_router(FakeLaya(choice="device_status")))

        received: List[Optional[str]] = []
        original = client.chat
        def spy(messages, *, model=None, **kw):
            received.append(model)
            return "ok"
        client.chat = spy  # type: ignore[method-assign]

        bob.chat("test")
        assert received[-1] is None

    # ── NVIDIA_MODEL env var still works when no routing ─────────────────────

    def test_nvidia_model_env_var_is_respected_without_router(self, monkeypatch):
        """Without a router, NVIDIA_MODEL env var controls the model unchanged."""
        from app.llm.nvidia_client import NvidiaClient
        monkeypatch.setenv("NVIDIA_MODEL", "z-ai/glm-5.3-flash")
        client = NvidiaClient(api_key=FAKE_KEY)
        bob = JarvisAssistant(llm_client=client)  # no router
        with patch(_POST, return_value=_mock_resp("Hello!")) as mock_post:
            bob.chat("Hello")
        assert mock_post.call_args[1]["json"]["model"] == "z-ai/glm-5.3-flash"


# ═══════════════════════════════════════════════════════════════════════════════
# Live: the real installed Laya checkpoint on CUDA
# ═══════════════════════════════════════════════════════════════════════════════

def _live_available() -> bool:
    try:
        import laya  # noqa: F401
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def live() -> LayaRouter:
    router = LayaRouter()
    if not router.load():
        pytest.skip(f"Laya could not load on CUDA: {router.failure_reason}")
    return router


@pytest.mark.skipif(not _live_available(), reason="laya or CUDA not available")
@pytest.mark.filterwarnings("ignore:laya. this checkpoint ships invalid temperatures")
class TestLiveLaya:
    def test_runs_on_cuda_with_one_model(self, live):
        assert live.device.startswith("cuda")
        assert live.loaded_models == ["english"]

    @pytest.mark.parametrize("text, expected", [
        ("What is my battery status?", "device_status"),
        ("Hello BOB, how are you?", "chitchat"),
        ("What is the capital of France?", "general_knowledge"),
        ("Is Chrome running?", "device_status"),
        ("Search the web for the latest Python release.", "web_lookup"),
        ("Open Notepad.", "local_action"),
        ("Where is the file called budget.xlsx?", "file_search"),
    ])
    def test_known_examples(self, live, text, expected):
        assert live.route(text).category == expected

    def test_warm_latency_and_no_second_model(self, live):
        texts = ["What is my battery status?", "Hello BOB, how are you?", "Is Chrome running?",
                 "Mein Konto wurde zweimal belastet"]    # non-English must not load multilingual
        latencies = []
        for _ in range(5):
            for text in texts:
                t = time.perf_counter()
                live.route(text)
                latencies.append((time.perf_counter() - t) * 1000)
        assert live.loaded_models == ["english"]
        # ~70-90 ms expected on the RTX 3050. When another process (e.g. Ollama with a
        # partly GPU-offloaded model) fills the 4 GB of VRAM, Laya spills into shared
        # memory and runs at ~400+ ms: an environment condition, not a regression.
        median = statistics.median(latencies)
        if median >= 300:
            import torch
            free, _total = torch.cuda.mem_get_info()
            if free < 512 * 2**20:
                pytest.skip(f"GPU memory contended (free {free / 2**20:.0f} MiB); "
                            f"median routing latency {median:.0f} ms")
        assert median < 300, f"median routing latency {median:.0f} ms"
