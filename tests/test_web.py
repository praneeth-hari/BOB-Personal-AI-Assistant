"""
tests/test_web.py
─────────────────
Tests for the local FastAPI web interface.

All tests use a mocked assistant (and mocked LLM metadata), so neither Ollama,
an API key, nor a microphone is required.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.interfaces.web import create_app
from app.llm.errors import LLMConfigurationError
from app.llm.ollama_client import OllamaConnectionError

PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"


# ─── Mocks ────────────────────────────────────────────────────────────────────

class MockLLM:
    model = "mock-model"


class MockRegistry:
    def list_tools(self) -> List[str]:
        return ["get_system_info", "open_application"]


class MockAssistant:
    """Minimal duck-typed stand-in for JarvisAssistant."""

    def __init__(
        self,
        reply: str = "Mock reply.",
        tools: List[str] | None = None,
        fail: bool = False,
    ) -> None:
        self.reply = reply
        self.tools = tools or []
        self.fail = fail
        self.reset_calls = 0
        self._history: List[Dict[str, Any]] = [
            {"role": "system", "content": "Test system prompt."}
        ]
        self._llm = MockLLM()
        self.tool_registry = MockRegistry()

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def chat(self, message: str) -> str:
        if self.fail:
            raise RuntimeError("secret internal failure")
        self._history.append({"role": "user", "content": message})
        for name in self.tools:
            self._history.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": name, "arguments": {}}}],
                }
            )
            self._history.append({"role": "tool", "content": "{}"})
        self._history.append({"role": "assistant", "content": self.reply})
        return self.reply

    def reset(self) -> None:
        self.reset_calls += 1
        self._history = [{"role": "system", "content": "Test system prompt."}]


@pytest.fixture
def assistant() -> MockAssistant:
    return MockAssistant(tools=["open_application"])


@pytest.fixture
def client(assistant: MockAssistant) -> TestClient:
    app = create_app(assistant_factory=lambda: assistant, port=PORT)
    with TestClient(app, base_url=BASE_URL) as test_client:
        yield test_client


# ─── Page / assets ────────────────────────────────────────────────────────────

def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "BOB" in response.text
    assert "/static/app.js" in response.text


def test_static_assets_are_served(client: TestClient) -> None:
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "api/chat" in response.text


# ─── Health ───────────────────────────────────────────────────────────────────

def test_health_returns_status_provider_and_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider"] == "ollama"
    assert body["model"] == "mock-model"
    assert "get_system_info" in body["tools"]


# ─── Chat ─────────────────────────────────────────────────────────────────────

def test_chat_returns_reply_and_tools_used(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "Hello BOB"})

    assert response.status_code == 200
    assert response.json() == {
        "reply": "Mock reply.",
        "tools_used": ["open_application"],
    }


def test_chat_rejects_empty_message(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 400
    assert response.json() == {"error": "Please enter a message."}


def test_chat_rejects_invalid_body(client: TestClient) -> None:
    response = client.post("/api/chat", json={})

    assert response.status_code == 422
    assert "error" in response.json()


def test_chat_failure_returns_friendly_error(
    client: TestClient, assistant: MockAssistant
) -> None:
    assistant.fail = True
    response = client.post("/api/chat", json={"message": "Hello"})

    assert response.status_code == 500
    body = response.json()
    assert "error" in body
    assert "secret internal failure" not in response.text
    assert "Traceback" not in response.text


def test_chat_ollama_connection_error_is_friendly() -> None:
    class UnreachableAssistant(MockAssistant):
        def chat(self, message: str) -> str:
            raise OllamaConnectionError("secret host detail")

    app = create_app(assistant_factory=UnreachableAssistant, port=PORT)
    with TestClient(app, base_url=BASE_URL) as test_client:
        response = test_client.post("/api/chat", json={"message": "Hi"})

    assert response.status_code == 503
    assert "Ollama" in response.json()["error"]
    assert "secret host detail" not in response.text


def test_configuration_error_is_reported_safely() -> None:
    def failing_factory() -> MockAssistant:
        raise LLMConfigurationError(
            "NVIDIA_API_KEY is required when LLM_PROVIDER=nvidia."
        )

    app = create_app(assistant_factory=failing_factory, port=PORT)
    with TestClient(app, base_url=BASE_URL) as test_client:
        health = test_client.get("/api/health")
        chat = test_client.post("/api/chat", json={"message": "Hi"})

    assert health.status_code == 200
    assert health.json()["status"] == "error"
    assert "NVIDIA_API_KEY" in health.json()["error"]

    assert chat.status_code == 503
    assert "NVIDIA_API_KEY" in chat.json()["error"]


# ─── Reset ────────────────────────────────────────────────────────────────────

def test_reset_clears_conversation(
    client: TestClient, assistant: MockAssistant
) -> None:
    client.post("/api/chat", json={"message": "Hello"})
    response = client.post("/api/reset", json={})

    assert response.status_code == 200
    assert response.json() == {"status": "reset"}
    assert assistant.reset_calls == 1
    assert len(assistant.history) == 1


# ─── Security middleware ──────────────────────────────────────────────────────

def test_rejects_bad_host_header(client: TestClient) -> None:
    response = client.get("/api/health", headers={"host": "evil.example:8000"})

    assert response.status_code == 403
    assert response.json() == {"error": "Invalid Host header."}


def test_rejects_wrong_port_in_host_header(client: TestClient) -> None:
    response = client.get("/api/health", headers={"host": "localhost:9999"})

    assert response.status_code == 403
    assert response.json() == {"error": "Invalid Host header."}


def test_accepts_localhost_host_header(client: TestClient) -> None:
    response = client.get("/api/health", headers={"host": f"localhost:{PORT}"})

    assert response.status_code == 200


def test_rejects_foreign_origin(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "Hello"},
        headers={"origin": "http://evil.example"},
    )

    assert response.status_code == 403
    assert response.json() == {"error": "Invalid Origin header."}


def test_accepts_local_origin(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "Hello"},
        headers={"origin": BASE_URL},
    )

    assert response.status_code == 200


def test_requires_json_content_type_on_post(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        content="message=Hello",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 415
    assert response.json() == {
        "error": "Content-Type must be application/json."
    }


def test_reset_also_requires_json_content_type(client: TestClient) -> None:
    response = client.post(
        "/api/reset",
        content=b"",
        headers={"content-type": "text/plain"},
    )

    assert response.status_code == 415


def test_accepts_json_content_type_with_charset(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        content=b'{"message": "Hello"}',
        headers={"content-type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
