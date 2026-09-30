"""
tests/test_ollama_client.py
────────────────────────────
Unit tests for OllamaClient.

Uses unittest.mock to patch requests – Ollama is NOT required.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from app.llm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_mock_response(status_code: int = 200, json_data: dict | None = None, text: str = ""):
    """Create a mock requests.Response object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.ok = (200 <= status_code < 300)
    mock_resp.text = text or str(json_data)
    if json_data is not None:
        mock_resp.json.return_value = json_data
    else:
        mock_resp.json.side_effect = ValueError("no json")
    return mock_resp


def _good_response(content: str = "I am BOB."):
    return _make_mock_response(
        200,
        {"message": {"role": "assistant", "content": content}},
    )


# ─── Initialisation tests ─────────────────────────────────────────────────────

class TestOllamaClientInit:
    def test_defaults(self):
        client = OllamaClient()
        assert client.host == "http://localhost:11434"
        assert client.model == "qwen2.5-coder:7b"
        assert client.timeout == 120

    def test_explicit_params(self):
        client = OllamaClient(
            host="http://myhost:9999",
            model="llama3:8b",
            timeout=30,
        )
        assert client.host == "http://myhost:9999"
        assert client.model == "llama3:8b"
        assert client.timeout == 30

    def test_trailing_slash_stripped_from_host(self):
        client = OllamaClient(host="http://localhost:11434/")
        assert not client.host.endswith("/")

    def test_env_var_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "mistral:7b")
        client = OllamaClient()
        assert client.model == "mistral:7b"

    def test_env_var_host(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.50:11434")
        client = OllamaClient()
        assert client.host == "http://192.168.1.50:11434"

    def test_explicit_params_override_env_vars(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "from-env")
        client = OllamaClient(model="from-param")
        assert client.model == "from-param"


# ─── chat() tests ─────────────────────────────────────────────────────────────

class TestOllamaClientChat:
    def test_successful_chat_returns_content(self):
        client = OllamaClient()
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _good_response("Hello from Ollama!")
            result = client.chat([{"role": "user", "content": "Hi"}])
        assert result == "Hello from Ollama!"

    def test_chat_sends_correct_payload(self):
        client = OllamaClient(model="testmodel:1b")
        messages = [{"role": "user", "content": "Ping"}]
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _good_response()
            client.chat(messages)
            call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["model"] == "testmodel:1b"
        assert payload["messages"] == messages
        assert payload["stream"] is False

    def test_chat_model_override(self):
        client = OllamaClient(model="default-model:7b")
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _good_response()
            client.chat([{"role": "user", "content": "Hi"}], model="override-model:13b")
            payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "override-model:13b"

    def test_connection_error_raises_ollama_connection_error(self):
        import requests as _req
        client = OllamaClient()
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.side_effect = _req.exceptions.ConnectionError("refused")
            with pytest.raises(OllamaConnectionError, match="Cannot connect"):
                client.chat([{"role": "user", "content": "Hi"}])

    def test_timeout_raises_ollama_connection_error(self):
        import requests as _req
        client = OllamaClient()
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.side_effect = _req.exceptions.Timeout()
            with pytest.raises(OllamaConnectionError, match="timed out"):
                client.chat([{"role": "user", "content": "Hi"}])

    def test_http_error_status_raises_ollama_response_error(self):
        client = OllamaClient()
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response(500, text="Internal server error")
            with pytest.raises(OllamaResponseError, match="HTTP 500"):
                client.chat([{"role": "user", "content": "Hi"}])

    def test_malformed_json_raises_ollama_response_error(self):
        client = OllamaClient()
        bad_resp = MagicMock()
        bad_resp.ok = True
        bad_resp.status_code = 200
        bad_resp.text = "not json"
        bad_resp.json.side_effect = ValueError("no json")
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = bad_resp
            with pytest.raises(OllamaResponseError, match="Unexpected Ollama response"):
                client.chat([{"role": "user", "content": "Hi"}])

    def test_missing_message_key_raises_ollama_response_error(self):
        client = OllamaClient()
        with patch("app.llm.ollama_client.requests.post") as mock_post:
            mock_post.return_value = _make_mock_response(200, {"unexpected": "payload"})
            with pytest.raises(OllamaResponseError):
                client.chat([{"role": "user", "content": "Hi"}])


# ─── is_available() tests ─────────────────────────────────────────────────────

class TestOllamaClientAvailability:
    def test_is_available_true_when_server_ok(self):
        client = OllamaClient()
        with patch("app.llm.ollama_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response(200, {"models": []})
            assert client.is_available() is True

    def test_is_available_false_when_connection_refused(self):
        import requests as _req
        client = OllamaClient()
        with patch("app.llm.ollama_client.requests.get") as mock_get:
            mock_get.side_effect = _req.exceptions.ConnectionError()
            assert client.is_available() is False

    def test_list_local_models_returns_names(self):
        client = OllamaClient()
        models_payload = {
            "models": [
                {"name": "qwen2.5-coder:7b"},
                {"name": "llama3:8b"},
            ]
        }
        with patch("app.llm.ollama_client.requests.get") as mock_get:
            mock_get.return_value = _make_mock_response(200, models_payload)
            result = client.list_local_models()
        assert result == ["qwen2.5-coder:7b", "llama3:8b"]

    def test_list_local_models_returns_empty_on_error(self):
        import requests as _req
        client = OllamaClient()
        with patch("app.llm.ollama_client.requests.get") as mock_get:
            mock_get.side_effect = _req.exceptions.ConnectionError()
            assert client.list_local_models() == []
