"""
app/interfaces/web.py
─────────────────────
Local browser interface for BOB.

This module is *only* responsible for I/O and transport:
  • Serving the static chat page from web/.
  • Forwarding user messages to JarvisAssistant.
  • Returning replies (and the tools used) as JSON.
  • Clearing history on request.

All assistant logic lives in app/core/assistant.py.

Security:
  • Binds to 127.0.0.1 only.
  • Rejects requests whose Host header is not localhost/127.0.0.1 on the
    configured port (protects against DNS rebinding).
  • Rejects POST requests with a non-local Origin header (CSRF).
  • Requires Content-Type: application/json on POST routes.

Usage:
    python -m app.interfaces.web
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.cleanup.executor import execute_plan
from app.core.assistant import JarvisAssistant
from app.llm.errors import LLMConfigurationError
from app.llm.factory import get_provider_name
from app.llm.nvidia_client import (
    NvidiaAccessError,
    NvidiaAuthenticationError,
    NvidiaConnectionError,
    NvidiaRateLimitError,
    NvidiaResponseError,
)
from app.llm.ollama_client import OllamaConnectionError, OllamaResponseError
from app.pending_plans import (
    NoTargetsSelectedError,
    PlanAlreadyUsedError,
    PlanExpiredError,
    PlanNotFoundError,
    get_action_for_plan,
    get_pending_action,
    reject_plan,
)

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


# ─── Request / response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """A single user message sent from the browser."""

    message: str = Field(..., max_length=32_000)


class ApproveRequest(BaseModel):
    """Approval choices for a stored cleanup plan."""

    plan_id: str = Field(..., min_length=1, max_length=128)
    include_temp: bool = True
    include_recycle_bin: bool = False


class RejectRequest(BaseModel):
    """A stored cleanup plan to cancel."""

    plan_id: str = Field(..., min_length=1, max_length=128)


# ─── Application state ────────────────────────────────────────────────────────

@dataclass
class _WebState:
    """Per-process assistant state shared by all requests."""

    port: int
    factory: Callable[[], Any]
    init_lock: threading.Lock = field(default_factory=threading.Lock)
    chat_lock: threading.Lock = field(default_factory=threading.Lock)
    assistant: Optional[Any] = None
    init_error: Optional[str] = None
    initialised: bool = False


class AssistantUnavailableError(RuntimeError):
    """Raised when the assistant could not be configured at startup."""


# ─── Assistant wiring (same construction as the CLI entry point) ──────────────

def build_assistant() -> JarvisAssistant:
    """
    Build the BOB stack exactly like app/interfaces/cli.py does.

    The LLM backend is chosen by LLM_PROVIDER (see app/llm/factory.py).
    """
    from app.core.laya_router import build_router_from_env
    from app.llm.factory import create_llm_client
    from app.llm.nvidia_client import NvidiaClient
    from app.llm.ollama_client import OllamaClient
    from app.tools.registry import build_default_registry

    llm_client = create_llm_client()
    tool_registry = build_default_registry()

    router = build_router_from_env()
    if router is not None:
        try:
            loaded = router.load()
        except Exception:  # noqa: BLE001 – routing is optional, never fatal
            logger.exception("Laya router failed to load; continuing without routing.")
            loaded = False
        if loaded:
            logger.info("Laya router ready on %s (%.1fs).", router.device, router.load_seconds)
        else:
            logger.warning(
                "Laya router unavailable (%s); continuing without routing.",
                getattr(router, "failure_reason", "unknown error"),
            )
            router = None

    # When the primary provider is NVIDIA, use Ollama as a silent fallback for
    # transient connection failures and rate-limit errors.
    fallback_llm = OllamaClient() if isinstance(llm_client, NvidiaClient) else None

    return JarvisAssistant(
        llm_client=llm_client,
        tool_registry=tool_registry,
        router=router,
        fallback_llm=fallback_llm,
    )


def _ensure_assistant(state: _WebState) -> Optional[Any]:
    """Build the assistant once; return None when configuration failed."""
    if state.initialised:
        return state.assistant

    with state.init_lock:
        if state.initialised:
            return state.assistant
        try:
            state.assistant = state.factory()
        except LLMConfigurationError as exc:
            # Configuration messages are designed to be user-facing and safe.
            state.init_error = str(exc)
            logger.error("BOB assistant configuration error: %s", exc)
        except Exception:  # noqa: BLE001 – keep the server alive
            state.init_error = "BOB could not start the assistant. Check the server logs."
            logger.exception("Failed to initialise the BOB assistant.")
        finally:
            state.initialised = True

    return state.assistant


# ─── Synchronous work run in the threadpool ───────────────────────────────────

def _extract_tools(messages: List[Dict[str, Any]]) -> List[str]:
    """Return unique tool names from the new history entries of a turn."""
    tools: List[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            name = str(fn.get("name") or "").strip() if isinstance(fn, dict) else ""
            if name and name not in tools:
                tools.append(name)
    return tools


def _extract_pending_plan_ids(messages: List[Dict[str, Any]]) -> List[str]:
    """Return plan IDs found in new tool-result history entries."""
    plan_ids: List[str] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        data = payload.get("data")
        if not isinstance(data, dict):
            continue
        plan_id = data.get("plan_id")
        if isinstance(plan_id, str) and plan_id and plan_id not in plan_ids:
            plan_ids.append(plan_id)
    return plan_ids


def _extract_pending_actions(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return current pending-action payloads for plan IDs in a turn."""
    actions: List[Dict[str, Any]] = []
    for plan_id in _extract_pending_plan_ids(messages):
        action = get_action_for_plan(plan_id)
        if action is not None:
            actions.append(action)
    return actions


def _run_chat(state: _WebState, message: str) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Run one assistant turn under the conversation lock."""
    assistant = _ensure_assistant(state)
    if assistant is None:
        raise AssistantUnavailableError(
            state.init_error or "BOB is not available. Check the server logs."
        )

    with state.chat_lock:
        before = len(assistant.history)
        reply = assistant.chat(message)
        new_messages = assistant.history[before:]

    return reply, _extract_tools(new_messages), _extract_pending_actions(new_messages)


def _run_reset(state: _WebState) -> None:
    """Clear conversation history under the same lock as chat."""
    assistant = _ensure_assistant(state)
    if assistant is None:
        raise AssistantUnavailableError(
            state.init_error or "BOB is not available. Check the server logs."
        )

    with state.chat_lock:
        assistant.reset()


def _health_payload(state: _WebState) -> Dict[str, Any]:
    """Return status plus the active provider/model name."""
    assistant = _ensure_assistant(state)
    if assistant is None:
        return {
            "status": "error",
            "provider": get_provider_name(),
            "model": None,
            "tools": [],
            "error": state.init_error,
        }

    llm = getattr(assistant, "_llm", None)
    registry = getattr(assistant, "tool_registry", None)
    return {
        "status": "ok",
        "provider": get_provider_name(),
        "model": getattr(llm, "model", None),
        "tools": registry.list_tools() if registry else [],
    }


# ─── Error helpers ────────────────────────────────────────────────────────────

def _json_error(status_code: int, message: str) -> JSONResponse:
    """Return a friendly JSON error without stack traces or secrets."""
    return JSONResponse(status_code=status_code, content={"error": message})


def _friendly_error(exc: BaseException) -> Tuple[int, str]:
    """Map assistant/LLM failures to safe HTTP statuses and messages."""
    if isinstance(exc, LLMConfigurationError):
        return 503, str(exc) or "BOB is not configured correctly."

    if isinstance(exc, OllamaConnectionError):
        return 503, "Cannot reach Ollama. Make sure it is running (`ollama serve`) and try again."
    if isinstance(exc, OllamaResponseError):
        return 502, "Ollama returned an unexpected response. Check the server logs."

    if isinstance(exc, NvidiaAuthenticationError):
        return 503, "NVIDIA rejected the API key. Check NVIDIA_API_KEY."
    if isinstance(exc, NvidiaRateLimitError):
        return 503, "NVIDIA rate limit exceeded. Wait a moment and try again."
    if isinstance(exc, NvidiaAccessError):
        return 503, "NVIDIA API access or credits problem. Check your NVIDIA account."
    if isinstance(exc, NvidiaConnectionError):
        return 503, "Cannot reach the NVIDIA API. Check your internet connection."
    if isinstance(exc, NvidiaResponseError):
        return 502, "The NVIDIA API returned an error. Check the server logs."

    return 500, "Something went wrong inside BOB. Check the server logs."


def _friendly_plan_error(exc: BaseException) -> Tuple[int, str]:
    """Map cleanup-plan failures to safe HTTP statuses and messages."""
    if isinstance(exc, PlanNotFoundError):
        return 404, "Cleanup plan not found. Ask BOB to scan again."
    if isinstance(exc, PlanExpiredError):
        return 410, "This cleanup plan has expired. Ask BOB to scan again."
    if isinstance(exc, PlanAlreadyUsedError):
        return 409, "This cleanup plan was already used or cancelled."
    if isinstance(exc, NoTargetsSelectedError):
        return 400, "Select at least one cleanup target."
    return 500, "Cleanup failed. Check the server logs."


def _log_failure(action: str, exc: BaseException, status: int) -> None:
    """Log details server-side only; the browser gets a friendly message."""
    if status >= 500 and status != 503:
        logger.exception("Unexpected error during web %s.", action)
    else:
        logger.warning("Web %s failed (%s).", action, type(exc).__name__)


# ─── Port / origin helpers ────────────────────────────────────────────────────

def _configured_port() -> int:
    """Read BOB_WEB_PORT, falling back to 8000 on invalid input."""
    raw = os.environ.get("BOB_WEB_PORT", str(DEFAULT_PORT))
    try:
        port = int(raw)
    except ValueError:
        logger.warning("Invalid BOB_WEB_PORT=%r; using %d.", raw, DEFAULT_PORT)
        return DEFAULT_PORT
    if not 1 <= port <= 65_535:
        logger.warning("BOB_WEB_PORT out of range (%r); using %d.", raw, DEFAULT_PORT)
        return DEFAULT_PORT
    return port


def _allowed_hosts(port: int) -> Set[str]:
    return {f"localhost:{port}", f"127.0.0.1:{port}"}


def _allowed_origins(port: int) -> Set[str]:
    return {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}


# ─── Application factory ──────────────────────────────────────────────────────

def create_app(
    assistant_factory: Optional[Callable[[], Any]] = None,
    *,
    port: Optional[int] = None,
) -> FastAPI:
    """
    Create the FastAPI application.

    :param assistant_factory: Override assistant construction (used by tests).
    :param port:              Port used for Host/Origin validation (default: env).
    """
    resolved_port = port if port is not None else _configured_port()
    state = _WebState(port=resolved_port, factory=assistant_factory or build_assistant)

    app = FastAPI(
        title="BOB Web",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.bob = state

    allowed_hosts = _allowed_hosts(resolved_port)
    allowed_origins = _allowed_origins(resolved_port)

    # ── Security middleware ───────────────────────────────────────────────────
    @app.middleware("http")
    async def _security_guard(request: Request, call_next):
        host = request.headers.get("host", "").strip().lower()
        if host not in allowed_hosts:
            return _json_error(403, "Invalid Host header.")

        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin is not None:
                normalised = origin.strip().rstrip("/").lower()
                if normalised not in allowed_origins:
                    return _json_error(403, "Invalid Origin header.")

            media_type = (
                request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            )
            if media_type != "application/json":
                return _json_error(415, "Content-Type must be application/json.")

        return await call_next(request)

    # ── Validation errors ─────────────────────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, _exc: RequestValidationError):
        return _json_error(
            422, 'Invalid request. Expected a JSON body like {"message": "..."}.'
        )

    @app.exception_handler(Exception)
    async def _unhandled_error(_request: Request, _exc: Exception):
        logger.exception("Unhandled web interface error.")
        return _json_error(500, "Something went wrong inside BOB. Check the server logs.")

    # ── Routes ────────────────────────────────────────────────────────────────
    @app.get("/")
    async def index():
        index_path = WEB_DIR / "index.html"
        if not index_path.is_file():
            return _json_error(500, "The web interface files are missing.")
        return FileResponse(index_path)

    @app.post("/api/chat")
    async def chat(payload: ChatRequest):
        if not payload.message.strip():
            return _json_error(400, "Please enter a message.")
        try:
            reply, tools_used, pending_actions = await run_in_threadpool(
                _run_chat, state, payload.message
            )
        except AssistantUnavailableError as exc:
            return _json_error(503, str(exc))
        except Exception as exc:  # noqa: BLE001 – never leak details to the browser
            status, message = _friendly_error(exc)
            _log_failure("chat", exc, status)
            return _json_error(status, message)
        return {
            "reply": reply,
            "tools_used": tools_used,
            "pending_actions": pending_actions,
        }

    @app.get("/api/pending")
    async def pending():
        action = await run_in_threadpool(get_pending_action)
        return {"pending_actions": [action] if action is not None else []}

    @app.post("/api/approve")
    async def approve(payload: ApproveRequest):
        try:
            result = await run_in_threadpool(
                execute_plan,
                payload.plan_id,
                include_temp=payload.include_temp,
                include_recycle_bin=payload.include_recycle_bin,
            )
        except (PlanNotFoundError, PlanExpiredError, PlanAlreadyUsedError, NoTargetsSelectedError) as exc:
            status, message = _friendly_plan_error(exc)
            _log_failure("approve", exc, status)
            return _json_error(status, message)
        except Exception as exc:  # noqa: BLE001
            status, message = _friendly_plan_error(exc)
            logger.warning("Web approve failed (%s).", type(exc).__name__)
            return _json_error(status, message)
        return result

    @app.post("/api/reject")
    async def reject(payload: RejectRequest):
        try:
            result = await run_in_threadpool(reject_plan, payload.plan_id)
        except (PlanNotFoundError, PlanExpiredError, PlanAlreadyUsedError) as exc:
            status, message = _friendly_plan_error(exc)
            _log_failure("reject", exc, status)
            return _json_error(status, message)
        except Exception as exc:  # noqa: BLE001
            status, message = _friendly_plan_error(exc)
            logger.warning("Web reject failed (%s).", type(exc).__name__)
            return _json_error(status, message)
        return result

    @app.post("/api/reset")
    async def reset():
        try:
            await run_in_threadpool(_run_reset, state)
        except AssistantUnavailableError as exc:
            return _json_error(503, str(exc))
        except Exception as exc:  # noqa: BLE001
            status, message = _friendly_error(exc)
            _log_failure("reset", exc, status)
            return _json_error(status, message)
        return {"status": "reset"}

    @app.get("/api/health")
    async def health():
        return await run_in_threadpool(_health_payload, state)

    # ── Static assets ─────────────────────────────────────────────────────────
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    return app


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Load settings, build the app, and serve on localhost only."""
    from app.main import _configure_logging, _load_dotenv

    _load_dotenv()
    _configure_logging()

    port = _configured_port()
    application = create_app(port=port)
    logger.info("BOB web interface: http://%s:%d", HOST, port)
    uvicorn.run(application, host=HOST, port=port, log_level="info")


if __name__ == "__main__":
    main()
