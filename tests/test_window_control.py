"""
tests/test_window_control.py
────────────────────────────
Comprehensive unit and integration tests for BOB window_control tool.

Tests:
  • Tool descriptor and schema validation
  • Valid operation names and invalid operation handling
  • Missing window_query for focus/minimize/maximize/restore
  • Extra/unnecessary arguments handled gracefully
  • 'list' operation (bounded count, default 40, max 100)
  • 'get_active' operation (found and not found)
  • 'focus', 'minimize', 'maximize', 'restore' operations
  • Case-insensitive matching (title and process_name)
  • Substring matching
  • No-match behavior (found=False)
  • Ambiguous multiple-match behavior (ambiguous=True, matches list)
  • Window filtering (invisible, tool windows, empty titles, cloaked)
  • Windows API error handling
  • Registry discovery
  • Security: No subprocess, os.system, or shell execution
  • Live integration test with state restoration
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.tools.tool import ToolResult
from app.tools.registry import build_default_registry
from app.tools.window_control import (
    _window_control,
    _get_window_info,
    _get_window_state,
    _get_process_info,
    _enumerate_windows,
    WINDOW_CONTROL_TOOL,
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    SUPPORTED_OPERATIONS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Mock Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock_window(
    hwnd: int = 1001,
    title: str = "Test App - Window",
    proc_name: str = "testapp.exe",
    pid: int = 5000,
    is_minimized: bool = False,
    is_maximized: bool = False,
) -> Dict[str, Any]:
    return {
        "hwnd": hwnd,
        "title": title,
        "process_name": proc_name,
        "process_id": pid,
        "is_minimized": is_minimized,
        "is_maximized": is_maximized,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Descriptor & Schema Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestWindowControlDescriptor:
    def test_descriptor_properties(self):
        assert WINDOW_CONTROL_TOOL.name == "window_control"
        assert callable(WINDOW_CONTROL_TOOL.fn)
        assert "application windows" in WINDOW_CONTROL_TOOL.description.lower()

    def test_schema_structure(self):
        schema = WINDOW_CONTROL_TOOL.to_llm_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "window_control"
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "operation" in params["properties"]
        assert "window_query" in params["properties"]
        assert "limit" in params["properties"]
        assert "operation" in params["required"]
        assert "window_query" not in params["required"]  # only required dynamically for actions

    def test_schema_operations_enum(self):
        schema = WINDOW_CONTROL_TOOL.to_llm_schema()
        ops = schema["function"]["parameters"]["properties"]["operation"]["enum"]
        assert set(ops) == SUPPORTED_OPERATIONS


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter & Operation Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestOperationValidation:
    def test_invalid_operation_returns_error(self):
        result = _window_control("kill")
        assert result.success is False
        assert "Invalid operation 'kill'" in result.error

    def test_empty_operation_returns_error(self):
        result = _window_control("")
        assert result.success is False
        assert "Invalid operation" in result.error

    @pytest.mark.parametrize("op", ["focus", "minimize", "maximize", "restore"])
    def test_missing_window_query_for_actions(self, op: str):
        result = _window_control(op)
        assert result.success is False
        assert "window_query" in result.error.lower()
        assert op in result.error.lower()

    @pytest.mark.parametrize("op", ["focus", "minimize", "maximize", "restore"])
    def test_whitespace_window_query_for_actions(self, op: str):
        result = _window_control(op, window_query="   ")
        assert result.success is False
        assert "window_query" in result.error.lower()

    def test_extra_arguments_ignored_safely(self):
        with patch("app.tools.window_control._enumerate_windows", return_value=[]):
            result = _window_control(
                operation="list",
                unknown_arg="ignored",
                another_arg=12345,
            )
        assert result.success is True
        assert result.data["operation"] == "list"


# ═══════════════════════════════════════════════════════════════════════════════
# 'list' Operation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestListOperation:
    def test_list_returns_windows(self):
        fake_windows = [
            _make_mock_window(101, "Google Chrome", "chrome.exe", 1000),
            _make_mock_window(102, "Visual Studio Code", "code.exe", 2000),
        ]
        with patch("app.tools.window_control._enumerate_windows", return_value=fake_windows):
            result = _window_control("list")

        assert result.success is True
        assert result.data["operation"] == "list"
        assert result.data["count"] == 2
        assert len(result.data["windows"]) == 2
        assert result.data["windows"][0]["title"] == "Google Chrome"
        assert result.data["windows"][0]["process_name"] == "chrome.exe"
        assert "hwnd" not in result.data["windows"][0]  # HWND should not be exposed to LLM

    def test_list_default_limit_applied(self):
        fake_windows = [_make_mock_window(i, f"App {i}") for i in range(60)]
        with patch("app.tools.window_control._enumerate_windows", return_value=fake_windows):
            result = _window_control("list")

        assert result.success is True
        assert result.data["count"] == DEFAULT_LIST_LIMIT
        assert len(result.data["windows"]) == DEFAULT_LIST_LIMIT

    def test_list_custom_limit_clamped_to_max(self):
        fake_windows = [_make_mock_window(i, f"App {i}") for i in range(150)]
        with patch("app.tools.window_control._enumerate_windows", return_value=fake_windows):
            result = _window_control("list", limit=200)

        assert result.success is True
        assert result.data["count"] == MAX_LIST_LIMIT
        assert len(result.data["windows"]) == MAX_LIST_LIMIT

    def test_list_empty_returns_zero(self):
        with patch("app.tools.window_control._enumerate_windows", return_value=[]):
            result = _window_control("list")

        assert result.success is True
        assert result.data["count"] == 0
        assert result.data["windows"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# 'get_active' Operation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetActiveOperation:
    def test_get_active_found(self):
        mock_info = _make_mock_window(2001, "Notepad - notes.txt", "notepad.exe", 3300)
        with patch("win32gui.GetForegroundWindow", return_value=2001), \
             patch("win32gui.IsWindow", return_value=True), \
             patch("app.tools.window_control._get_window_info", return_value=mock_info):
            result = _window_control("get_active")

        assert result.success is True
        data = result.data
        assert data["operation"] == "get_active"
        assert data["found"] is True
        assert data["title"] == "Notepad - notes.txt"
        assert data["process_name"] == "notepad.exe"
        assert data["process_id"] == 3300
        assert "Active window" in data["message"]

    def test_get_active_no_foreground_window(self):
        with patch("win32gui.GetForegroundWindow", return_value=0):
            result = _window_control("get_active")

        assert result.success is True
        assert result.data["operation"] == "get_active"
        assert result.data["found"] is False
        assert result.data["title"] is None
        assert "No active foreground window" in result.data["message"]


# ═══════════════════════════════════════════════════════════════════════════════
# Matching & Ambiguity Resolution Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWindowMatching:
    @pytest.fixture()
    def sample_windows(self) -> List[Dict[str, Any]]:
        return [
            _make_mock_window(101, "Untitled - Notepad", "notepad.exe", 1000),
            _make_mock_window(102, "Google Chrome - GitHub", "chrome.exe", 2000),
            _make_mock_window(103, "Spotify Free", "spotify.exe", 3000),
        ]

    def test_case_insensitive_matching(self, sample_windows):
        with patch("app.tools.window_control._enumerate_windows", return_value=sample_windows), \
             patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.ShowWindow"), \
             patch("win32gui.BringWindowToTop"), \
             patch("app.tools.window_control._force_foreground_window"):
            result = _window_control("focus", window_query="notepad")

        assert result.success is True
        assert result.data["found"] is True
        assert result.data["title"] == "Untitled - Notepad"

    def test_process_name_matching(self, sample_windows):
        with patch("app.tools.window_control._enumerate_windows", return_value=sample_windows), \
             patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.ShowWindow"):
            result = _window_control("minimize", window_query="chrome")

        assert result.success is True
        assert result.data["found"] is True
        assert result.data["title"] == "Google Chrome - GitHub"

    def test_substring_matching(self, sample_windows):
        with patch("app.tools.window_control._enumerate_windows", return_value=sample_windows), \
             patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.ShowWindow"):
            result = _window_control("maximize", window_query="GitHub")

        assert result.success is True
        assert result.data["found"] is True
        assert result.data["title"] == "Google Chrome - GitHub"

    def test_no_match_returns_found_false(self, sample_windows):
        with patch("app.tools.window_control._enumerate_windows", return_value=sample_windows):
            result = _window_control("focus", window_query="NonExistentApp123")

        assert result.success is True
        assert result.data["found"] is False
        assert "No visible window matching" in result.data["message"]

    def test_ambiguous_multiple_matches(self):
        ambiguous_windows = [
            _make_mock_window(101, "Document 1 - Notepad", "notepad.exe", 1001),
            _make_mock_window(102, "Document 2 - Notepad", "notepad.exe", 1002),
        ]
        with patch("app.tools.window_control._enumerate_windows", return_value=ambiguous_windows):
            result = _window_control("focus", window_query="Notepad")

        assert result.success is True
        assert result.data.get("ambiguous") is True
        assert len(result.data["matches"]) == 2
        assert "Multiple windows matched" in result.data["message"]


# ═══════════════════════════════════════════════════════════════════════════════
# Action Operations ('focus', 'minimize', 'maximize', 'restore')
# ═══════════════════════════════════════════════════════════════════════════════

class TestWindowActions:
    def test_focus_action(self):
        w = _make_mock_window(101, "Notepad", "notepad.exe", 1000, is_minimized=True)
        with patch("app.tools.window_control._enumerate_windows", return_value=[w]), \
             patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.ShowWindow") as mock_show, \
             patch("win32gui.BringWindowToTop") as mock_top, \
             patch("app.tools.window_control._force_foreground_window") as mock_force:
            result = _window_control("focus", window_query="Notepad")

        assert result.success is True
        assert result.data["operation"] == "focus"
        mock_show.assert_called_once()
        mock_top.assert_called_once_with(101)
        mock_force.assert_called_once_with(101)
        assert "Brought window 'Notepad' to the foreground" in result.data["message"]

    def test_minimize_action(self):
        w = _make_mock_window(101, "Notepad", "notepad.exe", 1000)
        with patch("app.tools.window_control._enumerate_windows", return_value=[w]), \
             patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.ShowWindow") as mock_show:
            result = _window_control("minimize", window_query="Notepad")

        assert result.success is True
        assert result.data["operation"] == "minimize"
        assert "Minimized window 'Notepad'" in result.data["message"]
        import win32con
        mock_show.assert_called_once_with(101, win32con.SW_MINIMIZE)

    def test_maximize_action(self):
        w = _make_mock_window(101, "Notepad", "notepad.exe", 1000)
        with patch("app.tools.window_control._enumerate_windows", return_value=[w]), \
             patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.ShowWindow") as mock_show:
            result = _window_control("maximize", window_query="Notepad")

        assert result.success is True
        assert result.data["operation"] == "maximize"
        assert "Maximized window 'Notepad'" in result.data["message"]
        import win32con
        mock_show.assert_called_once_with(101, win32con.SW_MAXIMIZE)

    def test_restore_action(self):
        w = _make_mock_window(101, "Notepad", "notepad.exe", 1000, is_minimized=True)
        with patch("app.tools.window_control._enumerate_windows", return_value=[w]), \
             patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.ShowWindow") as mock_show:
            result = _window_control("restore", window_query="Notepad")

        assert result.success is True
        assert result.data["operation"] == "restore"
        assert "Restored window 'Notepad'" in result.data["message"]
        import win32con
        mock_show.assert_called_once_with(101, win32con.SW_RESTORE)

    def test_disappeared_window_returns_error(self):
        w = _make_mock_window(101, "Notepad", "notepad.exe", 1000)
        with patch("app.tools.window_control._enumerate_windows", return_value=[w]), \
             patch("win32gui.IsWindow", return_value=False):
            result = _window_control("focus", window_query="Notepad")

        assert result.success is False
        assert "no longer accessible" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Window Filtering Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWindowFiltering:
    def test_invisible_window_skipped(self):
        with patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.IsWindowVisible", return_value=False):
            info = _get_window_info(9999)
        assert info is None

    def test_empty_title_skipped(self):
        with patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.IsWindowVisible", return_value=True), \
             patch("win32gui.GetWindowText", return_value="   "):
            info = _get_window_info(9999)
        assert info is None

    def test_system_ignored_title_skipped(self):
        with patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.IsWindowVisible", return_value=True), \
             patch("win32gui.GetWindowText", return_value="Program Manager"):
            info = _get_window_info(9999)
        assert info is None

    def test_cloaked_window_skipped(self):
        import win32con
        with patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.IsWindowVisible", return_value=True), \
             patch("win32gui.GetWindowText", return_value="Cloaked App"), \
             patch("win32gui.GetWindowLong", return_value=0), \
             patch("win32gui.GetWindowRect", return_value=(0, 0, 500, 500)), \
             patch("win32gui.IsIconic", return_value=False), \
             patch("app.tools.window_control._is_cloaked", return_value=True):
            info = _get_window_info(9999)
        assert info is None

    def test_tool_window_without_appwindow_skipped(self):
        import win32con
        with patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.IsWindowVisible", return_value=True), \
             patch("win32gui.GetWindowText", return_value="Tooltip"), \
             patch("win32gui.GetWindowLong", return_value=win32con.WS_EX_TOOLWINDOW):
            info = _get_window_info(9999)
        assert info is None


# ═══════════════════════════════════════════════════════════════════════════════
# Error Handling & Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_api_exception_during_list(self):
        with patch("app.tools.window_control._enumerate_windows", side_effect=RuntimeError("Win32 failure")):
            result = _window_control("list")

        assert result.success is False
        assert "Failed to list windows" in result.error

    def test_api_exception_during_action(self):
        w = _make_mock_window(101, "Notepad", "notepad.exe", 1000)
        with patch("app.tools.window_control._enumerate_windows", return_value=[w]), \
             patch("win32gui.IsWindow", return_value=True), \
             patch("win32gui.ShowWindow", side_effect=RuntimeError("ShowWindow failed")):
            result = _window_control("minimize", window_query="Notepad")

        assert result.success is False
        assert "ShowWindow failed" in result.error


# ═══════════════════════════════════════════════════════════════════════════════
# Registry Discovery
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegistryDiscovery:
    def test_registry_contains_window_control(self):
        registry = build_default_registry()
        assert "window_control" in registry.list_tools()
        tool = registry.get("window_control")
        assert tool is not None
        assert tool.name == "window_control"

    def test_registry_executes_window_control_list(self):
        registry = build_default_registry()
        with patch("app.tools.window_control._enumerate_windows", return_value=[]):
            res_dict = registry.execute("window_control", {"operation": "list"})
        assert res_dict["success"] is True
        assert res_dict["data"]["operation"] == "list"


# ═══════════════════════════════════════════════════════════════════════════════
# Security Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWindowControlSecurity:
    def test_no_shell_or_subprocess_imported(self):
        import ast
        import app.tools.window_control as wc_mod
        source = inspect.getsource(wc_mod)

        # Ensure no subprocess, cmd, or shell invocation exists
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        assert "subprocess" not in imported_modules
        assert "os" not in imported_modules
        assert "subprocess" not in source.split('"""', 2)[-1]  # code body after module docstring
        assert "os.system" not in source
        assert "shell=True" not in source
        assert "cmd.exe" not in source.split('"""', 2)[-1].lower()
        assert "powershell" not in source.split('"""', 2)[-1].lower()

    def test_cannot_pass_hwnd_as_command(self):
        # Even if an integer or raw HWND string is passed as query, it only matches titles/names
        fake_windows = [_make_mock_window(9999, "Real App", "realapp.exe", 1234)]
        with patch("app.tools.window_control._enumerate_windows", return_value=fake_windows):
            result = _window_control("focus", window_query="9999")

        # It matches against title/process name, not raw HWND
        assert result.success is True
        assert result.data["found"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Live Integration Test (Clean State Restoration)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiveIntegration:
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_live_window_enumeration_and_active(self):
        """Verify real Win32 enumeration and get_active without altering state."""
        list_res = _window_control("list")
        assert list_res.success is True
        assert isinstance(list_res.data["windows"], list)

        active_res = _window_control("get_active")
        assert active_res.success is True
        assert "operation" in active_res.data
