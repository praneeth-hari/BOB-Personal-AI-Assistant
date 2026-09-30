"""
tests/test_tools.py
────────────────────
Unit tests for the BOB tool system.

Covers:
  • Tool / ToolResult abstraction
  • ToolRegistry (registration, lookup, unknown rejection, execution, schemas)
  • get_system_info (structure, success path, psutil mock)
  • open_application (allowlist enforcement, rejection, empty input)
  • search_files (existing dir, nonexistent dir, glob, substring, limit,
                  path safety, non-directory path, empty args)
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ── Imports under test ────────────────────────────────────────────────────────
from app.tools.tool import Tool, ToolResult
from app.tools.registry import ToolRegistry, build_default_registry
from app.tools.system_info import _get_system_info, SYSTEM_INFO_TOOL
from app.tools.open_application import (
    _open_application,
    OPEN_APPLICATION_TOOL,
    ALLOWED_APPLICATIONS,
    SUPPORTED_APPS,
)
from app.tools.close_application import (
    _close_application,
    CLOSE_APPLICATION_TOOL,
    ALLOWED_CLOSE_APPLICATIONS,
    SUPPORTED_CLOSE_APPS,
)
from app.tools.search_files import _search_files, SEARCH_FILES_TOOL, MAX_RESULTS


# ═══════════════════════════════════════════════════════════════════════════════
# ToolResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolResult:
    def test_success_result(self):
        r = ToolResult(success=True, data={"key": "value"})
        assert r.success is True
        assert r.data == {"key": "value"}
        assert r.error is None

    def test_error_result(self):
        r = ToolResult(success=False, error="Something broke")
        assert r.success is False
        assert r.error == "Something broke"
        assert r.data is None

    def test_to_dict(self):
        r = ToolResult(success=True, data=42)
        d = r.to_dict()
        assert d == {"success": True, "data": 42, "error": None}

    def test_to_json_is_valid(self):
        r = ToolResult(success=True, data={"x": 1})
        parsed = json.loads(r.to_json())
        assert parsed["success"] is True
        assert parsed["data"]["x"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tool
# ═══════════════════════════════════════════════════════════════════════════════

def _dummy_fn() -> ToolResult:
    return ToolResult(success=True, data="dummy")


class TestTool:
    def _make_tool(self, name="dummy"):
        return Tool(
            name=name,
            description="A dummy tool",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=_dummy_fn,
        )

    def test_to_llm_schema_structure(self):
        tool = self._make_tool("test_tool")
        schema = tool.to_llm_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A dummy tool"
        assert "parameters" in schema["function"]

    def test_to_llm_schema_parameters_passthrough(self):
        params = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        tool = Tool(name="t", description="d", parameters=params, fn=_dummy_fn)
        assert tool.to_llm_schema()["function"]["parameters"] == params


# ═══════════════════════════════════════════════════════════════════════════════
# ToolRegistry
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(Tool(name="alpha", description="Alpha", parameters={}, fn=_dummy_fn))
    return r


class TestToolRegistry:
    def test_register_and_list(self, registry):
        assert "alpha" in registry.list_tools()

    def test_len(self, registry):
        assert len(registry) == 1

    def test_get_known_tool(self, registry):
        tool = registry.get("alpha")
        assert tool is not None
        assert tool.name == "alpha"

    def test_get_unknown_tool_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_overwrite_warning(self, registry, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="app.tools.registry"):
            registry.register(
                Tool(name="alpha", description="New", parameters={}, fn=_dummy_fn)
            )
        assert "already registered" in caplog.text

    def test_execute_known_tool(self, registry):
        result = registry.execute("alpha", {})
        assert result["success"] is True

    def test_execute_unknown_tool_returns_error(self, registry):
        result = registry.execute("nonexistent", {})
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_execute_bad_args_returns_error(self):
        """Tool that requires an arg; passing none should give error, not raise."""
        def needs_arg(required_param: str) -> ToolResult:
            return ToolResult(success=True, data=required_param)

        reg = ToolRegistry()
        reg.register(Tool(name="needs_arg", description="", parameters={}, fn=needs_arg))
        result = reg.execute("needs_arg", {})   # missing required_param
        assert result["success"] is False
        assert "Invalid arguments" in result["error"]

    def test_get_tool_schemas_returns_list(self, registry):
        schemas = registry.get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"

    def test_get_tool_schemas_contains_name(self, registry):
        schemas = registry.get_tool_schemas()
        assert schemas[0]["function"]["name"] == "alpha"

    def test_build_default_registry_has_all_tools(self):
        reg = build_default_registry()
        names = reg.list_tools()
        assert "get_system_info" in names
        assert "open_application" in names
        assert "search_files" in names
        assert "close_application" in names
        assert "search_computer" in names
        assert "get_battery_status" in names
        assert "get_running_applications" in names
        assert "audio_volume" in names
        assert "screen_brightness" in names
        assert "window_control" in names
        assert "web_search" in names
        assert len(names) == 11

    def test_execute_tool_exception_returns_error(self):
        def exploding_fn() -> ToolResult:
            raise RuntimeError("Boom!")

        reg = ToolRegistry()
        reg.register(Tool(name="boom", description="", parameters={}, fn=exploding_fn))
        result = reg.execute("boom", {})
        assert result["success"] is False
        assert "Boom!" in result["error"]


# ═══════════════════════════════════════════════════════════════════════════════
# System Info Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemInfoTool:
    def test_result_is_tool_result(self):
        result = _get_system_info()
        assert isinstance(result, ToolResult)

    def test_success_on_normal_system(self):
        result = _get_system_info()
        assert result.success is True

    def test_data_has_cpu_key(self):
        result = _get_system_info()
        assert "cpu" in result.data

    def test_data_has_ram_key(self):
        result = _get_system_info()
        assert "ram" in result.data

    def test_data_has_platform_key(self):
        result = _get_system_info()
        assert "platform" in result.data

    def test_data_has_gpu_key(self):
        result = _get_system_info()
        assert "gpu" in result.data
        assert isinstance(result.data["gpu"], list)

    def test_ram_fields_present(self):
        result = _get_system_info()
        ram = result.data["ram"]
        assert "total_gb" in ram
        assert "used_gb" in ram
        assert "free_gb" in ram
        assert "usage_percent" in ram

    def test_cpu_fields_present(self):
        result = _get_system_info()
        cpu = result.data["cpu"]
        assert "usage_percent" in cpu
        assert "logical_cores" in cpu

    def test_platform_fields_present(self):
        result = _get_system_info()
        pl = result.data["platform"]
        assert "system" in pl

    def test_psutil_error_returns_failure(self):
        with patch("app.tools.system_info.psutil.cpu_percent", side_effect=RuntimeError("psutil down")):
            result = _get_system_info()
        assert result.success is False
        assert "psutil down" in result.error

    def test_tool_descriptor_name(self):
        assert SYSTEM_INFO_TOOL.name == "get_system_info"

    def test_tool_schema_has_no_required_params(self):
        schema = SYSTEM_INFO_TOOL.to_llm_schema()
        params = schema["function"]["parameters"]
        assert params.get("required", []) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Open Application Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenApplicationTool:
    def test_allowed_notepad_launches(self):
        with patch("app.tools.open_application.subprocess.Popen") as mock_popen:
            result = _open_application("notepad")
        assert result.success is True
        assert result.data["application"] == "notepad"
        assert result.data["status"] == "launched"
        mock_popen.assert_called_once()

    def test_case_insensitive_alias(self):
        with patch("app.tools.open_application.subprocess.Popen"):
            result = _open_application("Notepad")
        assert result.success is True

    def test_calculator_allowed(self):
        with patch("app.tools.open_application.subprocess.Popen"):
            result = _open_application("calculator")
        assert result.success is True

    def test_vscode_allowed(self):
        with patch("app.tools.open_application.subprocess.Popen"):
            result = _open_application("vscode")
        assert result.success is True

    def test_rejected_application_returns_error(self):
        result = _open_application("cmd")
        assert result.success is False
        assert "allowlist" in result.error.lower()

    def test_rejected_arbitrary_path(self):
        result = _open_application("C:\\Windows\\System32\\cmd.exe")
        assert result.success is False

    def test_rejected_powershell(self):
        result = _open_application("powershell")
        assert result.success is False

    def test_rejected_python(self):
        result = _open_application("python")
        assert result.success is False

    def test_empty_application_name(self):
        result = _open_application("")
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_whitespace_only_name(self):
        result = _open_application("   ")
        assert result.success is False

    def test_popen_uses_shell_false(self):
        with patch("app.tools.open_application.subprocess.Popen") as mock_popen:
            _open_application("notepad")
        _, kwargs = mock_popen.call_args
        assert kwargs.get("shell") is False

    def test_executable_comes_from_allowlist_not_llm_input(self):
        """The actual command must be the allowlist value, not the LLM's raw string."""
        with patch("app.tools.open_application.subprocess.Popen") as mock_popen:
            _open_application("notepad")
        args, _ = mock_popen.call_args
        actual_command = args[0]
        assert actual_command == ALLOWED_APPLICATIONS["notepad"]
        assert "notepad.exe" in actual_command[0].lower()

    def test_file_not_found_returns_error(self):
        with patch("app.tools.open_application.subprocess.Popen",
                   side_effect=FileNotFoundError("not found")):
            result = _open_application("notepad")
        assert result.success is False
        assert "not found" in result.error.lower() or "not installed" in result.error.lower()

    def test_tool_descriptor_name(self):
        assert OPEN_APPLICATION_TOOL.name == "open_application"

    def test_tool_schema_requires_application(self):
        schema = OPEN_APPLICATION_TOOL.to_llm_schema()
        assert "application" in schema["function"]["parameters"]["required"]


# ═══════════════════════════════════════════════════════════════════════════════
# Close Application Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestCloseApplicationTool:
    def _make_mock_proc(self, name: str = "notepad.exe", pid: int = 1234) -> MagicMock:
        mock_proc = MagicMock()
        mock_proc.pid = pid
        mock_proc.info = {"name": name, "pid": pid}
        return mock_proc

    def test_allowed_notepad_closes(self):
        proc = self._make_mock_proc("notepad.exe", 1234)
        with patch("app.tools.close_application.psutil.process_iter", return_value=[proc]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([proc], [])):
            result = _close_application("notepad")
        assert result.success is True
        assert result.data["application"] == "notepad"
        assert result.data["status"] == "closed"
        assert result.data["count"] == 1
        assert result.data["terminated_pids"] == [1234]
        proc.terminate.assert_called_once()

    def test_case_insensitive_alias(self):
        proc = self._make_mock_proc("notepad.exe", 1234)
        with patch("app.tools.close_application.psutil.process_iter", return_value=[proc]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([proc], [])):
            result = _close_application("Notepad")
        assert result.success is True
        assert result.data["application"] == "notepad"

    def test_calculator_allowed(self):
        proc = self._make_mock_proc("CalculatorApp.exe", 2020)
        with patch("app.tools.close_application.psutil.process_iter", return_value=[proc]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([proc], [])):
            result = _close_application("calculator")
        assert result.success is True
        assert result.data["application"] == "calculator"
        proc.terminate.assert_called_once()

    def test_calc_alias_allowed(self):
        proc = self._make_mock_proc("calc.exe", 2021)
        with patch("app.tools.close_application.psutil.process_iter", return_value=[proc]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([proc], [])):
            result = _close_application("calc")
        assert result.success is True
        proc.terminate.assert_called_once()

    def test_paint_allowed(self):
        proc = self._make_mock_proc("mspaint.exe", 3030)
        with patch("app.tools.close_application.psutil.process_iter", return_value=[proc]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([proc], [])):
            result = _close_application("paint")
        assert result.success is True
        proc.terminate.assert_called_once()

    def test_vscode_allowed(self):
        proc = self._make_mock_proc("code.exe", 4040)
        with patch("app.tools.close_application.psutil.process_iter", return_value=[proc]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([proc], [])):
            result = _close_application("vscode")
        assert result.success is True
        proc.terminate.assert_called_once()

    def test_code_alias_allowed(self):
        proc = self._make_mock_proc("code.exe", 4041)
        with patch("app.tools.close_application.psutil.process_iter", return_value=[proc]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([proc], [])):
            result = _close_application("code")
        assert result.success is True
        proc.terminate.assert_called_once()

    def test_rejected_application_returns_error(self):
        for bad_app in ["cmd", "powershell", "python", "taskmgr", "explorer", "chrome"]:
            result = _close_application(bad_app)
            assert result.success is False
            assert "allowlist" in result.error.lower()

    def test_rejected_arbitrary_path(self):
        result = _close_application("C:\\Windows\\System32\\calc.exe")
        assert result.success is False
        assert "allowlist" in result.error.lower()

    def test_rejected_pid_input(self):
        result = _close_application("1234")
        assert result.success is False
        assert "allowlist" in result.error.lower()

    def test_empty_application_name(self):
        result = _close_application("")
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_whitespace_only_name(self):
        result = _close_application("   ")
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_application_not_currently_running(self):
        other_proc = self._make_mock_proc("other.exe", 999)
        with patch("app.tools.close_application.psutil.process_iter", return_value=[other_proc]):
            result = _close_application("notepad")
        assert result.success is False
        assert "not currently running" in result.error.lower()
        assert result.data["count"] == 0

    def test_multiple_instances_terminated(self):
        p1 = self._make_mock_proc("notepad.exe", 1001)
        p2 = self._make_mock_proc("notepad.exe", 1002)
        with patch("app.tools.close_application.psutil.process_iter", return_value=[p1, p2]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([p1, p2], [])):
            result = _close_application("notepad")
        assert result.success is True
        assert result.data["count"] == 2
        assert result.data["terminated_pids"] == [1001, 1002]
        p1.terminate.assert_called_once()
        p2.terminate.assert_called_once()

    def test_access_denied_handled_gracefully(self):
        import psutil
        proc = self._make_mock_proc("notepad.exe", 1234)
        proc.terminate.side_effect = psutil.AccessDenied()
        with patch("app.tools.close_application.psutil.process_iter", return_value=[proc]):
            result = _close_application("notepad")
        assert result.success is False
        assert "access denied" in result.error.lower()

    def test_no_such_process_handled_gracefully(self):
        import psutil
        proc = self._make_mock_proc("notepad.exe", 1234)
        proc.terminate.side_effect = psutil.NoSuchProcess(pid=1234)
        with patch("app.tools.close_application.psutil.process_iter", return_value=[proc]), \
             patch("app.tools.close_application.psutil.wait_procs", return_value=([proc], [])):
            result = _close_application("notepad")
        assert result.success is True
        assert result.data["count"] == 1

    def test_process_iter_exception_handled_gracefully(self):
        with patch("app.tools.close_application.psutil.process_iter", side_effect=RuntimeError("Process scan failed")):
            result = _close_application("notepad")
        assert result.success is False
        assert "process scan failed" in result.error.lower()

    def test_tool_descriptor_name(self):
        assert CLOSE_APPLICATION_TOOL.name == "close_application"

    def test_tool_schema_requires_application(self):
        schema = CLOSE_APPLICATION_TOOL.to_llm_schema()
        assert "application" in schema["function"]["parameters"]["required"]
        assert schema["function"]["name"] == "close_application"


# ═══════════════════════════════════════════════════════════════════════════════
# Search Files Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchFilesTool:
    # ── Fixtures ──────────────────────────────────────────────────────────────

    @pytest.fixture()
    def search_dir(self, tmp_path: Path) -> Path:
        """Create a small temp directory structure for search tests."""
        (tmp_path / "hello.py").write_text("print('hello')")
        (tmp_path / "world.py").write_text("print('world')")
        (tmp_path / "report.txt").write_text("some report")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("# deep")
        (sub / "notes.md").write_text("# notes")
        return tmp_path

    # ── Basic functionality ───────────────────────────────────────────────────

    def test_glob_finds_python_files(self, search_dir):
        result = _search_files(str(search_dir), "*.py")
        assert result.success is True
        names = [r["name"] for r in result.data["results"]]
        assert "hello.py" in names
        assert "world.py" in names
        assert "deep.py" in names

    def test_substring_finds_files(self, search_dir):
        result = _search_files(str(search_dir), "hello")
        assert result.success is True
        assert any(r["name"] == "hello.py" for r in result.data["results"])

    def test_substring_is_case_insensitive(self, search_dir):
        result = _search_files(str(search_dir), "HELLO")
        assert result.success is True
        assert any(r["name"] == "hello.py" for r in result.data["results"])

    def test_match_type_glob(self, search_dir):
        result = _search_files(str(search_dir), "*.py")
        assert result.data["match_type"] == "glob"

    def test_match_type_substring(self, search_dir):
        result = _search_files(str(search_dir), "report")
        assert result.data["match_type"] == "substring"

    def test_result_count_in_data(self, search_dir):
        result = _search_files(str(search_dir), "*.py")
        assert result.data["count"] == len(result.data["results"])

    def test_results_have_path_and_name(self, search_dir):
        result = _search_files(str(search_dir), "*.py")
        for r in result.data["results"]:
            assert "path" in r
            assert "name" in r

    def test_recursive_search(self, search_dir):
        """Files in sub-directories are found."""
        result = _search_files(str(search_dir), "deep.py")
        assert result.success is True
        assert result.data["count"] == 1

    # ── Error cases ───────────────────────────────────────────────────────────

    def test_nonexistent_directory(self, tmp_path):
        result = _search_files(str(tmp_path / "does_not_exist"), "*.py")
        assert result.success is False
        assert "does not exist" in result.error.lower()

    def test_empty_directory_arg(self):
        result = _search_files("", "*.py")
        assert result.success is False

    def test_empty_query_arg(self, search_dir):
        result = _search_files(str(search_dir), "")
        assert result.success is False

    def test_file_path_not_directory(self, search_dir):
        file_path = str(search_dir / "hello.py")
        result = _search_files(file_path, "*.py")
        assert result.success is False
        assert "not a directory" in result.error.lower()

    # ── Result limit ──────────────────────────────────────────────────────────

    def test_result_limit_enforced(self, tmp_path):
        """Create more files than MAX_RESULTS and verify the cap."""
        for i in range(MAX_RESULTS + 10):
            (tmp_path / f"file_{i:04d}.txt").write_text("x")
        result = _search_files(str(tmp_path), "*.txt")
        assert result.success is True
        assert result.data["count"] == MAX_RESULTS
        assert len(result.data["results"]) == MAX_RESULTS
        assert len(result.data["warnings"]) > 0

    # ── No false positives ────────────────────────────────────────────────────

    def test_no_matches_returns_empty_list(self, search_dir):
        result = _search_files(str(search_dir), "*.xyz_no_such_extension")
        assert result.success is True
        assert result.data["count"] == 0

    # ── Tool descriptor ───────────────────────────────────────────────────────

    def test_tool_descriptor_name(self):
        assert SEARCH_FILES_TOOL.name == "search_files"

    def test_tool_schema_requires_directory_and_query(self):
        schema = SEARCH_FILES_TOOL.to_llm_schema()
        required = schema["function"]["parameters"]["required"]
        assert "directory" in required
        assert "query" in required
