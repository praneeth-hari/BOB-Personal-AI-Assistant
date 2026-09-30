"""
tests/test_running_applications.py
──────────────────────────────────
Unit and integration tests for the get_running_applications tool.

Covers:
  • Tool descriptor / JSON schema
  • Normal process discovery and background-process filtering
  • Application-name filtering (aliases, substrings, exe names)
  • No matches
  • Deduplication of repeated process names
  • AccessDenied / NoSuchProcess / ZombieProcess handling
  • Limits
  • Registry registration and execution
  • Live execution against the current machine
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import psutil
import pytest

from app.tools import RUNNING_APPLICATIONS_TOOL as EXPORTED_TOOL
from app.tools.get_running_applications import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MAX_PIDS_PER_APP,
    RUNNING_APPLICATIONS_TOOL,
    _get_running_applications,
)
from app.tools.registry import build_default_registry
from app.tools.tool import ToolResult

ME = "PC\\user"


# ─── Fakes ────────────────────────────────────────────────────────────────────

class FakeProc:
    """Mimics a psutil.Process yielded by process_iter(attrs=...)."""

    def __init__(self, pid: int, name: Optional[str], exe: Optional[str] = None,
                 username: Optional[str] = ME) -> None:
        self.info = {"pid": pid, "name": name, "exe": exe, "username": username}


class RaisingProc:
    """A process whose info access raises (e.g. it vanished mid-scan)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def info(self) -> Dict[str, Any]:
        raise self._exc


def _run(procs: List[Any], current_user: Optional[str] = ME, **kwargs: Any) -> ToolResult:
    with patch("app.tools.get_running_applications.psutil.process_iter", return_value=iter(procs)), \
         patch("app.tools.get_running_applications._current_username", return_value=current_user):
        return _get_running_applications(**kwargs)


def _names(result: ToolResult) -> List[str]:
    return [a["name"] for a in result.data["applications"]]


SAMPLE = [
    FakeProc(4, "System", None, None),
    FakeProc(100, "svchost.exe", r"C:\Windows\System32\svchost.exe", "NT AUTHORITY\\SYSTEM"),
    FakeProc(101, "svchost.exe", r"C:\Windows\System32\svchost.exe", ME),
    FakeProc(200, "chrome.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    FakeProc(201, "chrome.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    FakeProc(202, "chrome.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    FakeProc(300, "Code.exe", r"C:\Users\user\AppData\Local\Programs\Microsoft VS Code\Code.exe"),
    FakeProc(400, "notepad.exe", r"C:\Windows\System32\notepad.exe"),
    FakeProc(500, "MsMpEng.exe", None, None),
    FakeProc(600, "someservice.exe", r"C:\Program Files\Vendor\someservice.exe", "NT AUTHORITY\\SYSTEM"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Descriptor / schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestDescriptor:
    def test_name_and_description(self):
        assert RUNNING_APPLICATIONS_TOOL.name == "get_running_applications"
        desc = RUNNING_APPLICATIONS_TOOL.description.lower()
        assert "running" in desc
        assert "read-only" in desc

    def test_schema(self):
        params = RUNNING_APPLICATIONS_TOOL.parameters
        assert params["type"] == "object"
        assert params["required"] == []
        assert params["properties"]["application"]["type"] == "string"
        assert params["properties"]["limit"]["type"] == "integer"

    def test_llm_schema_format(self):
        schema = RUNNING_APPLICATIONS_TOOL.to_llm_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_running_applications"

    def test_exported_from_package(self):
        assert EXPORTED_TOOL is RUNNING_APPLICATIONS_TOOL


# ═══════════════════════════════════════════════════════════════════════════════
# General discovery
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiscovery:
    def test_lists_user_applications(self):
        result = _run(SAMPLE)
        assert isinstance(result, ToolResult)
        assert result.success is True
        names = _names(result)
        assert "chrome.exe" in names
        assert "Code.exe" in names
        assert "notepad.exe" in names
        assert result.data["total_processes"] == len(SAMPLE)
        assert result.data["application_count"] == 3
        assert result.data["truncated"] is False

    def test_hides_background_and_other_user_processes(self):
        names = _names(_run(SAMPLE))
        assert "svchost.exe" not in names       # background denylist, even for current user
        assert "System" not in names
        assert "MsMpEng.exe" not in names
        assert "someservice.exe" not in names   # owned by SYSTEM
        assert _run(SAMPLE).data["hidden_background_processes"] == 4

    def test_display_names_and_sorted(self):
        apps = _run(SAMPLE).data["applications"]
        display = [a["display_name"] for a in apps]
        assert display == sorted(display, key=str.lower)
        assert "Google Chrome" in display
        assert "Visual Studio Code" in display

    def test_listing_is_compact(self):
        app = _run(SAMPLE).data["applications"][0]
        assert set(app) == {"name", "display_name", "process_count", "pids"}

    def test_falls_back_to_readable_exe_when_user_unknown(self):
        names = _names(_run(SAMPLE, current_user=None))
        assert "chrome.exe" in names
        assert "someservice.exe" in names       # exe readable, owner unknown
        assert "MsMpEng.exe" not in names       # denylisted
        assert "System" not in names            # no exe

    def test_name_falls_back_to_exe_basename(self):
        result = _run([FakeProc(1, None, r"C:\Apps\mystery.exe")])
        assert _names(result) == ["mystery.exe"]

    def test_limit_truncates(self):
        procs = [FakeProc(i, f"app{i:03d}.exe", rf"C:\Apps\app{i:03d}.exe") for i in range(10)]
        result = _run(procs, limit=3)
        assert len(result.data["applications"]) == 3
        assert result.data["application_count"] == 10
        assert result.data["truncated"] is True
        assert "showing first 3" in result.data["message"]

    def test_default_limit_caps_large_lists(self):
        procs = [FakeProc(i, f"app{i:03d}.exe", rf"C:\Apps\app{i:03d}.exe") for i in range(150)]
        result = _run(procs)
        assert len(result.data["applications"]) == DEFAULT_LIMIT

    @pytest.mark.parametrize("limit, expected", [
        (0, 1), (-5, 1), (1000, MAX_LIMIT), ("5", 5), ("abc", DEFAULT_LIMIT), (None, DEFAULT_LIMIT),
    ])
    def test_limit_coercion(self, limit, expected):
        procs = [FakeProc(i, f"app{i:03d}.exe", rf"C:\Apps\app{i:03d}.exe") for i in range(150)]
        result = _run(procs, limit=limit)
        assert len(result.data["applications"]) == expected

    def test_empty_process_list(self):
        result = _run([])
        assert result.success is True
        assert result.data["applications"] == []
        assert result.data["application_count"] == 0

    def test_process_iter_failure_returns_error(self):
        with patch("app.tools.get_running_applications.psutil.process_iter",
                   side_effect=RuntimeError("enumeration failed")):
            result = _get_running_applications()
        assert result.success is False
        assert "enumeration failed" in result.error


# ═══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeduplication:
    def test_duplicate_names_are_grouped(self):
        apps = _run(SAMPLE).data["applications"]
        chrome = [a for a in apps if a["name"] == "chrome.exe"]
        assert len(chrome) == 1
        assert chrome[0]["process_count"] == 3
        assert chrome[0]["pids"] == [200, 201, 202]

    def test_grouping_is_case_insensitive(self):
        procs = [FakeProc(1, "Spotify.exe", r"C:\S\Spotify.exe"), FakeProc(2, "spotify.exe", r"C:\S\spotify.exe")]
        apps = _run(procs).data["applications"]
        assert len(apps) == 1
        assert apps[0]["process_count"] == 2

    def test_pids_are_capped(self):
        procs = [FakeProc(i, "chrome.exe", r"C:\C\chrome.exe") for i in range(30)]
        app = _run(procs).data["applications"][0]
        assert app["process_count"] == 30
        assert len(app["pids"]) == MAX_PIDS_PER_APP


# ═══════════════════════════════════════════════════════════════════════════════
# Application-name filtering
# ═══════════════════════════════════════════════════════════════════════════════

class TestFiltering:
    @pytest.mark.parametrize("query", ["chrome", "Chrome", "google chrome", "chrome.exe", "CHROME"])
    def test_chrome_is_running(self, query):
        result = _run(SAMPLE, application=query)
        assert result.success is True
        data = result.data
        assert data["running"] is True
        assert data["match_count"] == 3
        assert [m["name"] for m in data["matches"]] == ["chrome.exe"]
        assert data["matches"][0]["exe"].endswith("chrome.exe")
        assert data["message"].startswith("Yes")

    @pytest.mark.parametrize("query", ["vs code", "VS Code", "vscode", "vscode.exe", "visual studio code", "code"])
    def test_vscode_aliases(self, query):
        data = _run(SAMPLE, application=query).data
        assert data["running"] is True
        assert data["matches"][0]["name"] == "Code.exe"

    def test_substring_match_on_process_name(self):
        data = _run(SAMPLE, application="note").data
        assert data["running"] is True
        assert data["matches"][0]["name"] == "notepad.exe"

    def test_filter_includes_background_processes(self):
        data = _run(SAMPLE, application="svchost").data
        assert data["running"] is True
        assert data["match_count"] == 2

    def test_no_matches(self):
        data = _run(SAMPLE, application="spotify").data
        assert data["running"] is False
        assert data["match_count"] == 0
        assert data["matches"] == []
        assert data["message"].startswith("No")

    def test_alias_does_not_substring_match(self):
        # "edge" is an alias for msedge.exe only; it must not match e.g. "knowledge.exe".
        procs = [FakeProc(1, "knowledge.exe", r"C:\K\knowledge.exe")]
        assert _run(procs, application="edge").data["running"] is False

    @pytest.mark.parametrize("query", ["", "   ", None])
    def test_blank_query_lists_everything(self, query):
        data = _run(SAMPLE, application=query).data
        assert "applications" in data
        assert "query" not in data

    def test_punctuation_only_query_matches_nothing(self):
        assert _run(SAMPLE, application="!!!").data["running"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Process access errors
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcessErrors:
    @pytest.mark.parametrize("exc", [
        psutil.AccessDenied(pid=10),
        psutil.NoSuchProcess(pid=11),
        psutil.ZombieProcess(pid=12),
        RuntimeError("weird"),
    ])
    def test_raising_process_is_skipped(self, exc):
        procs = [RaisingProc(exc), FakeProc(200, "chrome.exe", r"C:\C\chrome.exe")]
        result = _run(procs)
        assert result.success is True
        assert _names(result) == ["chrome.exe"]
        assert result.data["unreadable_processes"] == 1
        assert result.data["total_processes"] == 2

    def test_unavailable_fields_are_tolerated(self):
        # process_iter(ad_value=None) yields None for fields that raised AccessDenied.
        procs = [FakeProc(1, None, None, None), FakeProc(2, "explorer.exe", None, ME)]
        result = _run(procs)
        assert result.success is True
        assert _names(result) == ["explorer.exe"]
        assert result.data["applications"][0]["display_name"] == "File Explorer"

    def test_filter_survives_access_errors(self):
        procs = [RaisingProc(psutil.AccessDenied(pid=1)), FakeProc(2, "chrome.exe", None)]
        data = _run(procs, application="chrome").data
        assert data["running"] is True
        assert data["matches"][0]["exe"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# Registry integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_registered_in_default_registry(self):
        reg = build_default_registry()
        assert "get_running_applications" in reg.list_tools()
        schema_names = [s["function"]["name"] for s in reg.get_tool_schemas()]
        assert "get_running_applications" in schema_names

    def test_registry_execution_listing(self):
        with patch("app.tools.get_running_applications.psutil.process_iter", return_value=iter(SAMPLE)), \
             patch("app.tools.get_running_applications._current_username", return_value=ME):
            res = build_default_registry().execute("get_running_applications", {})
        assert res["success"] is True
        assert res["error"] is None
        assert res["data"]["application_count"] == 3

    def test_registry_execution_with_filter(self):
        with patch("app.tools.get_running_applications.psutil.process_iter", return_value=iter(SAMPLE)), \
             patch("app.tools.get_running_applications._current_username", return_value=ME):
            res = build_default_registry().execute("get_running_applications", {"application": "chrome"})
        assert res["success"] is True
        assert res["data"]["running"] is True

    def test_registry_result_is_json_serialisable(self):
        res = build_default_registry().execute("get_running_applications", {"limit": 5})
        ToolResult(**res).to_json()


# ═══════════════════════════════════════════════════════════════════════════════
# Live execution (real processes on this machine)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLive:
    def test_live_listing(self):
        result = _get_running_applications()
        assert result.success is True
        data = result.data
        assert data["total_processes"] > 0
        assert len(data["applications"]) <= DEFAULT_LIMIT
        for app in data["applications"]:
            assert app["name"]
            assert app["process_count"] >= 1
            assert len(app["pids"]) <= MAX_PIDS_PER_APP

    def test_live_current_python_process_is_found(self):
        # The test runner itself is always running.
        result = _get_running_applications(application="python")
        assert result.success is True
        assert result.data["running"] is True
        assert result.data["match_count"] >= 1

    def test_live_nonexistent_application(self):
        result = _get_running_applications(application="definitely-not-a-real-app-xyz123")
        assert result.success is True
        assert result.data["running"] is False
