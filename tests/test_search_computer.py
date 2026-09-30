"""
tests/test_search_computer.py
──────────────────────────────
Comprehensive unit and integration tests for the search_computer tool.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.core.assistant import JarvisAssistant
from app.tools.tool import Tool, ToolResult
from app.tools.registry import ToolRegistry
from app.tools.search_computer import (
    _search_computer,
    get_allowed_search_roots,
    is_path_blocked,
    is_path_within_allowed_roots,
    normalize_extension,
    resolve_search_locations,
    SEARCH_COMPUTER_TOOL,
    DEFAULT_RESULT_LIMIT,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_fs(tmp_path: Path) -> Path:
    """Create a temporary sandbox filesystem with known files."""
    docs = tmp_path / "Documents"
    docs.mkdir()
    (docs / "resume.pdf").write_bytes(b"%PDF-1.4 dummy")
    (docs / "taxes.pdf").write_bytes(b"%PDF-1.4 taxes")
    (docs / "notes.txt").write_text("plain notes")
    (docs / "script.py").write_text("print('hello')")

    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (downloads / "ebook.PDF").write_bytes(b"%PDF-1.4 ebook")
    (downloads / "report_2025.pdf").write_bytes(b"%PDF-1.4 report")
    (downloads / "data.csv").write_text("a,b,c\n1,2,3")
    (downloads / "analysis.py").write_text("import pandas")

    # Subdirectory with more files
    archive = downloads / "archive"
    archive.mkdir()
    (archive / "old_statement.pdf").write_bytes(b"%PDF-1.4 statement")
    (archive / "backup.py").write_text("# backup")

    # Hidden directory (should be skipped)
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.pdf").write_bytes(b"%PDF secret")

    # AppData directory (should be skipped)
    appdata = tmp_path / "AppData"
    appdata.mkdir()
    (appdata / "cache.pdf").write_bytes(b"%PDF cache")

    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Extension Normalization
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeExtension:
    def test_plain_extension(self):
        assert normalize_extension("pdf") == ".pdf"

    def test_dotted_extension(self):
        assert normalize_extension(".pdf") == ".pdf"

    def test_uppercase_extension(self):
        assert normalize_extension(".PDF") == ".pdf"
        assert normalize_extension("PY") == ".py"

    def test_wildcard_extension(self):
        assert normalize_extension("*.csv") == ".csv"
        assert normalize_extension("*docx") == ".docx"

    def test_whitespace_trimmed(self):
        assert normalize_extension("  .pdf  ") == ".pdf"

    def test_empty_and_none(self):
        assert normalize_extension("") == ""
        assert normalize_extension("   ") == ""
        assert normalize_extension(None) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Security & Path Checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurityPathChecks:
    def test_blocked_windows_directories(self):
        assert is_path_blocked(Path("C:\\Windows")) is True
        assert is_path_blocked(Path("C:\\Windows\\System32")) is True
        assert is_path_blocked(Path("C:\\Program Files")) is True
        assert is_path_blocked(Path("C:\\Program Files (x86)")) is True
        assert is_path_blocked(Path("C:\\ProgramData")) is True
        assert is_path_blocked(Path("C:\\$Recycle.Bin")) is True
        assert is_path_blocked(Path("D:\\System Volume Information")) is True

    def test_allowed_user_paths_not_blocked(self, tmp_path: Path):
        assert is_path_blocked(tmp_path / "Documents") is False
        assert is_path_blocked(tmp_path / "Downloads") is False

    def test_path_within_allowed_roots(self, tmp_path: Path):
        allowed_roots = [tmp_path]
        assert is_path_within_allowed_roots(tmp_path / "Documents", allowed_roots) is True
        assert is_path_within_allowed_roots(tmp_path / "Downloads" / "sub", allowed_roots) is True

    def test_path_outside_allowed_roots_rejected(self, tmp_path: Path):
        allowed_roots = [tmp_path / "allowed"]
        outside = tmp_path / "outside"
        assert is_path_within_allowed_roots(outside, allowed_roots) is False

    def test_resolve_search_locations_blocked_path_raises_permission_error(self):
        with pytest.raises(PermissionError):
            resolve_search_locations("C:\\Windows")

    def test_resolve_search_locations_outside_roots_raises_permission_error(self, tmp_path: Path):
        with patch("app.tools.search_computer.get_allowed_search_roots", return_value=[tmp_path / "safe"]):
            with pytest.raises(PermissionError):
                resolve_search_locations(str(tmp_path / "unsafe"))

    def test_resolve_search_locations_nonexistent_raises_file_not_found(self, tmp_path: Path):
        with patch("app.tools.search_computer.get_allowed_search_roots", return_value=[tmp_path]):
            with pytest.raises(FileNotFoundError):
                resolve_search_locations(str(tmp_path / "nonexistent_dir_12345"))


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: Searching and Counting
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchComputerExecution:
    def test_count_pdfs(self, mock_fs: Path):
        """Count all PDFs across mock search locations (should skip hidden and AppData)."""
        with patch("app.tools.search_computer.resolve_search_locations", return_value=[mock_fs]):
            result = _search_computer(extension=".pdf", count_only=True)

        assert result.success is True
        # resume.pdf, taxes.pdf, ebook.PDF, report_2025.pdf, old_statement.pdf = 5
        assert result.data["count"] == 5
        assert result.data["count_only"] is True
        assert result.data["results"] == []

    def test_count_python_files(self, mock_fs: Path):
        with patch("app.tools.search_computer.resolve_search_locations", return_value=[mock_fs]):
            result = _search_computer(extension="py", count_only=True)

        assert result.success is True
        # script.py, analysis.py, backup.py = 3
        assert result.data["count"] == 3

    def test_extension_without_dot(self, mock_fs: Path):
        with patch("app.tools.search_computer.resolve_search_locations", return_value=[mock_fs]):
            result = _search_computer(extension="csv", count_only=True)

        assert result.success is True
        assert result.data["count"] == 1

    def test_query_filter_substring(self, mock_fs: Path):
        with patch("app.tools.search_computer.resolve_search_locations", return_value=[mock_fs]):
            result = _search_computer(extension=".pdf", query="report", count_only=True)

        assert result.success is True
        assert result.data["count"] == 1

    def test_query_filter_glob(self, mock_fs: Path):
        with patch("app.tools.search_computer.resolve_search_locations", return_value=[mock_fs]):
            result = _search_computer(query="*.txt", count_only=False)

        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["results"][0]["name"] == "notes.txt"

    def test_results_bounded_by_limit(self, mock_fs: Path):
        with patch("app.tools.search_computer.resolve_search_locations", return_value=[mock_fs]):
            result = _search_computer(extension=".pdf", count_only=False, limit=2)

        assert result.success is True
        assert result.data["count"] == 5
        assert len(result.data["results"]) == 2
        assert result.data["limit_reached"] is True

    def test_specific_location_scope(self, mock_fs: Path):
        docs_dir = mock_fs / "Documents"
        with patch("app.tools.search_computer.resolve_search_locations", return_value=[docs_dir]):
            result = _search_computer(extension=".pdf", location="documents", count_only=True)

        assert result.success is True
        # Only resume.pdf and taxes.pdf in Documents
        assert result.data["count"] == 2

    def test_empty_extension_and_query_rejected(self):
        result = _search_computer(extension="", query="")
        assert result.success is False
        assert "specify at least a file extension" in result.error.lower()

    def test_inaccessible_location_returns_error(self):
        result = _search_computer(extension=".pdf", location="C:\\Windows")
        assert result.success is False
        assert "prohibited" in result.error.lower() or "restricted" in result.error.lower()

    def test_permission_error_during_walk_handled_gracefully(self, mock_fs: Path):
        with patch("app.tools.search_computer.resolve_search_locations", return_value=[mock_fs]), \
             patch("os.walk", side_effect=PermissionError("Locked directory")):
            result = _search_computer(extension=".pdf")

        # Scan should succeed with warning, not crash
        assert result.success is True
        assert len(result.data["warnings"]) >= 1
        assert "Permission denied" in result.data["warnings"][0]

    def test_tool_descriptor(self):
        assert SEARCH_COMPUTER_TOOL.name == "search_computer"
        schema = SEARCH_COMPUTER_TOOL.to_llm_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_computer"
        props = schema["function"]["parameters"]["properties"]
        assert "extension" in props
        assert "query" in props
        assert "count_only" in props
        assert "location" in props


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests: Tool Calling & Natural Language Response
# ═══════════════════════════════════════════════════════════════════════════════

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


class TestSearchComputerIntegration:
    def test_how_many_pdfs_executes_search_computer_and_responds_naturally(self, mock_fs: Path):
        """User asking 'How many PDFs are on my PC?' triggers search_computer and returns plain text."""
        llm = ToolCapableMockLLM()
        llm.set_responses(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_computer",
                            "arguments": {"extension": ".pdf", "count_only": True},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "I found 5 PDF files on your computer.",
            },
        )
        registry = ToolRegistry()
        registry.register(SEARCH_COMPUTER_TOOL)
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry)

        with patch("app.tools.search_computer.resolve_search_locations", return_value=[mock_fs]):
            reply = assistant.chat("How many PDFs are on my PC?")

        assert reply == "I found 5 PDF files on your computer."
        assert '{"name":' not in reply

        # Tool result should be in history
        tool_turn = next((m for m in assistant.history if m["role"] == "tool"), None)
        assert tool_turn is not None
        data = json.loads(tool_turn["content"])
        assert data["success"] is True
        assert data["data"]["count"] == 5
        assert data["data"]["count_only"] is True

    def test_find_python_files_code_fence_fallback_integration(self, mock_fs: Path):
        """User asking 'Find Python files on my PC' with text-mode fallback tool call."""
        llm = ToolCapableMockLLM()
        llm.set_responses(
            {
                "role": "assistant",
                "content": '```json\n{"name": "search_computer", "arguments": {"extension": ".py", "count_only": false}}\n```',
            },
            {
                "role": "assistant",
                "content": "I found 3 Python files on your PC.",
            },
        )
        registry = ToolRegistry()
        registry.register(SEARCH_COMPUTER_TOOL)
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry)

        with patch("app.tools.search_computer.resolve_search_locations", return_value=[mock_fs]):
            reply = assistant.chat("Find Python files on my PC.")

        assert reply == "I found 3 Python files on your PC."
        assert "search_computer" not in reply

        tool_turn = next((m for m in assistant.history if m["role"] == "tool"), None)
        assert tool_turn is not None
        data = json.loads(tool_turn["content"])
        assert data["success"] is True
        assert data["data"]["count"] == 3
        assert len(data["data"]["results"]) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# D:\ drive support (all tests use a fake drive under tmp_path — never the real D:)
# ═══════════════════════════════════════════════════════════════════════════════

import sys  # noqa: E402
from contextlib import ExitStack  # noqa: E402
from typing import Optional  # noqa: E402

import app.tools.search_computer as sc  # noqa: E402


@pytest.fixture()
def fake_d(tmp_path: Path) -> Path:
    """A temporary directory standing in for the D:\\ drive root."""
    d = tmp_path / "FakeD"
    d.mkdir()
    (d / "top.pdf").write_bytes(b"%PDF top")
    (d / "Pdf's").mkdir()
    (d / "Pdf's" / "a.pdf").write_bytes(b"%PDF a")
    (d / "Pdf's" / "b.PDF").write_bytes(b"%PDF b")

    # Project that is also the workspace (cwd) — must not be scanned twice
    proj = d / "jarvis"
    proj.mkdir()
    (proj / "main.py").write_text("print(1)")
    (proj / "guide.pdf").write_bytes(b"%PDF guide")
    for skipped in (".venv", "venv", "node_modules", "__pycache__", ".git"):
        (proj / skipped).mkdir()
        (proj / skipped / "hidden.py").write_text("# skip")
        (proj / skipped / "hidden.pdf").write_bytes(b"%PDF skip")

    # OS-style folders at the drive root — must be pruned
    for sysdir in ("Windows", "Program Files", "System Volume Information", "$RECYCLE.BIN", "Recovery"):
        (d / sysdir).mkdir()
        (d / sysdir / "sys.pdf").write_bytes(b"%PDF sys")
        (d / sysdir / "sys.py").write_text("# sys")
    return d


@pytest.fixture()
def fake_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    (home / "Documents" / "resume.pdf").write_bytes(b"%PDF resume")
    (home / "Downloads").mkdir()
    (home / "Downloads" / "tool.py").write_text("# tool")
    return home


def _patch_env(fake_d: Optional[Path], home: Path, cwd: Path) -> ExitStack:
    """Patch D:\\ root, home and cwd so no real machine locations are touched."""
    stack = ExitStack()
    stack.enter_context(patch.object(sc, "get_d_drive_root", return_value=fake_d))
    stack.enter_context(patch.object(sc.Path, "home", return_value=home))
    stack.enter_context(patch.object(sc.Path, "cwd", return_value=cwd))
    stack.enter_context(patch.dict(os.environ, {"JARVIS_SEARCH_ROOTS": ""}))
    return stack


def _as_drive_root(fake_d: Path):
    """Treat fake_d as a drive root for the drive-root pruning logic."""
    real = sc.is_drive_root
    return patch.object(
        sc, "is_drive_root",
        side_effect=lambda p: Path(p) == fake_d or real(p),
    )


class TestDDriveRoots:
    def test_get_d_drive_root_missing_returns_none(self, tmp_path: Path):
        with patch.object(sc, "D_DRIVE_ROOT", tmp_path / "no_such_drive"):
            assert sc.get_d_drive_root() is None

    def test_get_d_drive_root_present(self, fake_d: Path):
        with patch.object(sc, "D_DRIVE_ROOT", fake_d):
            assert sc.get_d_drive_root() == fake_d.resolve()

    def test_allowed_roots_include_d_drive(self, fake_d: Path, fake_home: Path):
        with _patch_env(fake_d, fake_home, fake_home):
            roots = get_allowed_search_roots()
        assert fake_d in roots

    def test_allowed_roots_without_d_drive(self, fake_home: Path):
        with _patch_env(None, fake_home, fake_home):
            roots = get_allowed_search_roots()
        assert roots == [fake_home.resolve()]

    def test_explicit_subfolder_of_d_drive_allowed(self, fake_d: Path, fake_home: Path):
        with _patch_env(fake_d, fake_home, fake_home):
            dirs = resolve_search_locations(str(fake_d / "Pdf's"))
        assert dirs == [(fake_d / "Pdf's").resolve()]

    def test_blocked_folder_on_d_drive_still_rejected(self, fake_d: Path, fake_home: Path):
        with _patch_env(fake_d, fake_home, fake_home):
            with pytest.raises(PermissionError):
                resolve_search_locations(str(fake_d / "Windows"))


class TestDDriveLocationResolution:
    @pytest.mark.parametrize("alias", ["d_drive", "D drive", "d", "D:", "D:\\", "drive d", "d-drive"])
    def test_aliases_resolve_to_d_drive(self, alias: str, fake_d: Path, fake_home: Path):
        with _patch_env(fake_d, fake_home, fake_home):
            assert resolve_search_locations(alias) == [fake_d]

    def test_d_drive_missing_raises(self, fake_home: Path):
        with _patch_env(None, fake_home, fake_home):
            with pytest.raises(FileNotFoundError):
                resolve_search_locations("d_drive")

    def test_search_d_drive_missing_returns_error(self, fake_home: Path):
        with _patch_env(None, fake_home, fake_home):
            result = _search_computer(extension=".pdf", location="d_drive", count_only=True)
        assert result.success is False
        assert "d:" in result.error.lower()

    def test_all_includes_d_drive_and_user_folders(self, fake_d: Path, fake_home: Path, tmp_path: Path):
        ws = tmp_path / "ws"
        ws.mkdir()
        with _patch_env(fake_d, fake_home, ws):
            dirs = resolve_search_locations("all")
        assert fake_d in dirs
        assert (fake_home / "Documents").resolve() in dirs
        assert (fake_home / "Downloads").resolve() in dirs

    @pytest.mark.parametrize("loc", ["all", "", "pc", "my computer"])
    def test_all_does_not_duplicate_workspace_inside_d(self, loc: str, fake_d: Path, fake_home: Path):
        with _patch_env(fake_d, fake_home, fake_d / "jarvis"):
            dirs = resolve_search_locations(loc)
        assert fake_d in dirs
        assert (fake_d / "jarvis") not in dirs
        assert len(dirs) == len(set(dirs))

    def test_all_without_d_drive_unchanged(self, fake_home: Path, tmp_path: Path):
        ws = tmp_path / "ws"
        ws.mkdir()
        with _patch_env(None, fake_home, ws):
            dirs = resolve_search_locations("all")
        assert dirs == [
            (fake_home / "Documents").resolve(),
            (fake_home / "Downloads").resolve(),
            ws,
        ]


class TestDedupeNestedRoots:
    def test_drops_nested_and_duplicates(self, tmp_path: Path):
        d = tmp_path / "D"
        roots = [tmp_path / "home" / "Docs", d / "jarvis", d, d, d / "a" / "b"]
        assert sc.dedupe_nested_roots(roots) == [tmp_path / "home" / "Docs", d]

    def test_sibling_prefix_not_treated_as_nested(self, tmp_path: Path):
        roots = [tmp_path / "jarvis", tmp_path / "jarvis2"]
        assert sc.dedupe_nested_roots(roots) == roots


class TestDDriveScanning:
    def test_count_pdfs_on_d_drive(self, fake_d: Path, fake_home: Path):
        with _patch_env(fake_d, fake_home, fake_d / "jarvis"), _as_drive_root(fake_d):
            result = _search_computer(extension=".pdf", location="d_drive", count_only=True)
        assert result.success is True
        # top.pdf, Pdf's/a.pdf, Pdf's/b.PDF, jarvis/guide.pdf
        assert result.data["count"] == 4
        assert result.data["results"] == []
        assert result.data["roots_searched"] == [str(fake_d)]
        assert result.data["excluded_dirs_skipped"] >= 10

    def test_find_python_files_on_d_drive_skips_excluded(self, fake_d: Path, fake_home: Path):
        with _patch_env(fake_d, fake_home, fake_d / "jarvis"), _as_drive_root(fake_d):
            result = _search_computer(extension=".py", location="d_drive")
        assert result.success is True
        assert result.data["count"] == 1
        paths = [r["path"] for r in result.data["results"]]
        assert paths == [str(fake_d / "jarvis" / "main.py")]

    def test_all_counts_d_drive_files_once(self, fake_d: Path, fake_home: Path):
        with _patch_env(fake_d, fake_home, fake_d / "jarvis"), _as_drive_root(fake_d):
            result = _search_computer(extension=".pdf", location="all", count_only=True)
        assert result.success is True
        # 4 on D + resume.pdf in Documents; jarvis/guide.pdf counted once
        assert result.data["count"] == 5
        assert str(fake_d / "jarvis") not in result.data["roots_searched"]
        assert str(fake_d) in result.data["roots_searched"]

    def test_os_folders_only_pruned_at_drive_root(self, fake_d: Path, fake_home: Path):
        nested = fake_d / "Pdf's" / "Recovery"
        nested.mkdir()
        (nested / "kept.pdf").write_bytes(b"%PDF kept")
        with _patch_env(fake_d, fake_home, fake_home), _as_drive_root(fake_d):
            result = _search_computer(extension=".pdf", location="d_drive", count_only=True)
        assert result.data["count"] == 5

    def test_results_bounded_on_d_drive(self, fake_d: Path, fake_home: Path):
        with _patch_env(fake_d, fake_home, fake_home), _as_drive_root(fake_d):
            result = _search_computer(extension=".pdf", location="d_drive", limit=2)
        assert result.data["count"] == 4
        assert len(result.data["results"]) == 2
        assert result.data["limit_reached"] is True

    def test_failed_root_not_reported_as_searched(self, fake_d: Path, fake_home: Path):
        with _patch_env(fake_d, fake_home, fake_home), \
             patch("os.walk", side_effect=PermissionError("denied")):
            result = _search_computer(extension=".pdf", location="d_drive", count_only=True)
        assert result.data["roots_searched"] == []
        assert result.data["locations_searched"] == [str(fake_d)]


class TestLinkAndJunctionSafety:
    def test_linked_dirs_are_not_traversed(self, tmp_path: Path):
        (tmp_path / "real").mkdir()
        (tmp_path / "real" / "r.pdf").write_bytes(b"%PDF")
        (tmp_path / "link").mkdir()
        (tmp_path / "link" / "l.pdf").write_bytes(b"%PDF")
        link_path = str(tmp_path / "link")
        with patch("app.tools.search_computer.resolve_search_locations", return_value=[tmp_path]), \
             patch.object(sc, "is_link_or_junction", side_effect=lambda p: p == link_path):
            result = _search_computer(extension=".pdf", count_only=True)
        assert result.data["count"] == 1

    @pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions are Windows-only")
    def test_real_junction_not_followed(self, tmp_path: Path):
        import _winapi
        target = tmp_path / "target"
        target.mkdir()
        (target / "doc.pdf").write_bytes(b"%PDF")
        junction = tmp_path / "junction"
        _winapi.CreateJunction(str(target), str(junction))
        assert sc.is_link_or_junction(str(junction)) is True
        assert sc.is_link_or_junction(str(target)) is False
        with patch("app.tools.search_computer.resolve_search_locations", return_value=[tmp_path]):
            result = _search_computer(extension=".pdf", count_only=True)
        assert result.data["count"] == 1  # only via target, not via the junction

    def test_unreadable_path_treated_as_unsafe(self, tmp_path: Path):
        assert sc.is_link_or_junction(str(tmp_path / "missing")) is True


class TestDDriveIntegration:
    def _run(self, question: str, args: Dict[str, Any], fake_d: Path, fake_home: Path):
        llm = ToolCapableMockLLM()
        llm.set_responses(
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "search_computer", "arguments": args}}]},
            {"role": "assistant", "content": "Done."},
        )
        registry = ToolRegistry()
        registry.register(SEARCH_COMPUTER_TOOL)
        assistant = JarvisAssistant(llm_client=llm, tool_registry=registry)
        with _patch_env(fake_d, fake_home, fake_d / "jarvis"), _as_drive_root(fake_d):
            reply = assistant.chat(question)
        tool_turn = next(m for m in assistant.history if m["role"] == "tool")
        return reply, json.loads(tool_turn["content"])

    def test_how_many_pdfs_on_d_drive(self, fake_d: Path, fake_home: Path):
        reply, data = self._run(
            "How many PDFs are on my D drive?",
            {"extension": ".pdf", "count_only": True, "location": "d_drive"},
            fake_d, fake_home,
        )
        assert reply == "Done."
        assert data["success"] is True
        assert data["data"]["count"] == 4
        assert data["data"]["roots_searched"] == [str(fake_d)]

    def test_find_python_files_on_d_drive(self, fake_d: Path, fake_home: Path):
        _, data = self._run(
            "Find Python files on D drive.",
            {"extension": ".py", "location": "D drive"},
            fake_d, fake_home,
        )
        assert data["data"]["count"] == 1
        assert data["data"]["results"][0]["name"] == "main.py"

    def test_how_many_pdfs_on_pc_includes_d_drive(self, fake_d: Path, fake_home: Path):
        _, data = self._run(
            "How many PDFs are on my PC?",
            {"extension": ".pdf", "count_only": True},
            fake_d, fake_home,
        )
        assert data["data"]["count"] == 5
        assert str(fake_d) in data["data"]["roots_searched"]

    def test_tool_description_mentions_d_drive(self):
        desc = SEARCH_COMPUTER_TOOL.description.lower()
        loc_desc = SEARCH_COMPUTER_TOOL.parameters["properties"]["location"]["description"]
        assert "d drive" in desc
        assert "d_drive" in loc_desc
