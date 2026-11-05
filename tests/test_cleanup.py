"""
tests/test_cleanup.py
─────────────────────
Tests for cleanup planning, approval state, and the single deletion path.

All filesystem work uses tmp_path. The real temp folder and Recycle Bin are
never touched.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock

import pytest

from app.cleanup import executor
from app.pending_plans import (
    PlanAlreadyUsedError,
    PlanExpiredError,
    TargetPlan,
    create_plan,
    get_action_for_plan,
    get_pending_action,
    get_plan,
    reject_plan,
    reset_plans_for_tests,
)
from app.tools import propose_cleanup
from app.tools.propose_cleanup import CLEANUP_TARGETS, PROPOSE_CLEANUP_TOOL, _propose_cleanup
from app.tools.registry import build_default_registry
from app.tools.windows_facts import (
    RecycleBinInfo,
    is_within_root,
    scan_files,
)


@pytest.fixture(autouse=True)
def _clean_store():
    reset_plans_for_tests()
    yield
    reset_plans_for_tests()


def _old_file(root: Path, name: str, *, age_hours: float = 25.0, size: int = 128) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    stamp = time.time() - age_hours * 3600
    os.utime(path, (stamp, stamp))
    return path


def _empty_recycle() -> RecycleBinInfo:
    return RecycleBinInfo(count=0, bytes=0)


def _make_temp_plan(tmp_path: Path, files: Tuple[Path, ...] = ()) -> str:
    target = TargetPlan(
        key="temp",
        label="Temporary files older than 24 hours",
        kind="files",
        file_count=len(files),
        bytes=sum(path.stat().st_size for path in files),
        root=str(tmp_path),
        paths=tuple(str(path) for path in files),
        default_include=True,
    )
    plan = create_plan({"temp": target})
    return plan.plan_id


def test_allowlist_is_explicit_and_has_no_path_argument():
    assert set(CLEANUP_TARGETS) == {"temp", "recycle_bin"}
    assert PROPOSE_CLEANUP_TOOL.parameters == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_proposal_only_includes_old_temp_files(tmp_path, monkeypatch):
    old_file = _old_file(tmp_path, "old.tmp", age_hours=25)
    new_file = _old_file(tmp_path, "new.tmp", age_hours=1)
    nested_old = _old_file(tmp_path, Path("nested") / "old.log", age_hours=30)

    monkeypatch.setattr(propose_cleanup, "resolve_temp_root", lambda: tmp_path)
    monkeypatch.setattr(propose_cleanup, "query_recycle_bin", _empty_recycle)

    result = _propose_cleanup()

    assert result.success is True
    assert result.data["plan_id"]
    plan = get_plan(result.data["plan_id"])
    assert plan is not None
    paths = set(plan.targets["temp"].paths)
    assert str(old_file) in paths
    assert str(nested_old) in paths
    assert str(new_file) not in paths


def test_proposal_result_does_not_expose_paths(tmp_path, monkeypatch):
    old_file = _old_file(tmp_path, "secret-name.tmp", age_hours=25)
    monkeypatch.setattr(propose_cleanup, "resolve_temp_root", lambda: tmp_path)
    monkeypatch.setattr(propose_cleanup, "query_recycle_bin", _empty_recycle)

    result = _propose_cleanup()
    encoded = json.dumps(result.to_dict(), default=str)

    assert result.success is True
    assert str(tmp_path) not in encoded
    assert old_file.name not in encoded


def test_proposal_returns_no_plan_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.setattr(propose_cleanup, "resolve_temp_root", lambda: tmp_path)
    monkeypatch.setattr(propose_cleanup, "query_recycle_bin", _empty_recycle)

    result = _propose_cleanup()

    assert result.success is True
    assert result.data["plan_id"] is None
    assert get_pending_action() is None


def test_new_scan_cancels_old_pending_even_if_nothing_found(tmp_path, monkeypatch):
    old = create_plan({"temp": TargetPlan("temp", "Temp", "files", 1, 10)})
    monkeypatch.setattr(propose_cleanup, "resolve_temp_root", lambda: tmp_path)
    monkeypatch.setattr(propose_cleanup, "query_recycle_bin", _empty_recycle)

    result = _propose_cleanup()

    assert result.data["plan_id"] is None
    assert get_plan(old.plan_id).status == "cancelled"
    assert get_pending_action() is None


def test_plan_id_is_random_and_long(tmp_path):
    first = create_plan({"temp": TargetPlan("temp", "Temp", "files", 0, 0)})
    second = create_plan({"temp": TargetPlan("temp", "Temp", "files", 0, 0)})
    reset_plans_for_tests()

    assert len(first.plan_id) >= 22
    assert first.plan_id != second.plan_id


def test_new_plan_cancels_older_pending_plan():
    first = create_plan({"temp": TargetPlan("temp", "Temp", "files", 1, 10)})
    second = create_plan({"temp": TargetPlan("temp", "Temp", "files", 1, 10)})

    assert first.status == "cancelled"
    assert get_action_for_plan(first.plan_id) is None
    assert get_pending_action()["plan_id"] == second.plan_id
    with pytest.raises(PlanAlreadyUsedError):
        executor.execute_plan(first.plan_id, include_temp=True)


def test_path_traversal_is_refused(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    outside_file = outside / "keep.txt"
    outside_file.write_bytes(b"keep")

    assert is_within_root(root, outside_file) is False

    target = TargetPlan(
        key="temp",
        label="Temp",
        kind="files",
        file_count=1,
        bytes=4,
        root=str(root),
        paths=(str(outside_file),),
    )
    result = executor._delete_temp_target(target, now=time.time())

    assert result["files_removed"] == 0
    assert result["files_skipped"] == 1
    assert outside_file.exists()


def test_reparse_parent_is_refused(tmp_path, monkeypatch):
    safe = _old_file(tmp_path, "safe.tmp", age_hours=25)

    monkeypatch.setattr(executor, "has_reparse_parent", lambda _root, _path: True)
    target = TargetPlan(
        key="temp",
        label="Temp",
        kind="files",
        file_count=1,
        bytes=safe.stat().st_size,
        root=str(tmp_path),
        paths=(str(safe),),
    )
    result = executor._delete_temp_target(target, now=time.time())

    assert result["files_removed"] == 0
    assert result["files_skipped"] == 1
    assert safe.exists()


def test_scan_skips_symlinks(tmp_path):
    old_file = _old_file(tmp_path, "target.tmp", age_hours=25)
    link = tmp_path / "link.tmp"
    try:
        os.symlink(old_file, link)
    except OSError:
        pytest.skip("Symlink creation is not permitted on this machine.")

    scan = scan_files(tmp_path, min_age_seconds=0)

    paths = {str(entry.path) for entry in scan.files}
    assert str(old_file) in paths
    assert str(link) not in paths


def test_locked_file_is_skipped(tmp_path, monkeypatch):
    old_file = _old_file(tmp_path, "locked.tmp", age_hours=25)
    target = TargetPlan(
        key="temp",
        label="Temp",
        kind="files",
        file_count=1,
        bytes=old_file.stat().st_size,
        root=str(tmp_path),
        paths=(str(old_file),),
    )

    def _locked(_path):
        raise PermissionError("in use")

    monkeypatch.setattr(executor.os, "remove", _locked)
    result = executor._delete_temp_target(target, now=time.time())

    assert result["files_removed"] == 0
    assert result["files_skipped"] == 1
    assert old_file.exists()


def test_execute_plan_single_use(tmp_path):
    old_file = _old_file(tmp_path, "old.tmp", age_hours=25, size=64)
    plan_id = _make_temp_plan(tmp_path, (old_file,))

    result = executor.execute_plan(plan_id, include_temp=True, include_recycle_bin=False)

    assert result["status"] == "completed"
    assert result["files_removed"] == 1
    assert result["bytes_freed"] == 64
    assert not old_file.exists()
    with pytest.raises(PlanAlreadyUsedError):
        executor.execute_plan(plan_id, include_temp=True)


def test_expired_plan_is_rejected(tmp_path):
    plan = create_plan(
        {"temp": TargetPlan("temp", "Temp", "files", 1, 10, root=str(tmp_path), paths=())}
    )
    plan.expires_at = time.time() - 1

    with pytest.raises(PlanExpiredError):
        executor.execute_plan(plan.plan_id, include_temp=True)
    with pytest.raises(PlanExpiredError):
        reject_plan(plan.plan_id)


def test_reject_consumes_plan(tmp_path):
    plan_id = _make_temp_plan(tmp_path)
    result = reject_plan(plan_id)

    assert result["status"] == "rejected"
    with pytest.raises(PlanAlreadyUsedError):
        executor.execute_plan(plan_id, include_temp=True)


def test_recycle_bin_is_separate_and_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(propose_cleanup, "resolve_temp_root", lambda: tmp_path)
    monkeypatch.setattr(
        propose_cleanup,
        "query_recycle_bin",
        lambda: RecycleBinInfo(count=3, bytes=1000),
    )

    result = _propose_cleanup()
    action = get_pending_action()
    assert result.success is True
    assert action is not None

    targets = {target["key"]: target for target in action["targets"]}
    assert targets["recycle_bin"]["default_include"] is False
    assert "all drives" in targets["recycle_bin"]["warning"].lower()

    calls = {"empty": 0}

    def _fake_empty():
        calls["empty"] += 1

    monkeypatch.setattr(
        executor,
        "query_recycle_bin",
        MagicMock(side_effect=[RecycleBinInfo(3, 1000), RecycleBinInfo(0, 0)]),
    )
    monkeypatch.setattr(executor, "empty_recycle_bin", _fake_empty)

    executed = executor.execute_plan(
        action["plan_id"],
        include_temp=False,
        include_recycle_bin=True,
    )

    assert calls["empty"] == 1
    assert executed["status"] == "completed"
    assert executed["files_removed"] == 3
    assert executed["bytes_freed"] == 1000


def test_execute_plan_only_accepts_plan_id_and_flags():
    signature = inspect.signature(executor.execute_plan)
    assert "plan_id" in signature.parameters
    assert "paths" not in signature.parameters
    with pytest.raises(TypeError):
        executor.execute_plan("missing", include_temp=True, paths=["C:\\outside"])


def test_approve_reject_execute_logs_have_no_paths(tmp_path, caplog):
    marker = "SECRET_PATH_MARKER"
    root = tmp_path / marker
    root.mkdir()
    first = _old_file(root, "first.tmp", age_hours=25)
    second = _old_file(root, "second.tmp", age_hours=25)
    caplog.set_level(logging.INFO)

    execute_id = _make_temp_plan(root, (first,))
    executor.execute_plan(execute_id, include_temp=True, include_recycle_bin=False)

    reject_id = _make_temp_plan(root, (second,))
    reject_plan(reject_id)

    assert marker not in caplog.text


def test_registry_has_no_cleanup_delete_tool():
    names = set(build_default_registry().list_tools())

    assert "diagnose_system" in names
    assert "propose_cleanup" in names
    for forbidden in ("execute_plan", "execute_cleanup", "delete_files", "empty_recycle_bin"):
        assert forbidden not in names


def test_app_tools_never_imports_app_cleanup():
    tools_dir = Path(__file__).resolve().parents[1] / "app" / "tools"
    for source in tools_dir.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "app.cleanup" not in text, f"{source} imports the cleanup execution package"
