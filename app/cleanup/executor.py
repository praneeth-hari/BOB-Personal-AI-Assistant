"""
app/cleanup/executor.py
───────────────────────
The only code path that deletes cleanup targets.

This module is deliberately outside app/tools and is never wrapped in a Tool
descriptor. The web interface calls execute_plan only after the user clicks
Approve.
"""

from __future__ import annotations

import ctypes
import logging
import os
import stat
import time
from pathlib import Path
from typing import Dict, List

from app.pending_plans import (
    TargetPlan,
    begin_execution,
    mark_completed,
    mark_failed,
)
from app.tools.windows_facts import (
    TEMP_FILE_MIN_AGE_SECONDS,
    has_reparse_parent,
    is_within_root,
    query_recycle_bin,
)

logger = logging.getLogger(__name__)

SHERB_NOCONFIRMATION = 0x00000001
SHERB_NOPROGRESSUI = 0x00000002
SHERB_NOSOUND = 0x00000004


def empty_recycle_bin() -> None:
    """Empty the current user's Recycle Bin on all drives."""
    if os.name != "nt":
        return
    flags = SHERB_NOCONFIRMATION | SHERB_NOPROGRESSUI | SHERB_NOSOUND
    shell32 = ctypes.windll.shell32
    shell32.SHEmptyRecycleBinW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong]
    shell32.SHEmptyRecycleBinW.restype = ctypes.c_long
    hr = shell32.SHEmptyRecycleBinW(None, None, flags)
    if hr != 0:
        raise OSError(f"SHEmptyRecycleBinW failed with HRESULT {hr}")


def _delete_temp_target(
    target: TargetPlan,
    *,
    now: float,
    min_age_seconds: int = TEMP_FILE_MIN_AGE_SECONDS,
) -> Dict[str, object]:
    removed = 0
    freed = 0
    skipped = 0

    if not target.root:
        return {
            "key": target.key,
            "files_removed": 0,
            "bytes_freed": 0,
            "files_skipped": target.file_count,
        }

    root = Path(target.root)
    for path_text in target.paths:
        path = Path(path_text)
        try:
            if not is_within_root(root, path):
                skipped += 1
                continue
            if has_reparse_parent(root, path):
                skipped += 1
                continue

            st = os.lstat(path)
            if not stat.S_ISREG(st.st_mode):
                skipped += 1
                continue
            if now - st.st_mtime < min_age_seconds:
                skipped += 1
                continue

            os.remove(path)
            removed += 1
            freed += int(st.st_size)
        except OSError:
            skipped += 1
        except Exception:  # noqa: BLE001
            skipped += 1

    return {
        "key": target.key,
        "files_removed": removed,
        "bytes_freed": freed,
        "files_skipped": skipped,
    }


def _execute_recycle_bin(target: TargetPlan) -> Dict[str, object]:
    before = query_recycle_bin()
    empty_recycle_bin()
    after = query_recycle_bin()
    removed = max(0, before.count - after.count)
    freed = max(0, before.bytes - after.bytes)
    return {
        "key": target.key,
        "files_removed": removed,
        "bytes_freed": freed,
        "files_skipped": 0,
    }


def execute_plan(
    plan_id: str,
    *,
    include_temp: bool = True,
    include_recycle_bin: bool = False,
) -> Dict[str, object]:
    """
    Execute a stored plan once, using only paths stored in the plan.

    The HTTP body supplies only plan_id and the two include flags. It can never
    supply filesystem paths.
    """
    plan, selected = begin_execution(
        plan_id,
        include_temp=include_temp,
        include_recycle_bin=include_recycle_bin,
    )

    try:
        now = time.time()
        target_results: List[Dict[str, object]] = []
        errors: List[str] = []

        for key in selected:
            target = plan.targets[key]
            try:
                if key == "temp":
                    target_results.append(_delete_temp_target(target, now=now))
                elif key == "recycle_bin":
                    target_results.append(_execute_recycle_bin(target))
            except Exception:  # noqa: BLE001
                errors.append(f"{key} could not be cleaned.")
                target_results.append(
                    {
                        "key": key,
                        "files_removed": 0,
                        "bytes_freed": 0,
                        "files_skipped": target.file_count,
                    }
                )

        files_removed = sum(int(item["files_removed"]) for item in target_results)
        bytes_freed = sum(int(item["bytes_freed"]) for item in target_results)
        files_skipped = sum(int(item["files_skipped"]) for item in target_results)

        status = "completed" if not errors else "completed_with_errors"
        result: Dict[str, object] = {
            "status": status,
            "plan_id": plan.plan_id,
            "files_removed": files_removed,
            "bytes_freed": bytes_freed,
            "files_skipped": files_skipped,
            "targets": target_results,
            "errors": errors,
        }

        if status == "completed":
            mark_completed(plan.plan_id, result)
        else:
            mark_failed(plan.plan_id)

        logger.info(
            "Cleanup plan %s executed: %d files removed, %d bytes freed, %d skipped.",
            plan.plan_id,
            files_removed,
            bytes_freed,
            files_skipped,
        )
        return result
    except Exception:  # noqa: BLE001
        mark_failed(plan.plan_id)
        logger.info(
            "Cleanup plan %s failed: 0 files removed, 0 bytes freed, 0 skipped.",
            plan.plan_id,
        )
        return {
            "status": "failed",
            "plan_id": plan.plan_id,
            "files_removed": 0,
            "bytes_freed": 0,
            "files_skipped": 0,
            "targets": [],
            "errors": ["Cleanup failed. Check the server logs."],
        }
