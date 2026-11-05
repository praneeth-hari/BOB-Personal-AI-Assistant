"""
app/tools/propose_cleanup.py
────────────────────────────
Read-only cleanup planner. Deletes nothing.

Only the user's temp folder and the Recycle Bin are scanned. The result is an
in-memory plan that the web interface can offer for explicit approval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from app.pending_plans import (
    PLAN_TTL_SECONDS,
    TargetPlan,
    action_payload,
    cancel_pending_plans,
    create_plan,
)
from app.tools.tool import Tool, ToolResult
from app.tools.windows_facts import (
    TEMP_FILE_MIN_AGE_SECONDS,
    human_bytes,
    query_recycle_bin,
    resolve_temp_root,
    scan_files,
)

logger = logging.getLogger(__name__)


CLEANUP_TARGETS: Dict[str, Dict[str, Any]] = {
    "temp": {
        "label": "Temporary files older than 24 hours",
        "kind": "files",
        "default_include": True,
        "warning": None,
    },
    "recycle_bin": {
        "label": "Recycle Bin (all drives)",
        "kind": "recycle_bin",
        "default_include": False,
        "warning": "Irreversible. Empties the Recycle Bin for all drives.",
    },
}


def _propose_cleanup() -> ToolResult:
    """Scan allowlisted targets and store a pending cleanup plan."""
    try:
        cancel_pending_plans()
        temp_root = resolve_temp_root()
        temp_spec = CLEANUP_TARGETS["temp"]
        scan = scan_files(
            Path(temp_root),
            min_age_seconds=TEMP_FILE_MIN_AGE_SECONDS,
        )
        temp_target = TargetPlan(
            key="temp",
            label=str(temp_spec["label"]),
            kind=str(temp_spec["kind"]),
            file_count=len(scan.files),
            bytes=sum(entry.size for entry in scan.files),
            root=str(temp_root),
            paths=tuple(str(entry.path) for entry in scan.files),
            default_include=bool(temp_spec["default_include"]),
            warning=temp_spec["warning"],
        )

        recycle_spec = CLEANUP_TARGETS["recycle_bin"]
        recycle_info = query_recycle_bin()
        recycle_target = TargetPlan(
            key="recycle_bin",
            label=str(recycle_spec["label"]),
            kind=str(recycle_spec["kind"]),
            file_count=recycle_info.count,
            bytes=recycle_info.bytes,
            default_include=bool(recycle_spec["default_include"]),
            warning=recycle_spec["warning"],
        )

        targets = {"temp": temp_target, "recycle_bin": recycle_target}
        total_files = temp_target.file_count + recycle_target.file_count
        if total_files == 0:
            return ToolResult(
                success=True,
                data={
                    "plan_id": None,
                    "pending": False,
                    "summary": "No cleanup items were found.",
                },
            )

        plan = create_plan(targets)
        payload = action_payload(plan)
        summary = str(payload["summary"])
        if scan.truncated:
            summary += " The scan was limited to the first 20,000 files."
        if scan.skipped:
            summary += f" {scan.skipped} item(s) were skipped as unreadable or unsafe."

        return ToolResult(
            success=True,
            data={
                "plan_id": plan.plan_id,
                "pending": True,
                "summary": summary,
                "total_files": plan.total_files,
                "total_bytes": plan.total_bytes,
                "total_size": human_bytes(plan.total_bytes),
                "expires_in_seconds": PLAN_TTL_SECONDS,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cleanup proposal failed (%s).", type(exc).__name__)
        return ToolResult(success=False, error="Could not prepare cleanup.")


PROPOSE_CLEANUP_TOOL = Tool(
    name="propose_cleanup",
    description=(
        "Scans only the user's temp folder (files older than 24 hours) and the "
        "Recycle Bin, then proposes a cleanup plan for the user to approve in the "
        "web interface. It deletes nothing. The result is a plan ID and a short "
        "summary. The user must click Approve; there is no delete tool."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    fn=_propose_cleanup,
)
