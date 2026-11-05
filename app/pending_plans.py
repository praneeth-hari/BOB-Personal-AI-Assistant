"""
app/pending_plans.py
────────────────────
In-memory store for approval-gated cleanup plans.

This module contains no deletion code and no LLM tool descriptor. The web
interface reads pending actions from here after a chat turn and consumes a
plan exactly once during approval.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PLAN_TTL_SECONDS = 10 * 60
STATUS_PENDING = "pending"
STATUS_EXECUTING = "executing"
STATUS_COMPLETED = "completed"
STATUS_REJECTED = "rejected"
STATUS_CANCELLED = "cancelled"
STATUS_EXPIRED = "expired"
STATUS_FAILED = "failed"


class PlanError(RuntimeError):
    """Base class for safe, user-facing plan errors."""


class PlanNotFoundError(PlanError):
    pass


class PlanExpiredError(PlanError):
    pass


class PlanAlreadyUsedError(PlanError):
    pass


class NoTargetsSelectedError(PlanError):
    pass


@dataclass
class TargetPlan:
    key: str
    label: str
    kind: str
    file_count: int
    bytes: int
    root: Optional[str] = None
    paths: Tuple[str, ...] = ()
    warning: Optional[str] = None
    default_include: bool = True


@dataclass
class CleanupPlan:
    plan_id: str
    created_at: float
    expires_at: float
    targets: Dict[str, TargetPlan]
    status: str = STATUS_PENDING
    result: Optional[Dict[str, object]] = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def total_files(self) -> int:
        return sum(target.file_count for target in self.targets.values())

    @property
    def total_bytes(self) -> int:
        return sum(target.bytes for target in self.targets.values())


_plans: Dict[str, CleanupPlan] = {}
_store_lock = threading.RLock()


def _expire_locked(plan: CleanupPlan, now: Optional[float] = None) -> None:
    current = time.time() if now is None else now
    if plan.status == STATUS_PENDING and current >= plan.expires_at:
        plan.status = STATUS_EXPIRED
        logger.info(
            "Cleanup plan %s expired: %d files, %d bytes.",
            plan.plan_id,
            plan.total_files,
            plan.total_bytes,
        )


def _cancel_pending_locked() -> None:
    for plan in _plans.values():
        if plan.status == STATUS_PENDING:
            plan.status = STATUS_CANCELLED
            logger.info(
                "Cleanup plan %s cancelled: %d files, %d bytes.",
                plan.plan_id,
                plan.total_files,
                plan.total_bytes,
            )


def cancel_pending_plans() -> None:
    """Cancel any pending plan before a new scan starts."""
    with _store_lock:
        _cancel_pending_locked()


def create_plan(targets: Dict[str, TargetPlan]) -> CleanupPlan:
    """Create a new plan and cancel any older pending plan."""
    with _store_lock:
        _cancel_pending_locked()
        now = time.time()
        plan = CleanupPlan(
            plan_id=secrets.token_urlsafe(32),
            created_at=now,
            expires_at=now + PLAN_TTL_SECONDS,
            targets=targets,
        )
        _plans[plan.plan_id] = plan
        logger.info(
            "Cleanup plan %s created: %d files, %d bytes.",
            plan.plan_id,
            plan.total_files,
            plan.total_bytes,
        )
        return plan


def get_plan(plan_id: str) -> Optional[CleanupPlan]:
    with _store_lock:
        plan = _plans.get(plan_id)
        if plan is not None:
            _expire_locked(plan)
        return plan


def get_pending_plan() -> Optional[CleanupPlan]:
    with _store_lock:
        now = time.time()
        for plan in _plans.values():
            _expire_locked(plan, now)
        for plan in _plans.values():
            if plan.status == STATUS_PENDING:
                return plan
        return None


def action_payload(plan: CleanupPlan) -> Dict[str, object]:
    """Build the UI payload. It never includes filesystem paths."""
    targets: List[Dict[str, object]] = []
    for key in ("temp", "recycle_bin"):
        target = plan.targets.get(key)
        if target is None:
            continue
        targets.append(
            {
                "key": target.key,
                "label": target.label,
                "file_count": target.file_count,
                "bytes": target.bytes,
                "default_include": target.default_include,
                "warning": target.warning,
                "empty": target.file_count == 0,
            }
        )

    total_files = plan.total_files
    total_bytes = plan.total_bytes
    if total_files == 0:
        summary = "No cleanup items were found."
    else:
        parts = []
        temp = plan.targets.get("temp")
        recycle = plan.targets.get("recycle_bin")
        if temp is not None and temp.file_count:
            parts.append(f"{temp.file_count} temp file(s)")
        if recycle is not None and recycle.file_count:
            parts.append(f"{recycle.file_count} Recycle Bin item(s)")
        summary = "Ready to clean up " + " and ".join(parts) + "."

    return {
        "plan_id": plan.plan_id,
        "status": plan.status,
        "summary": summary,
        "targets": targets,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "expires_at": int(plan.expires_at),
    }


def get_pending_action() -> Optional[Dict[str, object]]:
    plan = get_pending_plan()
    return action_payload(plan) if plan is not None else None


def get_action_for_plan(plan_id: str) -> Optional[Dict[str, object]]:
    plan = get_plan(plan_id)
    if plan is None or plan.status != STATUS_PENDING:
        return None
    return action_payload(plan)


def reject_plan(plan_id: str) -> Dict[str, object]:
    with _store_lock:
        plan = _plans.get(plan_id)
        if plan is None:
            raise PlanNotFoundError("Cleanup plan not found.")
        _expire_locked(plan)
        if plan.status == STATUS_EXPIRED:
            raise PlanExpiredError("Cleanup plan expired.")
        if plan.status != STATUS_PENDING:
            raise PlanAlreadyUsedError("Cleanup plan was already used or cancelled.")
        plan.status = STATUS_REJECTED
        logger.info(
            "Cleanup plan %s rejected: %d files, %d bytes.",
            plan.plan_id,
            plan.total_files,
            plan.total_bytes,
        )
        return {
            "status": STATUS_REJECTED,
            "plan_id": plan.plan_id,
            "total_files": plan.total_files,
            "total_bytes": plan.total_bytes,
        }


def begin_execution(
    plan_id: str,
    *,
    include_temp: bool,
    include_recycle_bin: bool,
) -> Tuple[CleanupPlan, Tuple[str, ...]]:
    """Validate and atomically consume a pending plan for execution."""
    with _store_lock:
        plan = _plans.get(plan_id)
        if plan is None:
            raise PlanNotFoundError("Cleanup plan not found.")
        _expire_locked(plan)
        if plan.status == STATUS_EXPIRED:
            raise PlanExpiredError("Cleanup plan expired.")
        if plan.status != STATUS_PENDING:
            raise PlanAlreadyUsedError("Cleanup plan was already used or cancelled.")

        selected: List[str] = []
        if include_temp:
            target = plan.targets.get("temp")
            if target is not None and target.file_count > 0:
                selected.append("temp")
        if include_recycle_bin:
            target = plan.targets.get("recycle_bin")
            if target is not None and target.file_count > 0:
                selected.append("recycle_bin")

        if not selected:
            raise NoTargetsSelectedError("No cleanup targets were selected.")

        selected_files = sum(plan.targets[key].file_count for key in selected)
        selected_bytes = sum(plan.targets[key].bytes for key in selected)
        plan.status = STATUS_EXECUTING
        logger.info(
            "Cleanup plan %s approved: %d files, %d bytes.",
            plan.plan_id,
            selected_files,
            selected_bytes,
        )
        return plan, tuple(selected)


def mark_completed(plan_id: str, result: Dict[str, object]) -> None:
    with _store_lock:
        plan = _plans.get(plan_id)
        if plan is not None:
            plan.status = STATUS_COMPLETED
            plan.result = result


def mark_failed(plan_id: str) -> None:
    with _store_lock:
        plan = _plans.get(plan_id)
        if plan is not None:
            plan.status = STATUS_FAILED


def reset_plans_for_tests() -> None:
    """Clear the in-memory store. Intended for tests only."""
    with _store_lock:
        _plans.clear()
