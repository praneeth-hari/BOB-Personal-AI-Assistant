"""
app/tools/close_application.py
──────────────────────────────
BOB tool: close_application

Safely terminates open instances of an approved Windows application by alias.

SECURITY MODEL
──────────────
• An explicit allowlist maps application aliases to a fixed set of target
  process executable names (e.g. "notepad" → {"notepad.exe"}).
• The LLM sends an alias string (e.g. "notepad").
• The alias is strictly validated against ALLOWED_CLOSE_APPLICATIONS.
• The LLM cannot provide arbitrary executable names, PIDs, scripts, or wildcards.
• No shell commands (cmd, powershell, bash, taskkill via shell) are executed.
• Process termination uses psutil directly (Win32 API bindings) with shell=False.
• Explorer and critical OS processes are intentionally excluded to prevent
  system instability.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

import psutil

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)


# ─── Allowlist ────────────────────────────────────────────────────────────────
# key   : lowercase alias the LLM or user may request
# value : set of lowercase executable names corresponding to the application
#
# IMPORTANT: Never add explorer.exe, system services, shells (cmd, powershell,
# bash, python), or arbitrary process names here.

ALLOWED_CLOSE_APPLICATIONS: Dict[str, Set[str]] = {
    "notepad": {"notepad.exe"},
    "calculator": {"calc.exe", "calculatorapp.exe", "calculator.exe"},
    "calc": {"calc.exe", "calculatorapp.exe", "calculator.exe"},
    "paint": {"mspaint.exe", "paint.exe"},
    "wordpad": {"wordpad.exe"},
    "vscode": {"code.exe"},
    "code": {"code.exe"},
    "snippingtool": {"snippingtool.exe", "screenclippinghost.exe"},
}

# Public list of supported aliases (used in tool description & tests)
SUPPORTED_CLOSE_APPS: List[str] = sorted(ALLOWED_CLOSE_APPLICATIONS.keys())


# ─── Implementation ───────────────────────────────────────────────────────────

def _close_application(application: str) -> ToolResult:
    """
    Safely terminate running processes for an allowlisted application alias.

    :param application: Alias name (case-insensitive). Must be in ALLOWED_CLOSE_APPLICATIONS.
    :returns: ToolResult indicating success, count of terminated processes, or error.
    """
    if not application or not str(application).strip():
        return ToolResult(success=False, error="application name must not be empty.")

    key = str(application).strip().lower()

    if key not in ALLOWED_CLOSE_APPLICATIONS:
        return ToolResult(
            success=False,
            error=(
                f"Application {application!r} is not in the allowlist. "
                f"Supported applications to close: {SUPPORTED_CLOSE_APPS}. "
                "Arbitrary process termination is forbidden."
            ),
        )

    target_exes = ALLOWED_CLOSE_APPLICATIONS[key]
    logger.info("Attempting to close application %r (targets: %s)", key, target_exes)

    matching_procs: List[psutil.Process] = []
    try:
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                proc_name = proc.info.get("name")
                if proc_name and str(proc_name).strip().lower() in target_exes:
                    matching_procs.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to scan processes for %r", key)
        return ToolResult(success=False, error=f"Process scan failed: {exc}")

    if not matching_procs:
        logger.info("Application %r is not currently running.", key)
        return ToolResult(
            success=False,
            data={
                "application": key,
                "status": "not_running",
                "terminated_pids": [],
                "count": 0,
            },
            error=f"Application {application!r} is not currently running.",
        )

    closed_pids: List[int] = []
    errors: List[str] = []

    for proc in matching_procs:
        pid = proc.pid
        try:
            proc.terminate()
            closed_pids.append(pid)
        except psutil.NoSuchProcess:
            # Process exited before terminate() was called
            closed_pids.append(pid)
        except psutil.AccessDenied:
            logger.warning("Access denied terminating PID %d (%r)", pid, key)
            errors.append(f"Access denied for PID {pid}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error terminating PID %d (%r)", pid, key)
            errors.append(f"PID {pid}: {exc}")

    # Wait briefly for processes to exit gracefully; kill remaining if needed
    if closed_pids:
        try:
            alive_candidates = [p for p in matching_procs if p.pid in closed_pids]
            gone, alive = psutil.wait_procs(alive_candidates, timeout=2.0)
            for proc in alive:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:  # noqa: BLE001
            pass

    if not closed_pids and errors:
        return ToolResult(
            success=False,
            error=f"Failed to close {application!r}: {'; '.join(errors)}",
        )

    logger.info("Closed %d instance(s) of %r (PIDs: %s)", len(closed_pids), key, closed_pids)
    return ToolResult(
        success=True,
        data={
            "application": key,
            "terminated_pids": closed_pids,
            "count": len(closed_pids),
            "status": "closed",
        },
    )


# ─── Tool Descriptor ──────────────────────────────────────────────────────────

CLOSE_APPLICATION_TOOL = Tool(
    name="close_application",
    description=(
        "Closes or terminates open instances of an approved Windows application by name. "
        f"Supported applications: {SUPPORTED_CLOSE_APPS}. "
        "Cannot terminate arbitrary processes or system services."
    ),
    parameters={
        "type": "object",
        "properties": {
            "application": {
                "type": "string",
                "description": (
                    "The name of the application to close. "
                    f"Must be one of: {SUPPORTED_CLOSE_APPS}. "
                    "Example: 'notepad', 'calculator', 'vscode', 'paint'."
                ),
            }
        },
        "required": ["application"],
    },
    fn=_close_application,
)
