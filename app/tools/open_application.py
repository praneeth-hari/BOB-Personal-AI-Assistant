"""
app/tools/open_application.py
──────────────────────────────
BOB tool: open_application

Opens an approved Windows application by alias.

SECURITY MODEL
──────────────
• An explicit allowlist maps alias → [executable_path_or_name].
• The LLM sends an alias string (e.g. "notepad").
• The alias is matched against the allowlist (case-insensitive).
• The ACTUAL executable used comes from the allowlist value — never from
  the LLM's raw input.
• shell=False is used in all subprocess calls.
• Arbitrary paths, scripts, PowerShell, cmd.exe, etc. cannot be launched.
• Adding a new application requires a code change here — not a runtime request.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Dict, List

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)


# ─── Allowlist ────────────────────────────────────────────────────────────────
# key   : lowercase alias the LLM may request
# value : command list passed to subprocess.Popen (first element is the exe)
#
# IMPORTANT: Do not add any shell interpreters (cmd, powershell, bash, python)
# or arbitrary path expansion here.  All entries must be specific, named apps.

ALLOWED_APPLICATIONS: Dict[str, List[str]] = {
    "notepad":    ["notepad.exe"],
    "calculator": ["calc.exe"],
    "paint":      ["mspaint.exe"],
    "explorer":   ["explorer.exe"],
    "wordpad":    ["wordpad.exe"],
    "vscode":     ["code"],
    "code":       ["code"],
    "snippingtool": ["SnippingTool.exe"],
}

# Public list of supported aliases (used in tool description & tests)
SUPPORTED_APPS: List[str] = sorted(ALLOWED_APPLICATIONS.keys())


# ─── Implementation ───────────────────────────────────────────────────────────

def _open_application(application: str) -> ToolResult:
    """
    Launch an allowlisted application.

    :param application: Alias name (case-insensitive).  Must be a key in
                        ALLOWED_APPLICATIONS.
    :returns: ToolResult with data={"application", "command", "status"} on
              success, or error message on failure.
    """
    if not application or not application.strip():
        return ToolResult(success=False, error="application name must not be empty.")

    key = application.strip().lower()

    if key not in ALLOWED_APPLICATIONS:
        return ToolResult(
            success=False,
            error=(
                f"Application {application!r} is not in the allowlist. "
                f"Supported applications: {SUPPORTED_APPS}. "
                "Arbitrary executables and shell commands cannot be run."
            ),
        )

    command: List[str] = ALLOWED_APPLICATIONS[key]
    logger.info("Launching allowlisted application: %r → %r", key, command)

    try:
        # shell=False: command list is used directly; no shell interpolation.
        subprocess.Popen(command, shell=False)
        return ToolResult(
            success=True,
            data={
                "application": key,
                "command":     command[0],
                "status":      "launched",
            },
        )
    except FileNotFoundError:
        return ToolResult(
            success=False,
            error=(
                f"Executable not found for {application!r} ({command[0]!r}). "
                "The application may not be installed on this machine."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to launch %r", key)
        return ToolResult(success=False, error=str(exc))


# ─── Tool descriptor ──────────────────────────────────────────────────────────

OPEN_APPLICATION_TOOL = Tool(
    name="open_application",
    description=(
        "Opens an approved Windows application by name. "
        f"Supported applications: {SUPPORTED_APPS}. "
        "Cannot run arbitrary executables, scripts, or shell commands."
    ),
    parameters={
        "type": "object",
        "properties": {
            "application": {
                "type": "string",
                "description": (
                    "The name of the application to open. "
                    f"Must be one of: {SUPPORTED_APPS}. "
                    "Example: 'notepad', 'calculator', 'vscode'."
                ),
            }
        },
        "required": ["application"],
    },
    fn=_open_application,
)
