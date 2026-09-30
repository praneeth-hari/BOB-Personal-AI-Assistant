"""
app/tools/get_running_applications.py
─────────────────────────────────────
BOB tool: get_running_applications

Lists the applications currently running on this computer, or checks whether
a specific application (e.g. "chrome", "vs code") is running.

SAFETY & SECURITY MODEL
───────────────────────
• Completely read-only: processes are only inspected, never started,
  signalled, or terminated.
• Uses psutil directly in Python.
• NEVER executes shell commands, PowerShell, subprocesses, or network calls.
• Processes that disappear or deny access mid-scan are skipped gracefully.

OUTPUT SHAPING
──────────────
• Processes are deduplicated by executable name (Chrome's 30 helper
  processes become one entry with process_count=30).
• The general listing only includes processes owned by the current user
  (falling back to processes with a readable executable path when the user
  cannot be determined), hides well-known OS/background helpers, then caps
  the list at `limit` entries.
• A name check searches *all* readable processes, including background ones.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import psutil

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_LIMIT = 40
MAX_LIMIT = 100
MAX_PIDS_PER_APP = 5

# Friendly aliases → lowercase executable names they correspond to.
APPLICATION_ALIASES: Dict[str, List[str]] = {
    "chrome": ["chrome.exe"],
    "google chrome": ["chrome.exe"],
    "edge": ["msedge.exe"],
    "microsoft edge": ["msedge.exe"],
    "firefox": ["firefox.exe"],
    "brave": ["brave.exe"],
    "opera": ["opera.exe"],
    "vs code": ["code.exe"],
    "vscode": ["code.exe"],
    "visual studio code": ["code.exe"],
    "code": ["code.exe"],
    "visual studio": ["devenv.exe"],
    "word": ["winword.exe"],
    "microsoft word": ["winword.exe"],
    "excel": ["excel.exe"],
    "microsoft excel": ["excel.exe"],
    "powerpoint": ["powerpnt.exe"],
    "outlook": ["outlook.exe", "olk.exe"],
    "teams": ["ms-teams.exe", "teams.exe"],
    "microsoft teams": ["ms-teams.exe", "teams.exe"],
    "file explorer": ["explorer.exe"],
    "explorer": ["explorer.exe"],
    "notepad": ["notepad.exe"],
    "notepad++": ["notepad++.exe"],
    "calculator": ["calculatorapp.exe", "calc.exe", "calculator.exe"],
    "paint": ["mspaint.exe", "paint.exe"],
    "terminal": ["windowsterminal.exe"],
    "windows terminal": ["windowsterminal.exe"],
    "spotify": ["spotify.exe"],
    "discord": ["discord.exe"],
    "slack": ["slack.exe"],
    "zoom": ["zoom.exe"],
    "steam": ["steam.exe"],
    "vlc": ["vlc.exe"],
    "obs": ["obs64.exe", "obs32.exe"],
    "ollama": ["ollama.exe", "ollama app.exe"],
}

# Lowercase executable name → human-friendly display name.
DISPLAY_NAMES: Dict[str, str] = {
    "chrome.exe": "Google Chrome",
    "msedge.exe": "Microsoft Edge",
    "firefox.exe": "Mozilla Firefox",
    "brave.exe": "Brave",
    "opera.exe": "Opera",
    "code.exe": "Visual Studio Code",
    "devenv.exe": "Visual Studio",
    "winword.exe": "Microsoft Word",
    "excel.exe": "Microsoft Excel",
    "powerpnt.exe": "Microsoft PowerPoint",
    "outlook.exe": "Microsoft Outlook",
    "olk.exe": "Microsoft Outlook",
    "ms-teams.exe": "Microsoft Teams",
    "teams.exe": "Microsoft Teams",
    "explorer.exe": "File Explorer",
    "notepad.exe": "Notepad",
    "notepad++.exe": "Notepad++",
    "calculatorapp.exe": "Calculator",
    "calc.exe": "Calculator",
    "mspaint.exe": "Paint",
    "windowsterminal.exe": "Windows Terminal",
    "spotify.exe": "Spotify",
    "discord.exe": "Discord",
    "slack.exe": "Slack",
    "zoom.exe": "Zoom",
    "steam.exe": "Steam",
    "vlc.exe": "VLC Media Player",
    "obs64.exe": "OBS Studio",
    "ollama.exe": "Ollama",
    "ollama app.exe": "Ollama",
}

# OS / background helper processes hidden from the general listing.
BACKGROUND_PROCESSES = frozenset({
    "system", "system idle process", "idle", "registry", "memory compression",
    "secure system", "smss.exe", "csrss.exe", "wininit.exe", "winlogon.exe",
    "services.exe", "lsass.exe", "lsaiso.exe", "svchost.exe", "fontdrvhost.exe",
    "dwm.exe", "sihost.exe", "taskhostw.exe", "ctfmon.exe", "conhost.exe",
    "openconsole.exe", "dllhost.exe", "runtimebroker.exe", "backgroundtaskhost.exe",
    "searchhost.exe", "searchindexer.exe", "searchprotocolhost.exe",
    "searchfilterhost.exe", "startmenuexperiencehost.exe",
    "shellexperiencehost.exe", "shellhost.exe", "textinputhost.exe",
    "applicationframehost.exe", "systemsettingsbroker.exe", "smartscreen.exe",
    "securityhealthsystray.exe", "securityhealthservice.exe", "lockapp.exe",
    "useroobebroker.exe", "widgets.exe", "widgetservice.exe", "wmiprvse.exe",
    "audiodg.exe", "spoolsv.exe", "msmpeng.exe", "nissrv.exe",
    "crossdeviceresume.exe", "phoneexperiencehost.exe", "msedgewebview2.exe",
    "crashpad_handler.exe", "gamebar.exe", "gamebarftserver.exe",
    "unsecapp.exe", "wudfhost.exe", "sppsvc.exe", "mpdefendercoreservice.exe",
    "rundll32.exe", "identity_helper.exe", "crossdeviceservice.exe",
    "defendersessionhelper.exe", "nvcontainer.exe", "nvsphelper64.exe",
    "rtkauduservice64.exe", "sdxhelper.exe", "filecoauth.exe", "useroobebroker.exe",
})

_PROCESS_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase and drop '.exe' and every non-alphanumeric character."""
    lowered = text.lower().strip()
    if lowered.endswith(".exe"):
        lowered = lowered[:-4]
    return re.sub(r"[^a-z0-9+]", "", lowered)


def _coerce_limit(limit: Any) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(value, MAX_LIMIT))


def _snapshot_processes() -> tuple[List[Dict[str, Any]], int]:
    """
    Return (readable process records, number of processes skipped).

    Each record is {"pid": int, "name": str, "exe": Optional[str],
    "username": Optional[str]}.
    """
    records: List[Dict[str, Any]] = []
    skipped = 0
    for proc in psutil.process_iter(attrs=["pid", "name", "exe", "username"], ad_value=None):
        try:
            info = proc.info
            name = info.get("name")
            pid = info.get("pid")
        except _PROCESS_ERRORS:
            skipped += 1
            continue
        except Exception:  # noqa: BLE001 – never let one odd process break the scan
            logger.debug("Unexpected error reading process info.", exc_info=True)
            skipped += 1
            continue

        exe = info.get("exe") or None
        if not name and exe:
            name = os.path.basename(exe)
        if not name:
            skipped += 1
            continue
        records.append({"pid": pid, "name": str(name), "exe": exe, "username": info.get("username")})
    return records, skipped


def _group_by_name(
    records: List[Dict[str, Any]],
    current_user: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Deduplicate process records by lowercase process name."""
    groups: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        key = rec["name"].lower()
        group = groups.get(key)
        if group is None:
            group = {
                "name": rec["name"],
                "display_name": DISPLAY_NAMES.get(key, rec["name"]),
                "process_count": 0,
                "pids": [],
                "exe": None,
                "_owned": False,
            }
            groups[key] = group
        group["process_count"] += 1
        if rec["pid"] is not None and len(group["pids"]) < MAX_PIDS_PER_APP:
            group["pids"].append(rec["pid"])
        if group["exe"] is None and rec["exe"]:
            group["exe"] = rec["exe"]
        if current_user is not None and rec.get("username") == current_user:
            group["_owned"] = True
    return groups


def _current_username() -> Optional[str]:
    """Return the username owning this process, or None if unavailable."""
    try:
        return psutil.Process().username()
    except Exception:  # noqa: BLE001
        return None


def _public(group: Dict[str, Any]) -> Dict[str, Any]:
    """Strip internal bookkeeping keys from a process group."""
    return {k: v for k, v in group.items() if not k.startswith("_")}


def _matches_query(group_key: str, group: Dict[str, Any], query: str) -> bool:
    """True if a deduplicated process group matches the user's application name."""
    alias_key = query.lower().strip()
    alias_targets = APPLICATION_ALIASES.get(alias_key)
    if alias_targets is None and alias_key.endswith(".exe"):
        # LLMs often append ".exe" to friendly names ("vscode.exe").
        alias_targets = APPLICATION_ALIASES.get(alias_key[:-4])
    if alias_targets is not None:
        return group_key in alias_targets

    needle = _normalise(query)
    if not needle:
        return False
    candidates = [group_key, group["display_name"]]
    if group["exe"]:
        candidates.append(os.path.basename(group["exe"]))
    return any(needle in _normalise(c) for c in candidates)


# ─── Implementation ───────────────────────────────────────────────────────────

def _get_running_applications(
    application: Optional[str] = None,
    limit: Any = DEFAULT_LIMIT,
    **kwargs: Any,
) -> ToolResult:
    """
    List running applications, or check whether *application* is running.

    :param application: Optional application or process name to check for.
    :param limit:       Maximum number of applications in the general listing.
    """
    max_items = _coerce_limit(limit)

    try:
        records, skipped = _snapshot_processes()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to enumerate running processes.")
        return ToolResult(success=False, error=f"Failed to list running processes: {exc}")

    current_user = _current_username()
    groups = _group_by_name(records, current_user)
    query = application.strip() if isinstance(application, str) else ""

    # ── Specific application check ────────────────────────────────────────
    if query:
        matches = [g for key, g in groups.items() if _matches_query(key, g, query)]
        matches.sort(key=lambda g: (-g["process_count"], g["name"].lower()))
        process_count = sum(g["process_count"] for g in matches)
        if matches:
            names = ", ".join(g["display_name"] for g in matches[:5])
            message = f"Yes, {query} is running ({names}; {process_count} process(es))."
        else:
            message = f"No, {query} does not appear to be running."
        return ToolResult(
            success=True,
            data={
                "query": query,
                "running": bool(matches),
                "match_count": process_count,
                "matches": [_public(g) for g in matches[:max_items]],
                "message": message,
            },
        )

    # ── General listing ───────────────────────────────────────────────────
    def _is_user_app(key: str, g: Dict[str, Any]) -> bool:
        if key in BACKGROUND_PROCESSES:
            return False
        if current_user is not None:
            return g["_owned"]
        return g["exe"] is not None

    apps = [g for key, g in groups.items() if _is_user_app(key, g)]
    apps.sort(key=lambda g: g["display_name"].lower())
    shown = apps[:max_items]

    return ToolResult(
        success=True,
        data={
            "total_processes": len(records) + skipped,
            "application_count": len(apps),
            # Executable paths are omitted here to keep the listing compact;
            # a name check (application=...) returns them.
            "applications": [
                {k: v for k, v in _public(g).items() if k != "exe"} for g in shown
            ],
            "truncated": len(apps) > len(shown),
            "hidden_background_processes": len(groups) - len(apps),
            "unreadable_processes": skipped,
            "message": (
                f"{len(apps)} applications running"
                + (f" (showing first {len(shown)})." if len(apps) > len(shown) else ".")
            ),
        },
    )


# ─── Tool Descriptor ──────────────────────────────────────────────────────────

RUNNING_APPLICATIONS_TOOL = Tool(
    name="get_running_applications",
    description=(
        "Lists the applications currently running on this computer, or checks whether "
        "a specific application is running. "
        "Use this tool when the user asks: "
        "- 'What apps are running?' "
        "- 'Which applications are open?' "
        "- 'Is Chrome running?' "
        "- 'Is VS Code open?' "
        "Pass 'application' to check for one app; omit it to list running apps. "
        "Safe and read-only: it never starts or closes anything."
    ),
    parameters={
        "type": "object",
        "properties": {
            "application": {
                "type": "string",
                "description": (
                    "Optional application or process name to check, e.g. 'chrome', "
                    "'vs code', 'spotify', 'notepad.exe'. Omit to list all running applications."
                ),
            },
            "limit": {
                "type": "integer",
                "description": f"Optional maximum number of applications to return (1-{MAX_LIMIT}, default {DEFAULT_LIMIT}).",
            },
        },
        "required": [],
    },
    fn=_get_running_applications,
)
