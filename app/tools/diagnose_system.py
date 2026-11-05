"""
app/tools/diagnose_system.py
────────────────────────────
Read-only system diagnosis for slow or hanging PCs.

The output is intentionally short and pre-ranked so a small local model can
relay the important findings without re-sorting them.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil

from app.tools.get_running_applications import DISPLAY_NAMES
from app.tools.tool import Tool, ToolResult
from app.tools.windows_facts import (
    count_startup_programs,
    fixed_drives,
    get_power_status,
    human_bytes,
    query_recycle_bin,
    resolve_temp_root,
    scan_files,
)

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

CPU_WARNING_PERCENT = 85.0
CPU_CRITICAL_PERCENT = 95.0
RAM_WARNING_PERCENT = 85.0
RAM_CRITICAL_PERCENT = 95.0
SWAP_WARNING_PERCENT = 80.0
SWAP_CRITICAL_PERCENT = 95.0
DISK_CRITICAL_FREE_PERCENT = 10.0
DISK_WARNING_FREE_PERCENT = 20.0
TEMP_WARNING_BYTES = 2 * 1024 ** 3
TEMP_CRITICAL_BYTES = 5 * 1024 ** 3
RECYCLE_BIN_WARNING_BYTES = 2 * 1024 ** 3
RECYCLE_BIN_CRITICAL_BYTES = 5 * 1024 ** 3
STARTUP_INFO_COUNT = 15
STARTUP_WARNING_COUNT = 25
UPTIME_INFO_DAYS = 7.0
BATTERY_LOW_PERCENT = 20
PROCESS_CPU_WARNING_PERCENT = 50.0
PROCESS_RAM_WARNING_BYTES = 2 * 1024 ** 3

MAX_RESULT_CHARS = 1500
MAX_TEMP_SCAN_ENTRIES = 20_000
TOP_PROCESS_LIMIT = 5

_IGNORED_PROCESSES = {
    "system idle process",
    "idle",
    "system",
    "registry",
    "memory compression",
}

_PYTHON_PARENT_NAMES = frozenset({"python", "pythonw", "py"})
_ASSISTANT_LABEL = "BOB / Ollama (this assistant)"


def _finding(severity: str, message: str, fix: str) -> Dict[str, str]:
    return {"severity": severity, "message": message, "fix": fix}


def _percent(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _logical_cpu_count() -> int:
    """Return a safe logical CPU count for normalising per-process CPU."""
    try:
        count = psutil.cpu_count(logical=True)
        return max(1, int(count or 1))
    except Exception:  # noqa: BLE001
        return 1


def _process_basename(name: Any) -> str:
    """Lowercase a process name and drop a trailing .exe."""
    text = str(name or "").strip().lower()
    if text.endswith(".exe"):
        text = text[:-4]
    return text


def _short_executable(path: Any) -> str:
    """Return the final path components only, hiding user-profile prefixes."""
    text = str(path or "").strip().replace("/", "\\")
    parts = [part for part in text.split("\\") if part]
    if not parts:
        return ""
    short = "\\".join(parts[-3:])
    if len(short) > 60:
        short = "..." + short[-57:]
    return short


def _protected_pids(root_pid: Optional[int] = None) -> Set[int]:
    """
    Protect BOB's process, its direct children, and at most one Python parent.

    The parent walk deliberately never goes beyond the first parent and only
    accepts a Python/venv launcher. This avoids marking terminals, shells,
    editors, or explorer.exe as part of BOB.
    """
    protected: Set[int] = set()
    try:
        pid = int(os.getpid() if root_pid is None else root_pid)
        current = psutil.Process(pid)
        protected.add(int(current.pid))
    except Exception:  # noqa: BLE001
        return protected

    try:
        for child in current.children(recursive=False):
            protected.add(int(child.pid))
    except Exception:  # noqa: BLE001
        pass

    try:
        parent = current.parent()
    except Exception:  # noqa: BLE001
        parent = None

    if parent is not None:
        try:
            if _process_basename(parent.name()) in _PYTHON_PARENT_NAMES:
                protected.add(int(parent.pid))
        except Exception:  # noqa: BLE001
            pass

    return protected


def _is_assistant_process(name: str, pid: int, protected_pids: Set[int]) -> bool:
    """True for BOB's own process tree or any Ollama process."""
    return int(pid) in protected_pids or str(name or "").strip().lower().startswith("ollama")


def _top_processes(
    logical_cores: Optional[int] = None,
    protected_pids: Optional[Set[int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample processes and return top CPU/RAM groups by name and protection."""
    cores = _logical_cpu_count() if logical_cores is None else max(1, int(logical_cores))
    protected = _protected_pids() if protected_pids is None else set(protected_pids)

    try:
        processes = list(psutil.process_iter(["pid", "name", "memory_info", "exe"]))
    except Exception:  # noqa: BLE001
        return [], []

    for proc in processes:
        try:
            proc.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:  # noqa: BLE001
            continue

    time.sleep(0.2)

    groups: Dict[Tuple[str, bool], Dict[str, Any]] = {}
    for proc in processes:
        try:
            info = proc.info
            pid = int(info.get("pid") or getattr(proc, "pid", 0) or 0)
            name = str(info.get("name") or "").strip()
            if not name or name.lower() in _IGNORED_PROCESSES:
                continue
            raw_cpu = _percent(proc.cpu_percent(None))
            cpu = max(0.0, min(100.0, raw_cpu / cores))
            mem_info = info.get("memory_info")
            rss = int(getattr(mem_info, "rss", 0) or 0)
            ram_mb = rss / (1024 ** 2)
            exe = _short_executable(info.get("exe")) or "unknown"
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:  # noqa: BLE001
            continue

        is_protected = _is_assistant_process(name, pid, protected)
        key = (name.lower(), is_protected)
        group = groups.get(key)
        if group is None:
            group = {
                "name": _ASSISTANT_LABEL
                if is_protected
                else DISPLAY_NAMES.get(name.lower(), name),
                "process_count": 0,
                "cpu_percent": 0.0,
                "ram_mb": 0.0,
                "protected": is_protected,
                "_max_cpu": -1.0,
                "_cpu_pid": pid,
                "_cpu_exe": exe,
                "_max_ram": -1.0,
                "_ram_pid": pid,
                "_ram_exe": exe,
            }
            groups[key] = group

        group["process_count"] += 1
        group["cpu_percent"] += cpu
        group["ram_mb"] += ram_mb
        if cpu > group["_max_cpu"]:
            group["_max_cpu"] = cpu
            group["_cpu_pid"] = pid
            group["_cpu_exe"] = exe
        if ram_mb > group["_max_ram"]:
            group["_max_ram"] = ram_mb
            group["_ram_pid"] = pid
            group["_ram_exe"] = exe

    values = list(groups.values())
    top_cpu = sorted(values, key=lambda item: item["cpu_percent"], reverse=True)[
        :TOP_PROCESS_LIMIT
    ]
    top_ram = sorted(values, key=lambda item: item["ram_mb"], reverse=True)[
        :TOP_PROCESS_LIMIT
    ]

    def _clean(items: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
        return [
            {
                "name": str(item["name"])[:40],
                "process_count": int(item["process_count"]),
                "pid": int(item[f"_{metric}_pid"]),
                "exe": str(item[f"_{metric}_exe"])[:60],
                "cpu_percent": round(float(item["cpu_percent"]), 1),
                "ram_mb": round(float(item["ram_mb"]), 1),
                "protected": bool(item["protected"]),
            }
            for item in items
        ]

    return _clean(top_cpu, "cpu"), _clean(top_ram, "ram")


def _collect_facts() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    logical_cores = _logical_cpu_count()
    top_cpu, top_ram = _top_processes(logical_cores)

    temp_scan = scan_files(resolve_temp_root(), max_entries=MAX_TEMP_SCAN_ENTRIES)
    temp_bytes = sum(entry.size for entry in temp_scan.files)

    recycle = query_recycle_bin()
    startup_count = count_startup_programs()
    power = get_power_status()

    battery_percent = power.get("battery_percent")
    if battery_percent is None:
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                battery_percent = int(battery.percent)
        except Exception:  # noqa: BLE001
            battery_percent = None

    return {
        "cpu_percent": _percent(psutil.cpu_percent(interval=0.5)),
        "logical_cores": logical_cores,
        "ram_percent": _percent(vm.percent),
        "ram_used_gb": round(vm.used / (1024 ** 3), 1),
        "ram_total_gb": round(vm.total / (1024 ** 3), 1),
        "swap_percent": _percent(swap.percent),
        "swap_used_gb": round(swap.used / (1024 ** 3), 1),
        "top_cpu": top_cpu,
        "top_ram": top_ram,
        "drives": fixed_drives(),
        "temp_bytes": temp_bytes,
        "temp_files": len(temp_scan.files),
        "temp_truncated": temp_scan.truncated,
        "recycle_count": recycle.count,
        "recycle_bytes": recycle.bytes,
        "startup_count": startup_count,
        "uptime_days": round((time.time() - psutil.boot_time()) / 86400, 1),
        "battery_percent": battery_percent,
        "plugged_in": power.get("plugged_in"),
        "battery_saver": power.get("battery_saver"),
    }


def _process_label(item: Dict[str, Any]) -> str:
    """Build a self-contained process label with PID, path, and group size."""
    name = str(item.get("name") or "Unknown process")
    pid = item.get("pid")
    pid_text = str(pid) if pid is not None else "unknown"
    exe = str(item.get("exe") or "unknown path")
    count = int(item.get("process_count") or 1)
    count_text = f" ({count} processes)" if count > 1 else ""
    return f"{name}{count_text} [PID {pid_text}, {exe}]"


def _process_fix(item: Dict[str, Any]) -> str:
    if item.get("protected"):
        return "This is BOB/Ollama itself; do not close it."
    return "Close or restart it if you do not need it."


def _build_findings(facts: Dict[str, Any]) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    top_cpu = facts.get("top_cpu") or []
    top_ram = facts.get("top_ram") or []
    top_cpu_protected = bool(top_cpu and top_cpu[0].get("protected"))
    top_ram_protected = bool(top_ram and top_ram[0].get("protected"))

    ram = _percent(facts["ram_percent"])
    if ram >= RAM_CRITICAL_PERCENT:
        ram_fix = "Close unused apps or browser tabs, then retry."
        if top_ram_protected:
            ram_fix = "Close apps you are not using, but do not close BOB/Ollama."
        findings.append(
            _finding(
                "critical",
                f"RAM usage is {ram:.0f}% (almost full).",
                ram_fix,
            )
        )
    elif ram >= RAM_WARNING_PERCENT:
        ram_fix = "Close apps you are not using."
        if top_ram_protected:
            ram_fix = "Close apps you are not using, but do not close BOB/Ollama."
        findings.append(
            _finding(
                "warning",
                f"RAM usage is {ram:.0f}% (high).",
                ram_fix,
            )
        )

    cpu = _percent(facts["cpu_percent"])
    if cpu >= CPU_CRITICAL_PERCENT:
        cpu_fix = "Find the busiest app below and close or restart it."
        if top_cpu_protected:
            cpu_fix = (
                "BOB / Ollama is the busiest process. Do not close it; "
                "wait for the current task to finish."
            )
        findings.append(
            _finding(
                "critical",
                f"CPU usage is {cpu:.0f}% (saturated).",
                cpu_fix,
            )
        )
    elif cpu >= CPU_WARNING_PERCENT:
        cpu_fix = "Check the top CPU apps below."
        if top_cpu_protected:
            cpu_fix = "BOB / Ollama is the busiest process. Do not close it."
        findings.append(
            _finding(
                "warning",
                f"CPU usage is {cpu:.0f}% (high).",
                cpu_fix,
            )
        )

    swap = _percent(facts["swap_percent"])
    if swap >= SWAP_CRITICAL_PERCENT:
        findings.append(
            _finding(
                "critical",
                f"Swap usage is {swap:.0f}% (almost full).",
                "Close memory-heavy apps and restart the PC.",
            )
        )
    elif swap >= SWAP_WARNING_PERCENT:
        findings.append(
            _finding(
                "warning",
                f"Swap usage is {swap:.0f}% (high).",
                "Close unused apps to reduce memory pressure.",
            )
        )

    for drive in facts.get("drives", []):
        percent_used = _percent(drive.get("percent"))
        free_percent = max(0.0, 100.0 - percent_used)
        free_text = human_bytes(int(drive.get("free", 0)))
        mount = str(drive.get("mountpoint") or "drive")
        if free_percent < DISK_CRITICAL_FREE_PERCENT:
            findings.append(
                _finding(
                    "critical",
                    f"Drive {mount} free space is {free_percent:.0f}% ({free_text} free).",
                    "Move or delete large files on that drive.",
                )
            )
        elif free_percent < DISK_WARNING_FREE_PERCENT:
            findings.append(
                _finding(
                    "warning",
                    f"Drive {mount} free space is low at {free_percent:.0f}% ({free_text} free).",
                    "Free up some space on that drive.",
                )
            )

    temp_bytes = int(facts.get("temp_bytes", 0))
    if temp_bytes >= TEMP_CRITICAL_BYTES:
        findings.append(
            _finding(
                "critical",
                f"Temp folder size is {human_bytes(temp_bytes)} (very large).",
                "Ask BOB to propose cleanup and approve temp files.",
            )
        )
    elif temp_bytes >= TEMP_WARNING_BYTES:
        findings.append(
            _finding(
                "warning",
                f"Temp folder size is {human_bytes(temp_bytes)} (large).",
                "Ask BOB to propose cleanup and approve temp files.",
            )
        )

    recycle_bytes = int(facts.get("recycle_bytes", 0))
    if recycle_bytes >= RECYCLE_BIN_CRITICAL_BYTES:
        findings.append(
            _finding(
                "critical",
                f"Recycle Bin size is {human_bytes(recycle_bytes)} (very large).",
                "Review the Recycle Bin item on the approval card.",
            )
        )
    elif recycle_bytes >= RECYCLE_BIN_WARNING_BYTES:
        findings.append(
            _finding(
                "warning",
                f"Recycle Bin size is {human_bytes(recycle_bytes)} (large).",
                "Review the Recycle Bin item on the approval card.",
            )
        )

    startup_count = int(facts.get("startup_count", 0))
    if startup_count >= STARTUP_WARNING_COUNT:
        findings.append(
            _finding(
                "warning",
                f"Startup programs: {startup_count} start with Windows (high).",
                "Disable unnecessary startup apps in Task Manager.",
            )
        )
    elif startup_count >= STARTUP_INFO_COUNT:
        findings.append(
            _finding(
                "info",
                f"Startup programs: {startup_count} start with Windows.",
                "Review startup apps if boot feels slow.",
            )
        )

    uptime_days = _percent(facts.get("uptime_days"))
    if uptime_days > UPTIME_INFO_DAYS:
        findings.append(
            _finding(
                "info",
                f"PC uptime is {uptime_days:.1f} days.",
                "Restart to apply updates and clear memory leaks.",
            )
        )

    battery = facts.get("battery_percent")
    plugged_in = facts.get("plugged_in")
    if battery is not None and int(battery) < BATTERY_LOW_PERCENT and plugged_in is False:
        findings.append(
            _finding(
                "warning",
                f"Battery level is {int(battery)}% and the PC is unplugged.",
                "Plug in the charger.",
            )
        )
    if facts.get("battery_saver") is True:
        findings.append(
            _finding(
                "info",
                "Battery saver is on, which limits performance.",
                "Plug in the charger for full speed.",
            )
        )

    if top_cpu and _percent(top_cpu[0].get("cpu_percent")) >= PROCESS_CPU_WARNING_PERCENT:
        findings.append(
            _finding(
                "warning",
                f"{_process_label(top_cpu[0])} is using "
                f"{_percent(top_cpu[0].get('cpu_percent')):.1f}% CPU (system-wide).",
                _process_fix(top_cpu[0]),
            )
        )

    if (
        top_ram
        and int(float(top_ram[0].get("ram_mb", 0)) * 1024 ** 2)
        >= PROCESS_RAM_WARNING_BYTES
    ):
        findings.append(
            _finding(
                "warning",
                f"{_process_label(top_ram[0])} is using "
                f"{float(top_ram[0].get('ram_mb', 0)) / 1024:.1f} GB RAM.",
                _process_fix(top_ram[0]),
            )
        )

    if not findings:
        findings.append(
            _finding(
                "info",
                "No obvious resource problems were found.",
                "Nothing to do right now.",
            )
        )

    findings.sort(key=lambda item: SEVERITY_ORDER.get(item["severity"], 3))
    return findings


def _fit_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the serialized result under MAX_RESULT_CHARS without losing criticals."""
    limit = MAX_RESULT_CHARS - 80

    def _encoded() -> str:
        return json.dumps(payload, default=str)

    for severity in ("info", "warning"):
        while len(_encoded()) > limit:
            findings = payload.get("findings", [])
            index = None
            for pos in range(len(findings) - 1, -1, -1):
                if findings[pos]["severity"] == severity:
                    index = pos
                    break
            if index is None:
                break
            findings.pop(index)

    for key in ("top_cpu", "top_ram"):
        while len(_encoded()) > limit and len(payload.get(key, [])) > 3:
            payload[key].pop()

    while len(_encoded()) > limit and len(payload.get("drives", [])) > 1:
        payload["drives"].pop()

    for key in ("top_cpu", "top_ram"):
        while len(_encoded()) > limit and payload.get(key):
            payload[key].pop()

    while len(_encoded()) > limit and payload.get("drives"):
        payload["drives"].pop()

    while len(_encoded()) > limit and payload.get("findings"):
        if len(payload["findings"]) <= 1:
            break
        payload["findings"].pop()

    metrics = payload.get("metrics", {})
    for key in (
        "temp_files",
        "recycle_count",
        "ram_used_gb",
        "ram_total_gb",
        "swap_used_gb",
        "battery_percent",
        "plugged_in",
        "battery_saver",
        "startup_count",
        "uptime_days",
    ):
        if len(_encoded()) <= limit:
            break
        metrics.pop(key, None)

    return payload


def _diagnose_system() -> ToolResult:
    """Collect compact, ranked system findings."""
    try:
        facts = _collect_facts()
        findings = _build_findings(facts)

        metrics = {
            "cpu_percent": facts["cpu_percent"],
            "logical_cores": facts["logical_cores"],
            "ram_percent": facts["ram_percent"],
            "ram_used_gb": facts["ram_used_gb"],
            "ram_total_gb": facts["ram_total_gb"],
            "swap_percent": facts["swap_percent"],
            "swap_used_gb": facts["swap_used_gb"],
            "temp_bytes": facts["temp_bytes"],
            "temp_files": facts["temp_files"],
            "recycle_count": facts["recycle_count"],
            "recycle_bytes": facts["recycle_bytes"],
            "startup_count": facts["startup_count"],
            "uptime_days": facts["uptime_days"],
            "battery_percent": facts["battery_percent"],
            "plugged_in": facts["plugged_in"],
            "battery_saver": facts["battery_saver"],
        }

        payload = _fit_payload(
            {
                "findings": findings,
                "metrics": metrics,
                "top_cpu": facts["top_cpu"],
                "top_ram": facts["top_ram"],
                "drives": [
                    {
                        "mountpoint": str(drive.get("mountpoint")),
                        "free_bytes": int(drive.get("free", 0)),
                        "free_percent": round(
                            max(0.0, 100.0 - _percent(drive.get("percent"))), 1
                        ),
                    }
                    for drive in facts["drives"]
                ],
            }
        )
        return ToolResult(success=True, data=payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("System diagnosis failed (%s).", type(exc).__name__)
        return ToolResult(success=False, error="Could not diagnose the system.")


DIAGNOSE_SYSTEM_TOOL = Tool(
    name="diagnose_system",
    description=(
        "Diagnoses why the PC may be slow or hanging. Read-only. Returns ranked "
        "findings (critical, warning, info) with a one-line message and suggested "
        "fix, plus the top CPU and RAM process groups, disk space, temp folder size, "
        "Recycle Bin size, startup count, uptime, battery, and power mode. Process "
        "CPU is normalised to system-wide percentages (the same scale as overall "
        "CPU). BOB/Ollama processes are marked as protected and are never suggested "
        "for closing. Use this first when the user asks why their laptop is slow."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    fn=_diagnose_system,
)
