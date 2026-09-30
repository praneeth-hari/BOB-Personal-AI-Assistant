"""
app/tools/system_info.py
─────────────────────────
BOB tool: get_system_info

Returns CPU, RAM, GPU, and OS platform details.

Implementation notes:
  • Uses psutil for CPU/RAM – no shell commands.
  • Calls nvidia-smi with fixed, read-only arguments for GPU info.
    Falls back gracefully if nvidia-smi is absent.
  • Never executes arbitrary user-supplied commands.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Any, Dict, List

import psutil

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)


# ─── Implementation ───────────────────────────────────────────────────────────

def _get_gpu_info() -> List[Dict[str, Any]]:
    """
    Query GPU info via nvidia-smi using a fixed, read-only command.
    Returns an empty list if nvidia-smi is not available or fails.
    """
    _CMD = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            _CMD,
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,          # shell=False: no arbitrary shell interpretation
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        gpus: List[Dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            gpus.append({
                "name": parts[0],
                "vram_total_mb": int(parts[1]) if parts[1].isdigit() else parts[1],
                "vram_used_mb":  int(parts[2]) if parts[2].isdigit() else parts[2],
                "temperature_c": int(parts[3]) if parts[3].isdigit() else parts[3],
                "utilization_percent": int(parts[4]) if parts[4].isdigit() else parts[4],
            })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def _get_system_info() -> ToolResult:
    """Collect and return structured system information."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        vm = psutil.virtual_memory()

        info: Dict[str, Any] = {
            "cpu": {
                "usage_percent": cpu_percent,
                "logical_cores":  psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
            },
            "ram": {
                "total_gb":      round(vm.total     / (1024 ** 3), 2),
                "used_gb":       round(vm.used      / (1024 ** 3), 2),
                "free_gb":       round(vm.available / (1024 ** 3), 2),
                "usage_percent": vm.percent,
            },
            "platform": {
                "system":    platform.system(),
                "version":   platform.version(),
                "machine":   platform.machine(),
                "processor": platform.processor(),
            },
            "gpu": _get_gpu_info(),
        }
        return ToolResult(success=True, data=info)

    except Exception as exc:  # noqa: BLE001
        logger.exception("get_system_info failed.")
        return ToolResult(success=False, error=str(exc))


# ─── Tool descriptor ──────────────────────────────────────────────────────────

SYSTEM_INFO_TOOL = Tool(
    name="get_system_info",
    description=(
        "Returns current system resource usage: CPU percentage, RAM (total/used/free), "
        "GPU information (if an NVIDIA GPU is present), and OS platform details. "
        "No arguments are required."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    fn=_get_system_info,
)
