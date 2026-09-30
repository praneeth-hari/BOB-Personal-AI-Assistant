"""
app/tools/get_battery_status.py
───────────────────────────────
BOB tool: get_battery_status

Safely queries the computer's battery status using psutil.sensors_battery().
Returns battery percentage, power plug state, charging status, and
estimated remaining battery time.

SAFETY & SECURITY MODEL
───────────────────────
• Completely read-only.
• Uses standard library and psutil directly in Python.
• NEVER executes shell commands, PowerShell, subprocesses, or network calls.
• Gracefully handles desktops/systems where battery hardware is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import psutil

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)


# ─── Formatting Helpers ────────────────────────────────────────────────────────

def _format_time_remaining(secsleft: Optional[int]) -> tuple[Optional[int], Optional[str]]:
    """
    Format battery remaining seconds into a human-readable string.

    :param secsleft: Seconds remaining or psutil constants (POWER_TIME_UNKNOWN, POWER_TIME_UNLIMITED).
    :returns: Tuple of (seconds or None, human-readable string or None).
    """
    if secsleft is None:
        return None, "Unknown"

    try:
        secs_int = int(secsleft)
    except (ValueError, TypeError):
        return None, "Unknown"

    # psutil.POWER_TIME_UNLIMITED is -2 (plugged in and charging / full)
    if secs_int == getattr(psutil, "POWER_TIME_UNLIMITED", -2):
        return None, "Unlimited (Plugged In)"

    # psutil.POWER_TIME_UNKNOWN is -1 (battery is still calculating)
    if secs_int == getattr(psutil, "POWER_TIME_UNKNOWN", -1) or secs_int < 0:
        return None, "Calculating / Unknown"

    # Valid positive seconds remaining
    hours = secs_int // 3600
    minutes = (secs_int % 3600) // 60
    if hours > 0:
        h_unit = "hour" if hours == 1 else "hours"
        m_unit = "minute" if minutes == 1 else "minutes"
        formatted = f"{hours} {h_unit} {minutes} {m_unit}"
    else:
        m_unit = "minute" if minutes == 1 else "minutes"
        formatted = f"{minutes} {m_unit}"

    return secs_int, formatted


# ─── Implementation ───────────────────────────────────────────────────────────

def _get_battery_status(**kwargs: Any) -> ToolResult:
    """
    Query and return current battery status.

    :returns: ToolResult with structured battery information.
    """
    try:
        battery = psutil.sensors_battery()
    except Exception as exc:  # noqa: BLE001
        logger.exception("psutil.sensors_battery() raised an unexpected exception.")
        return ToolResult(
            success=False,
            error=f"Failed to query battery status: {exc}",
        )

    # When no battery is present (e.g. desktop PC connected directly to AC)
    if battery is None:
        return ToolResult(
            success=True,
            data={
                "available": False,
                "percent": None,
                "power_plugged": None,
                "is_plugged_in": None,
                "charging_status": "unavailable",
                "status": "No battery detected (system may be a desktop computer on AC power)",
                "time_remaining_seconds": None,
                "time_remaining_formatted": None,
                "message": "No battery detected on this system. It appears to be running directly on AC power.",
            },
        )

    # Safely extract attributes from the namedtuple
    raw_percent = getattr(battery, "percent", None)
    raw_plugged = getattr(battery, "power_plugged", None)
    raw_secsleft = getattr(battery, "secsleft", None)

    # 1. Battery Percentage
    try:
        if raw_percent is not None:
            percent_val: Optional[float] = round(float(raw_percent), 1)
        else:
            percent_val = None
    except (ValueError, TypeError):
        percent_val = None

    # 2. Plugged in status
    is_plugged: Optional[bool]
    if raw_plugged is None:
        is_plugged = None
    else:
        is_plugged = bool(raw_plugged)

    # 3. Charging status
    if is_plugged is True:
        if percent_val is not None and percent_val >= 99.9:
            charging_status = "fully_charged"
            status_text = f"{percent_val}% - Fully Charged (Plugged In)"
        else:
            charging_status = "charging"
            status_text = f"{percent_val}% - Charging (Plugged In)" if percent_val is not None else "Charging (Plugged In)"
    elif is_plugged is False:
        charging_status = "discharging"
        status_text = f"{percent_val}% - On Battery (Discharging)" if percent_val is not None else "On Battery (Discharging)"
    else:
        charging_status = "unknown"
        status_text = f"{percent_val}% - Unknown Power State" if percent_val is not None else "Unknown Power State"

    # 4. Remaining time
    secs_remaining, time_formatted = _format_time_remaining(raw_secsleft)

    info: Dict[str, Any] = {
        "available": True,
        "percent": percent_val,
        "power_plugged": is_plugged,
        "is_plugged_in": is_plugged,
        "charging_status": charging_status,
        "status": status_text,
        "time_remaining_seconds": secs_remaining,
        "time_remaining_formatted": time_formatted,
    }

    return ToolResult(success=True, data=info)


# ─── Tool Descriptor ──────────────────────────────────────────────────────────

BATTERY_STATUS_TOOL = Tool(
    name="get_battery_status",
    description=(
        "Returns the current battery status of the computer, including battery percentage, "
        "whether the power cable is plugged in, charging status, and estimated remaining battery time. "
        "Use this tool when the user asks: "
        "- 'How much battery do I have?' "
        "- 'What's my battery percentage?' "
        "- 'Is my laptop charging?' "
        "- 'Is the charger plugged in?' "
        "- 'How much battery time is left?' "
        "Safe and read-only. No arguments required."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    fn=_get_battery_status,
)
