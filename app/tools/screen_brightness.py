"""
app/tools/screen_brightness.py
──────────────────────────────
BOB tool: screen_brightness

Safely reads and controls the Windows display screen brightness percentage
using the screen_brightness_control library (WMI / DXVA2).

SAFETY & SECURITY MODEL
───────────────────────
• Completely local Windows display brightness control.
• Strictly avoids PowerShell, CMD, terminal commands, subprocesses, or shell=True.
• No filesystem, process, registry, or network modifications.
• Validates all operations, bounds target brightness to [0, 100], and bounds step to [1, 50].
• Gracefully catches hardware/driver errors when display brightness control is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)

SUPPORTED_OPERATIONS = {"get", "set", "increase", "decrease"}
DEFAULT_STEP_PERCENT = 10
MIN_STEP_PERCENT = 1
MAX_STEP_PERCENT = 50


# ─── Low-level Screen Brightness Helpers ────────────────────────────────────────

def _get_sbc_module() -> Any:
    """Import and return screen_brightness_control, or raise RuntimeError."""
    try:
        import screen_brightness_control as sbc
        return sbc
    except ImportError as exc:
        raise RuntimeError(
            "The 'screen_brightness_control' library is required for screen brightness control."
        ) from exc


def _read_current_brightness() -> Optional[int]:
    """
    Read the primary display brightness percentage.

    :returns: Current brightness (0-100) or None if unavailable.
    """
    try:
        sbc = _get_sbc_module()
        val = sbc.get_brightness()
        if isinstance(val, list) and len(val) > 0:
            return int(round(float(val[0])))
        if isinstance(val, (int, float)):
            return int(round(float(val)))
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read display brightness: %s", exc)
        return None


def _apply_brightness(target: int) -> int:
    """
    Apply the requested brightness percentage to the display.

    :param target: Target brightness between 0 and 100.
    :returns: Verified brightness after change.
    :raises RuntimeError: If setting brightness fails.
    """
    sbc = _get_sbc_module()
    clamped = max(0, min(100, int(target)))
    try:
        sbc.set_brightness(clamped)
    except Exception as exc:
        raise RuntimeError(f"Hardware display brightness update failed: {exc}") from exc

    # Confirm updated brightness
    verified = _read_current_brightness()
    return verified if verified is not None else clamped


# ─── Tool Implementation ──────────────────────────────────────────────────────

def _screen_brightness(
    operation: str = "get",
    brightness_percent: Optional[Union[int, float, str]] = None,
    step_percent: Optional[Union[int, float, str]] = None,
    **kwargs: Any,
) -> ToolResult:
    """
    Read or change the system screen brightness.

    :param operation:          'get', 'set', 'increase', or 'decrease'.
    :param brightness_percent: Target percentage (0-100), required only when operation='set'.
    :param step_percent:       Step percentage (1-50, default 10) for 'increase' or 'decrease'.
    :returns:                  ToolResult with structured data and human-friendly message.
    """
    op = str(operation or "").strip().lower()

    # 1. Validate operation
    if op not in SUPPORTED_OPERATIONS:
        return ToolResult(
            success=False,
            error=(
                f"Invalid operation {operation!r}. "
                f"Supported operations are: {', '.join(sorted(SUPPORTED_OPERATIONS))}."
            ),
        )

    # 2. Validate brightness_percent for 'set'
    target_percent: Optional[int] = None
    if op == "set":
        if brightness_percent is None or str(brightness_percent).strip() == "":
            return ToolResult(
                success=False,
                error="The 'brightness_percent' parameter is required for the 'set' operation (0-100).",
            )
        try:
            cleaned = str(brightness_percent).strip().rstrip("%").strip()
            num_val = float(cleaned)
        except (ValueError, TypeError):
            return ToolResult(
                success=False,
                error=(
                    f"Invalid brightness value {brightness_percent!r}. "
                    "Brightness must be a number between 0 and 100."
                ),
            )

        if num_val < 0.0 or num_val > 100.0:
            return ToolResult(
                success=False,
                error=f"Brightness value {num_val} is out of range. Must be between 0 and 100.",
            )

        target_percent = int(round(num_val))

    # 3. Validate step_percent for 'increase' / 'decrease'
    step: int = DEFAULT_STEP_PERCENT
    if op in ("increase", "decrease"):
        if step_percent is not None and str(step_percent).strip() != "":
            try:
                cleaned_step = str(step_percent).strip().rstrip("%").strip()
                step_num = float(cleaned_step)
            except (ValueError, TypeError):
                return ToolResult(
                    success=False,
                    error=(
                        f"Invalid step value {step_percent!r}. "
                        f"Step must be a number between {MIN_STEP_PERCENT} and {MAX_STEP_PERCENT}."
                    ),
                )
            if step_num < MIN_STEP_PERCENT or step_num > MAX_STEP_PERCENT:
                return ToolResult(
                    success=False,
                    error=(
                        f"Step value {step_num} is out of range. "
                        f"Must be between {MIN_STEP_PERCENT} and {MAX_STEP_PERCENT}."
                    ),
                )
            step = int(round(step_num))

    # 4. Check hardware brightness availability
    try:
        current_brightness = _read_current_brightness()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error querying display brightness.")
        return ToolResult(
            success=False,
            error=f"Screen brightness query failed: {exc}",
        )

    # When brightness hardware is unavailable (e.g. desktop monitor without DDC/CI)
    if current_brightness is None:
        if op == "get":
            return ToolResult(
                success=True,
                data={
                    "available": False,
                    "operation": op,
                    "brightness_percent": None,
                    "previous_brightness_percent": None,
                    "message": (
                        "Screen brightness control is not available on this display or system "
                        "(e.g. external desktop monitor without DDC/CI or unsupported display driver)."
                    ),
                },
            )
        return ToolResult(
            success=False,
            error=(
                "Screen brightness control is not available on this display or system "
                "(e.g. external desktop monitor without DDC/CI or unsupported display driver)."
            ),
        )

    # 5. Execute requested operation
    previous_brightness: Optional[int] = current_brightness
    final_brightness: int = current_brightness

    try:
        if op == "get":
            message = f"Current screen brightness is {final_brightness}%."

        elif op == "set":
            assert target_percent is not None
            final_brightness = _apply_brightness(target_percent)
            message = (
                f"Screen brightness set to {final_brightness}% "
                f"(was {previous_brightness}%)."
                if previous_brightness != final_brightness
                else f"Screen brightness is already {final_brightness}%."
            )

        elif op == "increase":
            new_target = min(100, current_brightness + step)
            final_brightness = _apply_brightness(new_target)
            actual_increase = final_brightness - current_brightness
            if actual_increase > 0:
                message = f"Screen brightness increased by {actual_increase}% to {final_brightness}% (was {current_brightness}%)."
            else:
                message = f"Screen brightness is already at maximum ({final_brightness}%)."

        elif op == "decrease":
            new_target = max(0, current_brightness - step)
            final_brightness = _apply_brightness(new_target)
            actual_decrease = current_brightness - final_brightness
            if actual_decrease > 0:
                message = f"Screen brightness decreased by {actual_decrease}% to {final_brightness}% (was {current_brightness}%)."
            else:
                message = f"Screen brightness is already at minimum ({final_brightness}%)."

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed executing screen brightness operation %r", op)
        return ToolResult(
            success=False,
            error=f"Failed to change screen brightness: {exc}",
        )

    data: Dict[str, Any] = {
        "available": True,
        "operation": op,
        "brightness_percent": final_brightness,
        "previous_brightness_percent": previous_brightness if op != "get" else None,
        "message": message,
    }

    return ToolResult(success=True, data=data)


# ─── Tool Descriptor ──────────────────────────────────────────────────────────

SCREEN_BRIGHTNESS_TOOL = Tool(
    name="screen_brightness",
    description=(
        "Reads or changes the Windows display screen brightness percentage. "
        "Use this tool when the user asks: "
        "- 'What's my brightness?' (operation='get') "
        "- 'Set brightness to 50%' (operation='set', brightness_percent=50) "
        "- 'Increase brightness' (operation='increase', optional step_percent=10) "
        "- 'Decrease brightness' (operation='decrease', optional step_percent=10) "
        "- 'Make the screen brighter' (operation='increase') "
        "- 'Dim the screen' (operation='decrease') "
        "- 'Turn brightness up to 80' (operation='set', brightness_percent=80)"
    ),
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["get", "set", "increase", "decrease"],
                "description": "Brightness operation to perform: 'get', 'set', 'increase', or 'decrease'.",
            },
            "brightness_percent": {
                "type": "number",
                "description": "Target brightness percentage (0-100). Required only when operation is 'set'.",
            },
            "step_percent": {
                "type": "number",
                "description": "Percentage step to increase or decrease brightness (1-50, default: 10).",
            },
        },
        "required": ["operation"],
    },
    fn=_screen_brightness,
)
