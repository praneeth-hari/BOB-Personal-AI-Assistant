"""
app/tools/audio_volume.py
─────────────────────────
BOB tool: audio_volume

Safely reads and controls the Windows system master audio volume and mute state
using the Python Core Audio Windows (pycaw) COM library.

SAFETY & SECURITY MODEL
───────────────────────
• Completely local Windows audio endpoint control via pycaw / WASAPI COM interface.
• Strictly avoids PowerShell, command prompt, subprocess, shell=True, or external APIs.
• Validates operations and bounds volume to [0, 100].
• Gracefully catches audio device / COM errors and returns clean ToolResult responses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)

# Allowed operations
SUPPORTED_OPERATIONS = {"get", "set", "mute", "unmute"}


# ─── Audio Endpoint Helper ───────────────────────────────────────────────────

def _get_volume_endpoint() -> Any:
    """
    Acquire the master audio endpoint volume interface via pycaw.

    Initialises COM for the current thread if needed, then returns
    the IAudioEndpointVolume interface for the default speaker device.
    :raises RuntimeError: If no audio playback device is found or pycaw fails.
    """
    try:
        import comtypes
        comtypes.CoInitialize()
    except Exception:  # noqa: BLE001
        pass

    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    except ImportError as exc:
        raise RuntimeError(
            "The 'pycaw' library is required for Windows audio volume control."
        ) from exc

    try:
        device = AudioUtilities.GetSpeakers()
        if device is None:
            raise RuntimeError("No active speaker or audio playback device detected.")

        # Modern pycaw (AudioDevice.EndpointVolume property)
        if hasattr(device, "EndpointVolume") and device.EndpointVolume is not None:
            return device.EndpointVolume

        # Fallback for devices exposing Activate directly or on _dev
        import ctypes
        from comtypes import CLSCTX_ALL

        dev_target = getattr(device, "_dev", device)
        if hasattr(dev_target, "Activate"):
            interface = dev_target.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))

        raise RuntimeError("Unable to access EndpointVolume interface on speaker device.")
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Failed to access Windows audio endpoint: {exc}") from exc


# ─── Tool Implementation ──────────────────────────────────────────────────────

def _audio_volume(
    operation: str = "get",
    volume: Optional[Union[int, float, str]] = None,
    **kwargs: Any,
) -> ToolResult:
    """
    Read or control the system master volume.

    :param operation: One of 'get', 'set', 'mute', 'unmute'.
    :param volume:    Target volume percentage (0-100), required only when operation='set'.
    :returns:         ToolResult with structured audio state and human message.
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

    # 2. Validate volume for 'set' operation
    vol_float: Optional[float] = None
    if op == "set":
        if volume is None or str(volume).strip() == "":
            return ToolResult(
                success=False,
                error="The 'volume' parameter is required for the 'set' operation (0-100).",
            )
        try:
            # Strip any trailing '%' (e.g. "50%", "30.5%")
            clean_str = str(volume).strip().rstrip("%").strip()
            vol_float = float(clean_str)
        except (ValueError, TypeError):
            return ToolResult(
                success=False,
                error=f"Invalid volume value {volume!r}. Volume must be a number between 0 and 100.",
            )

        if vol_float < 0.0 or vol_float > 100.0:
            return ToolResult(
                success=False,
                error=f"Volume {vol_float} is out of range. Must be between 0 and 100.",
            )

    # 3. Acquire Windows Audio Endpoint
    try:
        endpoint = _get_volume_endpoint()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to acquire Windows audio endpoint.")
        return ToolResult(
            success=False,
            error=f"Windows audio device error: {exc}",
        )

    # 4. Perform requested operation
    try:
        if op == "set":
            assert vol_float is not None
            scalar = max(0.0, min(1.0, vol_float / 100.0))
            endpoint.SetMasterVolumeLevelScalar(scalar, None)
        elif op == "mute":
            endpoint.SetMute(1, None)
        elif op == "unmute":
            endpoint.SetMute(0, None)
        elif op == "get":
            pass  # purely reading current status

        # 5. Read current state after action
        current_scalar = float(endpoint.GetMasterVolumeLevelScalar())
        current_percent = round(current_scalar * 100)
        is_muted = bool(endpoint.GetMute())

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed executing audio operation %r on endpoint.", op)
        return ToolResult(
            success=False,
            error=f"Failed to execute audio operation {op!r}: {exc}",
        )

    # 6. Construct natural human-friendly summary message
    if op == "get":
        if is_muted:
            message = f"Master volume is currently {current_percent}%, but audio is MUTED."
        else:
            message = f"Master volume is currently {current_percent}%."
    elif op == "set":
        if is_muted:
            message = f"Master volume set to {current_percent}% (note: audio is currently muted)."
        else:
            message = f"Master volume set to {current_percent}%."
    elif op == "mute":
        message = f"System audio muted (volume is {current_percent}%)."
    elif op == "unmute":
        message = f"System audio unmuted (volume is {current_percent}%)."
    else:
        message = f"Volume is {current_percent}%."

    data: Dict[str, Any] = {
        "operation": op,
        "volume_percent": current_percent,
        "muted": is_muted,
        "message": message,
    }

    return ToolResult(success=True, data=data)


# ─── Tool Descriptor ──────────────────────────────────────────────────────────

AUDIO_VOLUME_TOOL = Tool(
    name="audio_volume",
    description=(
        "Reads or controls the Windows master audio volume and mute state. "
        "Use this tool when the user asks: "
        "- 'What's my volume?' (operation='get') "
        "- 'Set volume to 50%' (operation='set', volume=50) "
        "- 'Turn the volume down to 30%' (operation='set', volume=30) "
        "- 'Mute my laptop' (operation='mute') "
        "- 'Unmute' (operation='unmute') "
        "- 'Set volume to 80' (operation='set', volume=80)"
    ),
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["get", "set", "mute", "unmute"],
                "description": "Audio operation to perform: 'get', 'set', 'mute', or 'unmute'.",
            },
            "volume": {
                "type": "number",
                "description": "Target volume percentage from 0 to 100. Required only for 'set' operation.",
            },
        },
        "required": ["operation"],
    },
    fn=_audio_volume,
)
