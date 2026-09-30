"""
app/tools/window_control.py
───────────────────────────
BOB tool: window_control

Safely inspects and controls application windows on Windows using direct
Win32 GUI APIs (pywin32 / ctypes user32 & dwmapi).

SAFETY & SECURITY MODEL
───────────────────────
• Completely local Windows GUI API access via pywin32 and ctypes.
• Strictly avoids PowerShell, CMD, terminal commands, or shell execution.
• Does NOT terminate processes, close applications, launch applications, or simulate mouse/keyboard.
• Does NOT accept raw HWNDs, arbitrary PIDs, executable paths, or code strings from LLM input.
• Resolves human-readable queries strictly against currently enumerated, visible top-level windows.
• If multiple windows match a query, returns a structured ambiguous result for user clarification.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)

SUPPORTED_OPERATIONS = {"list", "get_active", "focus", "minimize", "maximize", "restore"}
OPERATIONS_REQUIRING_QUERY = {"focus", "minimize", "maximize", "restore"}

DEFAULT_LIST_LIMIT = 40
MAX_LIST_LIMIT = 100

IGNORE_TITLES = {
    "Program Manager",
    "Windows Shell Experience Host",
    "Windows Input Experience",
}

_desktop_initialized = False


# ─── Low-level Win32 Helpers ──────────────────────────────────────────────────

def _ensure_interactive_desktop() -> None:
    """
    Ensure the current thread is attached to the default interactive desktop.

    Background worker threads or service threads may start on a separate desktop;
    switching to 'default' ensures visible user windows are properly enumerated.
    """
    global _desktop_initialized
    if _desktop_initialized:
        return
    try:
        user32 = ctypes.windll.user32
        # DESKTOP_READOBJECTS (0x0001) | DESKTOP_WRITEOBJECTS (0x0080) | DESKTOP_SWITCHDESKTOP (0x0100)
        h_desk = user32.OpenDesktopW("default", 0, False, 0x01FF)
        if h_desk:
            user32.SetThreadDesktop(h_desk)
            _desktop_initialized = True
    except Exception as exc:
        logger.debug("Failed to switch thread desktop to default: %s", exc)



def _is_cloaked(hwnd: int) -> bool:
    """Check if window is cloaked (e.g. suspended UWP app)."""
    try:
        dwm = ctypes.windll.dwmapi
        cloaked = ctypes.c_int(0)
        # DWMWA_CLOAKED = 14
        hr = dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        return hr == 0 and cloaked.value != 0
    except Exception:
        return False


def _get_window_state(hwnd: int) -> Tuple[bool, bool]:
    """
    Returns (is_minimized, is_maximized).
    """
    try:
        import win32con
        import win32gui

        placement = win32gui.GetWindowPlacement(hwnd)
        show_cmd = placement[1]
        is_min = show_cmd in (win32con.SW_SHOWMINIMIZED, win32con.SW_MINIMIZE, 11) or bool(
            win32gui.IsIconic(hwnd)
        )
        is_max = show_cmd == win32con.SW_SHOWMAXIMIZED
        return is_min, is_max
    except Exception:
        try:
            import win32gui
            return bool(win32gui.IsIconic(hwnd)), False
        except Exception:
            return False, False


def _get_process_info(hwnd: int) -> Tuple[Optional[int], Optional[str]]:
    """
    Extract (process_id, process_name) for a window handle.
    """
    try:
        import win32process
        import psutil

        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if not pid:
            return None, None
        proc_name = None
        try:
            p = psutil.Process(pid)
            proc_name = p.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return pid, proc_name
    except Exception:
        return None, None


def _get_window_info(hwnd: int) -> Optional[Dict[str, Any]]:
    """
    Validate and extract information for a top-level application window.
    Returns None if window should be ignored (invisible, tool window, empty title, cloaked, etc.).
    """
    try:
        import win32con
        import win32gui

        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            return None

        title = win32gui.GetWindowText(hwnd).strip()
        if not title or title in IGNORE_TITLES:
            return None

        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        is_tool = bool(ex_style & win32con.WS_EX_TOOLWINDOW)
        is_app = bool(ex_style & win32con.WS_EX_APPWINDOW)
        if is_tool and not is_app:
            return None

        rect = win32gui.GetWindowRect(hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if not win32gui.IsIconic(hwnd) and (width <= 0 or height <= 0):
            return None

        if _is_cloaked(hwnd):
            return None

        pid, proc_name = _get_process_info(hwnd)
        is_min, is_max = _get_window_state(hwnd)

        return {
            "hwnd": hwnd,
            "title": title,
            "process_name": proc_name,
            "process_id": pid,
            "is_minimized": is_min,
            "is_maximized": is_max,
        }
    except Exception as exc:
        logger.debug("Error inspecting window %s: %s", hwnd, exc)
        return None


def _enumerate_windows() -> List[Dict[str, Any]]:
    """Enumerate all visible top-level application windows."""
    try:
        import win32gui
    except ImportError as exc:
        raise RuntimeError("pywin32 is required for window control on Windows.") from exc

    _ensure_interactive_desktop()

    windows: List[Dict[str, Any]] = []

    def _callback(hwnd: int, extra: Any) -> bool:
        info = _get_window_info(hwnd)
        if info is not None:
            windows.append(info)
        return True

    try:
        win32gui.EnumWindows(_callback, None)
    except Exception as exc:
        try:
            user32 = ctypes.windll.user32
            h_desk = user32.OpenDesktopW("default", 0, False, 0x01FF)
            if h_desk:
                try:
                    win32gui.EnumDesktopWindows(h_desk, _callback, None)
                finally:
                    user32.CloseDesktop(h_desk)
            else:
                raise exc
        except Exception:
            raise exc

    return windows



def _force_foreground_window(hwnd: int) -> None:
    """Bring the window to the foreground even if invoked from a background thread."""
    try:
        import win32gui
        import win32process

        user32 = ctypes.windll.user32
        curr_tid = win32process.GetCurrentThreadId()
        fore_hwnd = win32gui.GetForegroundWindow()
        fore_tid, _ = win32process.GetWindowThreadProcessId(fore_hwnd) if fore_hwnd else (0, 0)
        if fore_tid and fore_tid != curr_tid:
            user32.AttachThreadInput(fore_tid, curr_tid, True)
            user32.SetForegroundWindow(hwnd)
            user32.AttachThreadInput(fore_tid, curr_tid, False)
        else:
            user32.SetForegroundWindow(hwnd)
    except Exception:
        try:
            import win32gui
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass


# ─── Tool Implementation ──────────────────────────────────────────────────────

def _window_control(
    operation: str = "list",
    window_query: Optional[str] = None,
    limit: Optional[int] = DEFAULT_LIST_LIMIT,
    **kwargs: Any,
) -> ToolResult:
    """
    Safely inspect and control application windows on Windows.

    :param operation: 'list', 'get_active', 'focus', 'minimize', 'maximize', or 'restore'.
    :param window_query: Window title or process name to match.
    :param limit: Result limit for 'list' operation (default 40, max 100).
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

    # 2. Validate window_query when required
    query = str(window_query or "").strip()
    if op in OPERATIONS_REQUIRING_QUERY and not query:
        return ToolResult(
            success=False,
            error=f"The 'window_query' parameter is required for the '{op}' operation.",
        )

    # 3. Ensure Win32 libraries are available
    try:
        import win32con
        import win32gui
    except ImportError as exc:
        return ToolResult(
            success=False,
            error=f"Window control is unavailable: {exc}",
        )

    # 4. Handle 'get_active'
    if op == "get_active":
        try:
            _ensure_interactive_desktop()
            fore_hwnd = win32gui.GetForegroundWindow()
            if not fore_hwnd or not win32gui.IsWindow(fore_hwnd):
                return ToolResult(
                    success=True,
                    data={
                        "operation": "get_active",
                        "found": False,
                        "title": None,
                        "process_name": None,
                        "process_id": None,
                        "is_minimized": False,
                        "is_maximized": False,
                        "message": "No active foreground window found.",
                    },
                )

            info = _get_window_info(fore_hwnd)
            if info is None:
                # Active window might have an unconventional style; extract core details directly
                title = win32gui.GetWindowText(fore_hwnd).strip() or "Untitled Window"
                pid, proc_name = _get_process_info(fore_hwnd)
                is_min, is_max = _get_window_state(fore_hwnd)
                info = {
                    "hwnd": fore_hwnd,
                    "title": title,
                    "process_name": proc_name,
                    "process_id": pid,
                    "is_minimized": is_min,
                    "is_maximized": is_max,
                }

            return ToolResult(
                success=True,
                data={
                    "operation": "get_active",
                    "found": True,
                    "title": info["title"],
                    "process_name": info["process_name"],
                    "process_id": info["process_id"],
                    "is_minimized": info["is_minimized"],
                    "is_maximized": info["is_maximized"],
                    "message": (
                        f"Active window: '{info['title']}' "
                        f"({info['process_name'] or 'PID ' + str(info['process_id'])})."
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to get active window.")
            return ToolResult(success=False, error=f"Failed to get active window: {exc}")

    # 5. Handle 'list'
    if op == "list":
        try:
            clamped_limit = DEFAULT_LIST_LIMIT
            if limit is not None:
                try:
                    clamped_limit = max(1, min(MAX_LIST_LIMIT, int(limit)))
                except (ValueError, TypeError):
                    clamped_limit = DEFAULT_LIST_LIMIT

            all_windows = _enumerate_windows()
            bounded = all_windows[:clamped_limit]

            window_items = [
                {
                    "title": w["title"],
                    "process_name": w["process_name"],
                    "process_id": w["process_id"],
                    "is_minimized": w["is_minimized"],
                    "is_maximized": w["is_maximized"],
                }
                for w in bounded
            ]

            return ToolResult(
                success=True,
                data={
                    "operation": "list",
                    "count": len(window_items),
                    "windows": window_items,
                    "message": f"Found {len(window_items)} visible application window(s).",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to list application windows.")
            return ToolResult(success=False, error=f"Failed to list windows: {exc}")

    # 6. Resolve target window for actions ('focus', 'minimize', 'maximize', 'restore')
    try:
        all_windows = _enumerate_windows()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to enumerate windows for action.")
        return ToolResult(success=False, error=f"Failed to enumerate windows: {exc}")

    q_lower = query.lower()

    # Precedence:
    # 1. Exact match on title or process_name
    exact_matches: List[Dict[str, Any]] = [
        w for w in all_windows
        if w["title"].strip().lower() == q_lower
        or (w["process_name"] and w["process_name"].lower() == q_lower)
        or (w["process_name"] and w["process_name"].lower().replace(".exe", "") == q_lower)
    ]

    if len(exact_matches) == 1:
        matched = exact_matches
    elif len(exact_matches) > 1:
        matched = exact_matches
    else:
        # 2. Case-insensitive substring match on title or process_name
        sub_matches: List[Dict[str, Any]] = [
            w for w in all_windows
            if q_lower in w["title"].lower()
            or (w["process_name"] and (
                q_lower in w["process_name"].lower()
                or q_lower in w["process_name"].lower().replace(".exe", "")
            ))
        ]
        matched = sub_matches

    # Handle no match
    if not matched:
        return ToolResult(
            success=True,
            data={
                "operation": op,
                "found": False,
                "message": f"No visible window matching '{query}' was found.",
            },
        )

    # Handle multiple matches (ambiguous)
    if len(matched) > 1:
        match_summaries = [
            {
                "title": m["title"],
                "process_name": m["process_name"],
                "process_id": m["process_id"],
            }
            for m in matched
        ]
        titles_str = ", ".join(repr(m["title"]) for m in matched)
        return ToolResult(
            success=True,
            data={
                "operation": op,
                "ambiguous": True,
                "matches": match_summaries,
                "message": (
                    f"Multiple windows matched '{query}': {titles_str}. "
                    "Please clarify which window you want to control."
                ),
            },
        )

    target = matched[0]
    hwnd = target["hwnd"]

    # 7. Execute action
    try:
        if not win32gui.IsWindow(hwnd):
            return ToolResult(
                success=False,
                error=f"Window '{target['title']}' is no longer accessible.",
            )

        if op == "focus":
            if target["is_minimized"]:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            else:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.BringWindowToTop(hwnd)
            _force_foreground_window(hwnd)
            action_msg = f"Brought window '{target['title']}' to the foreground."

        elif op == "minimize":
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            action_msg = f"Minimized window '{target['title']}'."

        elif op == "maximize":
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            action_msg = f"Maximized window '{target['title']}'."

        elif op == "restore":
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            action_msg = f"Restored window '{target['title']}'."

        else:
            action_msg = f"Completed {op} on '{target['title']}'."

        return ToolResult(
            success=True,
            data={
                "operation": op,
                "found": True,
                "title": target["title"],
                "process_name": target["process_name"],
                "process_id": target["process_id"],
                "message": action_msg,
            },
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed executing window operation %r on hwnd %s", op, hwnd)
        return ToolResult(
            success=False,
            error=f"Failed to execute '{op}' on window '{target['title']}': {exc}",
        )


# ─── Tool Descriptor ──────────────────────────────────────────────────────────

WINDOW_CONTROL_TOOL = Tool(
    name="window_control",
    description=(
        "Safely controls and inspects application windows on Windows. "
        "Supported operations: "
        "- 'list': list visible top-level application windows. "
        "- 'get_active': return the currently active/foreground window. "
        "- 'focus': bring a window matching window_query to the foreground. "
        "- 'minimize': minimize a window matching window_query. "
        "- 'maximize': maximize a window matching window_query. "
        "- 'restore': restore a minimized or maximized window matching window_query. "
        "window_query matches application name or window title (e.g. 'Notepad', 'Chrome', 'Spotify')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["list", "get_active", "focus", "minimize", "maximize", "restore"],
                "description": "Operation to perform: 'list', 'get_active', 'focus', 'minimize', 'maximize', or 'restore'.",
            },
            "window_query": {
                "type": "string",
                "description": "Application or window title to match (e.g. 'Notepad', 'Chrome', 'VS Code'). Required for focus, minimize, maximize, restore.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of windows to return for the 'list' operation (default: 40, max: 100).",
            },
        },
        "required": ["operation"],
    },
    fn=_window_control,
)
