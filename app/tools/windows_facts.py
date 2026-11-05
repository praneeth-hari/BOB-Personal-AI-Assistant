"""
app/tools/windows_facts.py
──────────────────────────
Read-only Windows facts shared by diagnostics and cleanup planning.

This module never deletes, terminates, writes, or launches anything. It only
inspects the local machine through psutil, ctypes, winreg, and the filesystem.
"""

from __future__ import annotations

import ctypes
import logging
import os
import stat
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

MAX_SCAN_ENTRIES = 20_000
TEMP_FILE_MIN_AGE_SECONDS = 24 * 60 * 60


@dataclass
class FileEntry:
    path: Path
    size: int
    mtime: float


@dataclass
class ScanResult:
    files: List[FileEntry] = field(default_factory=list)
    skipped: int = 0
    truncated: bool = False


@dataclass
class RecycleBinInfo:
    count: int = 0
    bytes: int = 0


def human_bytes(value: int) -> str:
    """Return a compact human-readable byte string."""
    try:
        size = float(max(0, int(value)))
    except (TypeError, ValueError):
        size = 0.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _is_reparse_stat(st: os.stat_result) -> bool:
    if stat.S_ISLNK(st.st_mode):
        return True
    attributes = getattr(st, "st_file_attributes", 0)
    return bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT)


def is_reparse_point(path: Path) -> bool:
    """True for symlinks, junctions, and other reparse points."""
    try:
        return _is_reparse_stat(os.lstat(path))
    except OSError:
        return False


def is_within_root(root: Path, candidate: Path) -> bool:
    """True when *candidate* resolves inside *root* on the same drive."""
    try:
        root_real = os.path.realpath(str(root))
        candidate_real = os.path.realpath(str(candidate))
        root_key = os.path.normcase(root_real)
        candidate_key = os.path.normcase(candidate_real)
        common = os.path.commonpath([root_key, candidate_key])
        return common == root_key
    except (OSError, ValueError):
        return False


def has_reparse_parent(root: Path, candidate: Path) -> bool:
    """
    True when *candidate*, or any lexical parent up to *root*, is a reparse
    point. This does not follow links, so it detects junctions before access.
    """
    root_key = os.path.normcase(os.path.abspath(str(root)))
    current = Path(os.path.abspath(str(candidate)))
    while True:
        if is_reparse_point(current):
            return True
        current_key = os.path.normcase(str(current))
        if current_key == root_key:
            return False
        parent = current.parent
        if parent == current:
            return True
        current = parent


def resolve_temp_root() -> Path:
    """Return the current user's temp directory as a Path."""
    return Path(tempfile.gettempdir())


def scan_files(
    root: Path,
    *,
    min_age_seconds: float = 0.0,
    max_entries: int = MAX_SCAN_ENTRIES,
    now: Optional[float] = None,
) -> ScanResult:
    """
    Enumerate regular files under *root* without following reparse points.

    Directories, symlinks, junctions, unreadable entries, and files younger
    than *min_age_seconds* are skipped. The scan is bounded by *max_entries*.
    """
    result = ScanResult()
    current_time = time.time() if now is None else now
    root = Path(root)
    stack: List[Path] = [root]

    if has_reparse_parent(root, root):
        result.skipped += 1
        return result

    while stack:
        current_dir = stack.pop()
        try:
            entries: Iterable[os.DirEntry[str]] = os.scandir(current_dir)
        except OSError:
            result.skipped += 1
            continue

        with entries:
            for entry in entries:
                if len(result.files) >= max_entries:
                    result.truncated = True
                    return result
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    result.skipped += 1
                    continue

                if _is_reparse_stat(st):
                    result.skipped += 1
                    continue

                if stat.S_ISDIR(st.st_mode):
                    stack.append(Path(entry.path))
                    continue

                if not stat.S_ISREG(st.st_mode):
                    result.skipped += 1
                    continue

                if min_age_seconds > 0 and current_time - st.st_mtime < min_age_seconds:
                    continue

                result.files.append(
                    FileEntry(path=Path(entry.path), size=int(st.st_size), mtime=st.st_mtime)
                )

    return result


class _SHQUERYRBINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("i64Size", ctypes.c_longlong),
        ("i64NumItems", ctypes.c_longlong),
    ]


def query_recycle_bin() -> RecycleBinInfo:
    """Query the current user's Recycle Bin count and size (read-only)."""
    if os.name != "nt":
        return RecycleBinInfo()
    try:
        info = _SHQUERYRBINFO()
        info.cbSize = ctypes.sizeof(info)
        shell32 = ctypes.windll.shell32
        shell32.SHQueryRecycleBinW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.POINTER(_SHQUERYRBINFO),
        ]
        shell32.SHQueryRecycleBinW.restype = ctypes.c_long
        hr = shell32.SHQueryRecycleBinW(None, ctypes.byref(info))
        if hr != 0:
            logger.debug("SHQueryRecycleBinW returned HRESULT %s", hr)
            return RecycleBinInfo()
        return RecycleBinInfo(count=int(info.i64NumItems), bytes=int(info.i64Size))
    except Exception:  # noqa: BLE001
        logger.debug("Recycle Bin query failed.", exc_info=True)
        return RecycleBinInfo()


def count_startup_programs() -> int:
    """Count read-only startup entries from Run keys and Startup folders."""
    count = 0

    try:
        import winreg

        run_keys = [
            (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
            (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Run",
            ),
        ]
        for hive, subkey in run_keys:
            try:
                with winreg.OpenKey(hive, subkey) as key:
                    index = 0
                    while True:
                        try:
                            winreg.EnumValue(key, index)
                        except OSError:
                            break
                        count += 1
                        index += 1
            except OSError:
                continue
    except Exception:  # noqa: BLE001
        logger.debug("Startup registry scan failed.", exc_info=True)

    startup_dirs = []
    appdata = os.environ.get("APPDATA")
    programdata = os.environ.get("PROGRAMDATA")
    if appdata:
        startup_dirs.append(
            Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        )
    if programdata:
        startup_dirs.append(
            Path(programdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        )

    for directory in startup_dirs:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            count += 1
                    except OSError:
                        continue
        except OSError:
            continue

    return count


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def get_power_status() -> Dict[str, object]:
    """Return AC/battery and battery-saver facts without running powercfg."""
    if os.name != "nt":
        return {"plugged_in": None, "battery_percent": None, "battery_saver": None}
    try:
        status = _SYSTEM_POWER_STATUS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return {"plugged_in": None, "battery_percent": None, "battery_saver": None}
        battery_percent = (
            None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent)
        )
        return {
            "plugged_in": status.ACLineStatus == 1 if status.ACLineStatus != 255 else None,
            "battery_percent": battery_percent,
            "battery_saver": bool(status.SystemStatusFlag),
        }
    except Exception:  # noqa: BLE001
        logger.debug("Power status query failed.", exc_info=True)
        return {"plugged_in": None, "battery_percent": None, "battery_saver": None}


def fixed_drives() -> List[Dict[str, object]]:
    """Return usage facts for fixed local drives."""
    drives: List[Dict[str, object]] = []
    try:
        partitions = psutil.disk_partitions(all=False)
    except Exception:  # noqa: BLE001
        return drives

    fixed_types = {"ntfs", "fat32", "exfat", "refs"}
    for part in partitions:
        opts = (getattr(part, "opts", "") or "").lower()
        fstype = (getattr(part, "fstype", "") or "").lower()
        if "cdrom" in opts or "removable" in opts:
            continue
        if "fixed" not in opts and fstype not in fixed_types:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except Exception:  # noqa: BLE001
            continue
        drives.append(
            {
                "mountpoint": part.mountpoint,
                "total": int(usage.total),
                "used": int(usage.used),
                "free": int(usage.free),
                "percent": float(usage.percent),
            }
        )
    return drives
