"""
app/tools/search_computer.py
────────────────────────────
BOB tool: search_computer

Safely searches and counts files across user-level locations on the PC.
Supports extension filtering, filename matching, count-only mode, and location scopes.

SAFETY & SECURITY MODEL
───────────────────────
• Confined strictly to user data locations (Documents, Downloads, Desktop,
  Pictures, Videos, Music, project workspace, and user home), plus the whole
  D:\\ data drive when it exists (location='d_drive', and included in 'all').
• Overlapping roots are collapsed (e.g. D:\\jarvis is not scanned again when
  D:\\ is already being scanned).
• Explicitly BLOCKS system directories:
    - C:\\Windows (and any OS system paths)
    - C:\\Program Files and C:\\Program Files (x86)
    - C:\\ProgramData
    - System Volume Information, $Recycle.Bin, Recovery
• Path traversal attacks (e.g. "..") attempting to escape allowed user roots
  are detected and rejected.
• System cache / package directories (AppData, node_modules, .git, __pycache__,
  .venv, venv) are skipped during broad scans to avoid infinite or expensive walks.
• Symlinks and junctions are NOT followed (followlinks=False, plus an explicit
  reparse-tag check, since os.walk on Windows still descends into junctions).
• At a drive root, OS folders (Windows, Program Files, Recovery, ...) are pruned.
• Read-only: no files are opened, modified, moved, executed, or deleted.
• Maximum recursion depth is bounded.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_RESULT_LIMIT: int = 20
MAX_RESULT_LIMIT: int = 100
MAX_RECURSION_DEPTH: int = 12

# System directories that must NEVER be scanned
BLOCKED_PATH_PATTERNS: Set[str] = {
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "system volume information",
    "$recycle.bin",
    "recovery",
    "msocache",
}

# Directories skipped during traversal to keep scans fast and safe
SKIP_DIR_NAMES: Set[str] = {
    "appdata",
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".cache",
    ".tox",
    ".pytest_cache",
    "system volume information",
    "$recycle.bin",
}

# Whole-drive root that may be searched (location='d_drive' / 'all')
D_DRIVE_ROOT: Path = Path("D:\\")

# OS-managed folders pruned when walking directly under a drive root
DRIVE_ROOT_SKIP_NAMES: Set[str] = BLOCKED_PATH_PATTERNS | {
    "windows.old",
    "perflogs",
    "config.msi",
    "boot",
    "efi",
    "onedrivetemp",
}

# Friendly names that map to the whole D:\ drive
D_DRIVE_ALIASES: Set[str] = {
    "d_drive", "d-drive", "d drive", "ddrive", "drive d", "d", "d:", "d:\\", "d:/",
}

# Reparse tags for symlinks and junctions (mount points)
_LINK_REPARSE_TAGS: Set[int] = {
    getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C),
    getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003),
}


# ─── Allowed roots & security checks ──────────────────────────────────────────

def get_d_drive_root() -> Optional[Path]:
    """Return the D:\\ drive root if it exists on this machine, else None."""
    try:
        if D_DRIVE_ROOT.exists() and D_DRIVE_ROOT.is_dir():
            return D_DRIVE_ROOT.resolve()
    except (OSError, ValueError):
        pass
    return None


def is_drive_root(path: str) -> bool:
    """Return True if path is a drive root such as D:\\."""
    p = Path(path)
    return p.parent == p


def is_link_or_junction(path: str) -> bool:
    """Return True if path is a symlink or an NTFS junction (never traversed)."""
    try:
        if os.path.islink(path):
            return True
        st = os.lstat(path)
    except OSError:
        return True  # unreadable → treat as unsafe and skip
    return getattr(st, "st_reparse_tag", 0) in _LINK_REPARSE_TAGS


def dedupe_nested_roots(paths: List[Path]) -> List[Path]:
    """
    Drop duplicate roots and roots nested inside another root, preserving order,
    so no directory is scanned twice (e.g. D:\\jarvis when D:\\ is present).
    """
    unique: List[Path] = []
    for p in paths:
        if p not in unique:
            unique.append(p)
    return [
        p for p in unique
        if not any(other != p and other in p.parents for other in unique)
    ]

def get_allowed_search_roots() -> List[Path]:
    """
    Return the list of allowed root directories that may be searched.
    Defaults to the user home directory and the project working directory,
    plus any roots explicitly defined in BOB_SEARCH_ROOTS / JARVIS_SEARCH_ROOTS.
    """
    roots: List[Path] = []

    # User home directory (e.g. C:\Users\<username>)
    try:
        home = Path.home().resolve()
        if home.exists():
            roots.append(home)
    except Exception:  # noqa: BLE001
        pass

    # Current project / workspace directory
    try:
        cwd = Path.cwd().resolve()
        if cwd.exists() and cwd not in roots:
            roots.append(cwd)
    except Exception:  # noqa: BLE001
        pass

    # Whole D:\ data drive, when present
    d_root = get_d_drive_root()
    if d_root is not None and d_root not in roots:
        roots.append(d_root)

    # Custom roots from environment variable
    custom_roots_env = os.environ.get("BOB_SEARCH_ROOTS") or os.environ.get("JARVIS_SEARCH_ROOTS", "")
    if custom_roots_env:
        for raw in custom_roots_env.replace(",", ";").split(";"):
            part = raw.strip()
            if part:
                try:
                    p = Path(part).resolve()
                    if p.exists() and p not in roots:
                        roots.append(p)
                except Exception:  # noqa: BLE001
                    pass

    return roots


def is_path_blocked(path: Path) -> bool:
    """Return True if path falls under a blocked system or OS directory."""
    path_str = str(path).lower()
    for pattern in BLOCKED_PATH_PATTERNS:
        # Check drive-root level system folders or directory components
        if f"\\{pattern}\\" in path_str or path_str.endswith(f"\\{pattern}") or f":\\{pattern}" in path_str:
            return True
    return False


def is_path_within_allowed_roots(path: Path, allowed_roots: List[Path]) -> bool:
    """Ensure path is within at least one allowed search root."""
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False

    for root in allowed_roots:
        try:
            if resolved == root or root in resolved.parents:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


# ─── Location resolution ───────────────────────────────────────────────────────

def resolve_search_locations(location: str = "") -> List[Path]:
    """
    Resolve the requested location into a list of concrete directories to search.

    :param location: Friendly name (e.g. 'documents', 'downloads', 'desktop',
                     'workspace', 'd_drive', 'all') or an allowed absolute path.
    :returns: List of existing directory Paths to search.
    :raises PermissionError: If the location is outside allowed roots or blocked.
    :raises FileNotFoundError: If the specified location does not exist.
    """
    loc_clean = (location or "").strip().lower()
    home = Path.home().resolve()
    allowed_roots = get_allowed_search_roots()

    # Friendly location mapping
    standard_folders = {
        "documents": [home / "Documents", home / "OneDrive" / "Documents"],
        "doc":       [home / "Documents", home / "OneDrive" / "Documents"],
        "docs":      [home / "Documents", home / "OneDrive" / "Documents"],
        "downloads": [home / "Downloads"],
        "download":  [home / "Downloads"],
        "desktop":   [home / "Desktop", home / "OneDrive" / "Desktop"],
        "pictures":  [home / "Pictures", home / "OneDrive" / "Pictures"],
        "photos":    [home / "Pictures", home / "OneDrive" / "Pictures"],
        "videos":    [home / "Videos", home / "OneDrive" / "Videos"],
        "movies":    [home / "Videos", home / "OneDrive" / "Videos"],
        "music":     [home / "Music"],
        "workspace": [Path.cwd().resolve()],
        "project":   [Path.cwd().resolve()],
        "jarvis":    [Path.cwd().resolve()],
        "home":      [home],
        "user":      [home],
    }

    # Case 0: Whole D:\ drive
    if loc_clean in D_DRIVE_ALIASES:
        d_root = get_d_drive_root()
        if d_root is None:
            raise FileNotFoundError("The D:\\ drive does not exist on this PC.")
        return [d_root]

    # Case 1: Broad search across standard user folders (+ D:\ drive)
    if not loc_clean or loc_clean in ("all", "pc", "computer", "my pc", "my computer"):
        candidates: List[Path] = [
            home / "Documents",
            home / "OneDrive" / "Documents",
            home / "Downloads",
            home / "Desktop",
            home / "OneDrive" / "Desktop",
            home / "Pictures",
            home / "OneDrive" / "Pictures",
            home / "Videos",
            home / "OneDrive" / "Videos",
            home / "Music",
            Path.cwd().resolve(),
        ]
        d_root = get_d_drive_root()
        if d_root is not None:
            candidates.append(d_root)
        resolved_dirs: List[Path] = []
        seen: Set[Path] = set()
        for c in candidates:
            try:
                res = c.resolve()
                if res.exists() and res.is_dir() and res not in seen:
                    if not is_path_blocked(res):
                        seen.add(res)
                        resolved_dirs.append(res)
            except (OSError, ValueError):
                continue

        # If none of the subfolders were found, fall back to home directory
        if not resolved_dirs and home.exists():
            resolved_dirs.append(home)
        return dedupe_nested_roots(resolved_dirs)

    # Case 2: Specific friendly location
    if loc_clean in standard_folders:
        candidates = standard_folders[loc_clean]
        resolved_dirs = []
        for c in candidates:
            try:
                res = c.resolve()
                if res.exists() and res.is_dir() and not is_path_blocked(res):
                    resolved_dirs.append(res)
            except (OSError, ValueError):
                continue
        if not resolved_dirs:
            raise FileNotFoundError(f"Folder for location '{location}' does not exist on this PC.")
        return resolved_dirs

    # Case 3: Explicit path provided
    try:
        custom_path = Path(location).resolve()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid path format: {location!r}") from exc

    if is_path_blocked(custom_path):
        raise PermissionError(
            f"Access to system location {location!r} is prohibited. "
            "Search is restricted to user data locations."
        )

    if not is_path_within_allowed_roots(custom_path, allowed_roots):
        raise PermissionError(
            f"Location {location!r} is outside allowed user search roots. "
            "Search is restricted to user directories."
        )

    if not custom_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {custom_path}")

    if not custom_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {custom_path}")

    return [custom_path]


# ─── Filter helpers ────────────────────────────────────────────────────────────

def normalize_extension(ext: Optional[str]) -> str:
    """Normalize file extension to lowercase with leading dot (e.g. '.pdf')."""
    if not ext or not str(ext).strip():
        return ""
    cleaned = str(ext).strip().lstrip("*")
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    return cleaned.lower()


# ─── Implementation ───────────────────────────────────────────────────────────

def _search_computer(
    extension: str = "",
    query: str = "",
    location: str = "",
    count_only: bool = False,
    limit: int = DEFAULT_RESULT_LIMIT,
) -> ToolResult:
    """
    Search or count files across allowed user locations on the PC.

    :param extension:  File extension to match, e.g. '.pdf', '.py', '.csv', '.txt'.
    :param query:      Filename substring or glob pattern (e.g. 'invoice*', 'notes').
    :param location:   Folder scope ('documents', 'downloads', 'desktop', 'workspace', 'all').
    :param count_only: If True, only count matching files without returning file path arrays.
    :param limit:      Maximum number of matching file details returned when count_only is False.
    :returns:          ToolResult with total count and matching file list.
    """
    # Validation: at least extension or query should be provided to prevent scanning everything
    norm_ext = normalize_extension(extension)
    query_clean = str(query or "").strip()

    if not norm_ext and not query_clean:
        return ToolResult(
            success=False,
            error="Please specify at least a file extension (e.g. '.pdf') or a filename query.",
        )

    # Sanitize limit
    try:
        limit_val = int(limit)
        limit_val = max(1, min(limit_val, MAX_RESULT_LIMIT))
    except (ValueError, TypeError):
        limit_val = DEFAULT_RESULT_LIMIT

    # Resolve locations
    try:
        search_dirs = resolve_search_locations(location)
    except (PermissionError, ValueError, FileNotFoundError, NotADirectoryError) as exc:
        return ToolResult(success=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error resolving search location %r", location)
        return ToolResult(success=False, error=f"Location resolution failed: {exc}")

    search_dirs = dedupe_nested_roots(search_dirs)

    use_glob: bool = any(c in query_clean for c in ("*", "?", "["))
    total_count: int = 0
    results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    roots_searched: List[str] = []
    skipped_dirs: int = 0

    logger.info(
        "search_computer: ext=%r query=%r count_only=%s dirs=%s",
        norm_ext, query_clean, count_only, [str(d) for d in search_dirs],
    )

    for search_dir in search_dirs:
        try:
            for root, dirs, files in os.walk(str(search_dir), followlinks=False):
                # Filter directories in-place to avoid hidden, heavy, linked and OS folders
                at_drive_root = is_drive_root(root)
                kept = [
                    d for d in dirs
                    if not d.startswith(".")
                    and not d.startswith("$")
                    and d.lower() not in SKIP_DIR_NAMES
                    and not (at_drive_root and d.lower() in DRIVE_ROOT_SKIP_NAMES)
                    and not is_link_or_junction(os.path.join(root, d))
                ]
                skipped_dirs += len(dirs) - len(kept)
                dirs[:] = kept

                # Check recursion depth to prevent runaway scans
                try:
                    rel_depth = len(Path(root).relative_to(search_dir).parts)
                    if rel_depth >= MAX_RECURSION_DEPTH:
                        dirs[:] = []
                except Exception:  # noqa: BLE001
                    pass

                for filename in files:
                    file_lower = filename.lower()

                    # 1. Extension filter
                    if norm_ext and not file_lower.endswith(norm_ext):
                        continue

                    # 2. Query filter
                    if query_clean:
                        if use_glob:
                            if not fnmatch.fnmatch(file_lower, query_clean.lower()):
                                continue
                        else:
                            if query_clean.lower() not in file_lower:
                                continue

                    total_count += 1

                    # Collect result paths if not count_only and within limit
                    if not count_only and len(results) < limit_val:
                        full_path = os.path.join(root, filename)
                        try:
                            size = os.path.getsize(full_path)
                        except OSError:
                            size = None
                        results.append({
                            "path": full_path,
                            "name": filename,
                            "size_bytes": size,
                        })

            roots_searched.append(str(search_dir))
        except PermissionError as exc:
            warnings.append(f"Permission denied accessing some folders in {search_dir}: {exc}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Error scanning {search_dir}: {exc}")

    limit_reached = (not count_only) and (total_count > limit_val)
    if limit_reached:
        warnings.append(
            f"Returned first {limit_val} files of {total_count} total matches."
        )

    return ToolResult(
        success=True,
        data={
            "extension":          norm_ext,
            "query":              query_clean,
            "locations_searched": [str(d) for d in search_dirs],
            "roots_searched":     roots_searched,
            "excluded_dirs_skipped": skipped_dirs,
            "count":              total_count,
            "count_only":         bool(count_only),
            "results":            results,
            "limit":              limit_val,
            "limit_reached":      limit_reached,
            "warnings":           warnings,
        },
    )


# ─── Tool Descriptor ──────────────────────────────────────────────────────────

SEARCH_COMPUTER_TOOL = Tool(
    name="search_computer",
    description=(
        "Searches or counts files across the computer's user folders (Documents, Downloads, Desktop, etc.) "
        "and the entire D: drive. "
        "Use this tool when the user asks: "
        "- 'How many PDFs are on my PC?' (extension='.pdf', count_only=True, location='all') "
        "- 'How many PDFs are on my D drive?' (extension='.pdf', count_only=True, location='d_drive') "
        "- 'Find Python files on D drive' (extension='.py', count_only=False, location='d_drive') "
        "- 'Find all Python files on my computer' (extension='.py', count_only=False) "
        "- 'How many CSV files do I have?' (extension='.csv', count_only=True) "
        "- 'Show me the PDFs in my Documents folder' (extension='.pdf', location='documents') "
        "Safe and read-only. Operating system and sensitive directories are blocked."
    ),
    parameters={
        "type": "object",
        "properties": {
            "extension": {
                "type": "string",
                "description": (
                    "File extension to search or count, e.g. '.pdf', '.py', '.csv', '.docx', '.txt'. "
                    "Case-insensitive."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Optional filename substring or glob pattern to match (e.g. 'report', 'invoice*', 'test_')."
                ),
            },
            "location": {
                "type": "string",
                "description": (
                    "Optional folder scope: 'documents', 'downloads', 'desktop', 'pictures', 'videos', 'music', "
                    "'workspace', 'd_drive' (the whole D: drive), or 'all' (user folders plus the D: drive; default: 'all')."
                ),
            },
            "count_only": {
                "type": "boolean",
                "description": (
                    "Set to true when the user asks 'how many' or to count files. "
                    "Quickly counts matches without returning a large list of file paths."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of matching file paths to return when count_only is false (default: 20, max: 100)."
                ),
            },
        },
        "required": [],
    },
    fn=_search_computer,
)
