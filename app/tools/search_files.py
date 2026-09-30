"""
app/tools/search_files.py
──────────────────────────
BOB tool: search_files

Searches filenames recursively within a given directory.

SAFETY GUARANTEES
──────────────────
• Read-only: no files are opened, modified, moved, executed, or deleted.
• Results are capped at MAX_RESULTS to avoid flooding the LLM context.
• Hidden directories (starting with ".") are skipped.
• os.walk() naturally confines the search to the requested directory tree.
• Inaccessible directories are reported as errors, not silently skipped.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)

MAX_RESULTS: int = 50


# ─── Implementation ───────────────────────────────────────────────────────────

def _resolve_directory(directory: str) -> Optional[Path]:
    """Resolve and validate the search root. Returns None on failure."""
    try:
        return Path(directory).resolve()
    except (OSError, ValueError):
        return None


def _search_files(directory: str, query: str) -> ToolResult:
    """
    Search for files whose names match *query* under *directory*.

    :param directory: Absolute (or relative) path to the search root.
    :param query:     Glob pattern (e.g. ``*.py``) or plain substring
                      (e.g. ``report``).  Matching is case-insensitive for
                      substring mode; fnmatch case rules apply for globs.
    :returns: ToolResult with structured match list.
    """
    if not directory or not directory.strip():
        return ToolResult(success=False, error="directory must not be empty.")
    if not query or not query.strip():
        return ToolResult(success=False, error="query must not be empty.")

    resolved = _resolve_directory(directory.strip())
    if resolved is None:
        return ToolResult(success=False, error=f"Cannot resolve path: {directory!r}")

    if not resolved.exists():
        return ToolResult(
            success=False,
            error=f"Directory does not exist: {resolved}",
        )
    if not resolved.is_dir():
        return ToolResult(
            success=False,
            error=f"Path is not a directory: {resolved}",
        )

    query = query.strip()
    use_glob: bool = any(c in query for c in ("*", "?", "["))

    matches: List[Dict[str, Any]] = []
    warnings: List[str] = []
    limit_reached = False

    try:
        for root, dirs, files in os.walk(str(resolved)):
            # Skip hidden dirs to avoid .git, __pycache__, etc.
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in files:
                if use_glob:
                    matched = fnmatch.fnmatch(filename, query)
                else:
                    matched = query.lower() in filename.lower()

                if matched:
                    full_path = os.path.join(root, filename)
                    try:
                        size = os.path.getsize(full_path)
                    except OSError:
                        size = None
                    matches.append(
                        {"path": full_path, "name": filename, "size_bytes": size}
                    )

                if len(matches) >= MAX_RESULTS:
                    limit_reached = True
                    break

            if limit_reached:
                warnings.append(
                    f"Result limit ({MAX_RESULTS}) reached; "
                    "there may be additional matching files."
                )
                break

    except PermissionError as exc:
        warnings.append(f"Permission denied accessing part of the tree: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_files failed.")
        return ToolResult(success=False, error=str(exc))

    return ToolResult(
        success=True,
        data={
            "directory":   str(resolved),
            "query":       query,
            "match_type":  "glob" if use_glob else "substring",
            "count":       len(matches),
            "results":     matches,
            "warnings":    warnings,
        },
    )


# ─── Tool descriptor ──────────────────────────────────────────────────────────

SEARCH_FILES_TOOL = Tool(
    name="search_files",
    description=(
        "Searches for files recursively within a directory. "
        "Matches by glob pattern (e.g. '*.py', '*.txt') or plain substring. "
        "Read-only: does not open, modify, move, execute, or delete any files. "
        f"Returns up to {MAX_RESULTS} results."
    ),
    parameters={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": (
                    "The directory to search in. "
                    "Use an absolute path, e.g. 'D:\\\\Projects' or 'C:\\\\Users\\\\me\\\\Documents'."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Filename pattern or substring to search for. "
                    "Glob wildcards are supported: '*.py', 'report*.xlsx', '*.md'. "
                    "Plain text matches any filename containing that text."
                ),
            },
        },
        "required": ["directory", "query"],
    },
    fn=_search_files,
)
