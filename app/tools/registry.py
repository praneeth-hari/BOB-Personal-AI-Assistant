"""
app/tools/registry.py
─────────────────────
BOB tool registry — rewritten for V2 tool system.

The registry stores Tool descriptors and can:
  • register(tool)          – add a Tool
  • get(name)               – retrieve by name (or None)
  • list_tools()            – names of all registered tools
  • get_tool_schemas()      – LLM-ready schema list (OpenAI/Ollama format)
  • execute(name, args)     – run a tool safely, always returns a dict

Design notes:
  • Completely independent of Ollama — any LLM backend can use the schemas.
  • execute() never raises; errors are returned as ToolResult dicts.
  • No eval(), exec(), or shell execution.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.tools.tool import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for BOB tools.

    Usage::

        registry = ToolRegistry()
        registry.register(SYSTEM_INFO_TOOL)
        result = registry.execute("get_system_info", {})
        schemas = registry.get_tool_schemas()   # pass to LLM
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        """Register a Tool descriptor.  Overwrites if the name already exists."""
        if tool.name in self._tools:
            logger.warning("Tool %r already registered; overwriting.", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %r", tool.name)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[Tool]:
        """Return the Tool with *name*, or None if not registered."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    # ── LLM schema export ─────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Return all tool definitions in OpenAI / Ollama function-calling format.

        Pass this list directly to the LLM ``tools`` parameter::

            client.chat_with_tools(messages, registry.get_tool_schemas())
        """
        return [tool.to_llm_schema() for tool in self._tools.values()]

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the named tool with *arguments*.

        Always returns a plain dict (``ToolResult.to_dict()``).  Never raises:
        unknown tools and bad arguments are reported as error results so the
        LLM can communicate the problem to the user naturally.

        :param name:      Tool name as provided by the LLM.
        :param arguments: Keyword arguments dict from the LLM tool call.
        :returns:         ``{"success": bool, "data": …, "error": …}``
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("LLM requested unknown tool: %r", name)
            return ToolResult(
                success=False,
                error=f"Unknown tool {name!r}. Available: {self.list_tools()}",
            ).to_dict()

        logger.debug("Executing tool %r with args: %r", name, arguments)
        try:
            result = tool.fn(**arguments)
            # Ensure we always return a ToolResult dict
            if isinstance(result, ToolResult):
                return result.to_dict()
            return ToolResult(success=True, data=result).to_dict()

        except TypeError as exc:
            # Wrong keyword arguments supplied by the LLM
            logger.warning("Tool %r received invalid arguments: %s", name, exc)
            return ToolResult(
                success=False,
                error=f"Invalid arguments for tool {name!r}: {exc}",
            ).to_dict()

        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %r raised an unexpected exception.", name)
            return ToolResult(success=False, error=str(exc)).to_dict()

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ToolRegistry(tools={self.list_tools()!r})"


# ─── Pre-built registry with all V2 tools ─────────────────────────────────────

def build_default_registry() -> ToolRegistry:
    """
    Construct and return a ToolRegistry pre-loaded with all available tools.

    Import and call this in the entry point rather than at module load time
    so that tool modules (and their imports like psutil) are only loaded when
    the registry is actually needed.
    """
    from app.tools.system_info import SYSTEM_INFO_TOOL
    from app.tools.open_application import OPEN_APPLICATION_TOOL
    from app.tools.search_files import SEARCH_FILES_TOOL
    from app.tools.close_application import CLOSE_APPLICATION_TOOL
    from app.tools.search_computer import SEARCH_COMPUTER_TOOL
    from app.tools.get_battery_status import BATTERY_STATUS_TOOL
    from app.tools.get_running_applications import RUNNING_APPLICATIONS_TOOL
    from app.tools.audio_volume import AUDIO_VOLUME_TOOL
    from app.tools.screen_brightness import SCREEN_BRIGHTNESS_TOOL
    from app.tools.window_control import WINDOW_CONTROL_TOOL
    from app.tools.web_search import WEB_SEARCH_TOOL
    from app.tools.diagnose_system import DIAGNOSE_SYSTEM_TOOL
    from app.tools.propose_cleanup import PROPOSE_CLEANUP_TOOL

    reg = ToolRegistry()
    reg.register(SYSTEM_INFO_TOOL)
    reg.register(OPEN_APPLICATION_TOOL)
    reg.register(SEARCH_FILES_TOOL)
    reg.register(CLOSE_APPLICATION_TOOL)
    reg.register(SEARCH_COMPUTER_TOOL)
    reg.register(BATTERY_STATUS_TOOL)
    reg.register(RUNNING_APPLICATIONS_TOOL)
    reg.register(AUDIO_VOLUME_TOOL)
    reg.register(SCREEN_BRIGHTNESS_TOOL)
    reg.register(WINDOW_CONTROL_TOOL)
    reg.register(WEB_SEARCH_TOOL)
    reg.register(DIAGNOSE_SYSTEM_TOOL)
    reg.register(PROPOSE_CLEANUP_TOOL)
    logger.debug("Default tool registry built with %d tools.", len(reg))
    return reg


