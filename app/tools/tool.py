"""
app/tools/tool.py
─────────────────
Core tool abstraction for BOB.

Every tool is a `Tool` dataclass that carries:
  • name            – unique identifier used by the LLM and registry
  • description     – plain-English description sent to the LLM
  • parameters      – JSON Schema object describing accepted arguments
  • fn              – the Python callable that does the actual work

`ToolResult` is the uniform return type from every tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ─── Result type ─────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """
    Structured result from any tool execution.

    :param success: True if the tool ran without error.
    :param data:    The tool's output (any JSON-serialisable value).
    :param error:   Human-readable error message when success=False.
    """
    success: bool
    data: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Serialise to a JSON string (for tool-result messages)."""
        return json.dumps(self.to_dict(), default=str)


# ─── Tool descriptor ──────────────────────────────────────────────────────────

@dataclass
class Tool:
    """
    Descriptor for a single BOB tool.

    The `parameters` field must be a valid JSON Schema *object* schema, e.g.::

        {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "…"},
            },
            "required": ["directory"],
        }

    This schema is forwarded verbatim to the LLM so it can construct
    valid tool-call arguments.
    """

    name: str
    description: str
    parameters: Dict[str, Any]          # JSON Schema object
    fn: Callable[..., ToolResult]

    def to_llm_schema(self) -> Dict[str, Any]:
        """
        Return the tool definition in OpenAI / Ollama function-calling format::

            {
                "type": "function",
                "function": {
                    "name": "…",
                    "description": "…",
                    "parameters": { … }
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
