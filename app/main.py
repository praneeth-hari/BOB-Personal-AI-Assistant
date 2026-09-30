"""
app/main.py
───────────
Entry point for BOB.

Usage:
    python -m app.main

Optionally configure via environment variables or a .env file:
    LLM_PROVIDER=nvidia                   (nvidia | ollama; default: nvidia)
    NVIDIA_API_KEY=...                    (required for the NVIDIA provider)
    NVIDIA_MODEL=openai/gpt-oss-20b
    OLLAMA_MODEL=qwen2.5-coder:7b
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_TIMEOUT=120
    BOB_SYSTEM_PROMPT="You are BOB ..."   (or JARVIS_SYSTEM_PROMPT for backwards compat)
    BOB_LOG_LEVEL=INFO                    (or JARVIS_LOG_LEVEL for backwards compat)
"""

from __future__ import annotations

import logging
import os
import sys


def _configure_logging() -> None:
    level_name = (
        os.environ.get("BOB_LOG_LEVEL")
        or os.environ.get("JARVIS_LOG_LEVEL", "WARNING")
    ).upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_dotenv() -> None:
    """
    Load a .env file from the project root if python-dotenv is available.
    If it is not installed we silently skip – env vars set in the shell still work.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        load_dotenv()
    except ImportError:
        pass


def main() -> None:
    _load_dotenv()
    _configure_logging()

    # Import here (after logging is configured) so that module-level loggers
    # inside app/ are set up correctly.
    from app.interfaces.cli import run_cli
    from app.llm.errors import LLMConfigurationError

    try:
        run_cli()
    except LLMConfigurationError as exc:
        print(f"\n[Configuration error] {exc}\n", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
