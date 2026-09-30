"""
app/interfaces/cli.py
─────────────────────
Interactive command-line interface for BOB.

This module is *only* responsible for I/O:
  * Printing the banner and prompt.
  • Reading user input.
  * Forwarding to JarvisAssistant and displaying the reply.
  * Handling exit commands and Ctrl-C gracefully.

All assistant logic lives in app/core/assistant.py.

Note on Windows encoding:
  Windows terminals often default to cp1252. run_cli() reconfigures
  sys.stdout/stderr to UTF-8 (errors='replace') so Unicode box-drawing
  characters in the banner render correctly.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Optional

from app.core.assistant import JarvisAssistant
from app.llm.nvidia_client import NvidiaConnectionError, NvidiaResponseError
from app.llm.ollama_client import OllamaClient, OllamaConnectionError, OllamaResponseError

logger = logging.getLogger(__name__)

# ─── ANSI colour helpers (Windows 10+ and most terminals support these) ───────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BLUE   = "\033[34m"


def _coloured(text: str, *codes: str) -> str:
    """Wrap text with ANSI codes, auto-reset at the end."""
    return "".join(codes) + text + RESET


# ─── Banner ───────────────────────────────────────────────────────────────────

_BANNER = f"""
{_coloured('╔══════════════════════════════════════╗', CYAN, BOLD)}
{_coloured('║', CYAN, BOLD)}  {_coloured('B  O  B', CYAN, BOLD)}  {_coloured('·', DIM)}  Local AI Assistant  {_coloured('║', CYAN, BOLD)}
{_coloured('╚══════════════════════════════════════╝', CYAN, BOLD)}
{_coloured('  Type your message, or "exit" / "quit" to leave.', DIM)}
{_coloured('  "voice" / "/voice"  : continuous voice chat (say "stop listening" to end)', DIM)}
{_coloured('  "/voice-reply"      : toggle speaking replies for typed text', DIM)}
{_coloured('  "/tts <text>"       : speak custom text aloud', DIM)}
{_coloured('  "reset" / "clear"   : clears conversation history', DIM)}
"""

_EXIT_COMMANDS = {"exit", "quit", "q", ":q"}
_RESET_COMMANDS = {"reset", "clear"}
_VOICE_COMMANDS = {"voice", "/voice", ":v"}
_VOICE_REPLY_COMMANDS = {"voice-reply", "/voice-reply", ":vr"}

# Spoken phrases that end continuous voice mode (never sent to the LLM).
_VOICE_EXIT_PHRASES = (
    "stop listening",
    "exit voice mode",
    "stop voice",
    "quit voice",
    "go back to text",
)
# Allow a couple of filler words around the phrase ("okay, stop listening please").
_VOICE_EXIT_EXTRA_WORDS = 2
_MAX_CONSECUTIVE_VOICE_ERRORS = 3


def _normalise_speech(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split())


def _is_voice_exit_phrase(text: str) -> bool:
    """True if a transcript is a (short) request to leave voice mode."""
    normalised = _normalise_speech(text)
    if not normalised:
        return False
    words = normalised.split()
    padded = f" {normalised} "
    for phrase in _VOICE_EXIT_PHRASES:
        if f" {phrase} " in padded and len(words) <= len(phrase.split()) + _VOICE_EXIT_EXTRA_WORDS:
            return True
    return False


# ─── Session ──────────────────────────────────────────────────────────────────

class CLISession:
    """
    Manages a single interactive CLI session.

    Separating this into a class makes it testable and easier to extend later
    (e.g. adding readline history, multi-line input, markdown rendering, …).
    """

    def __init__(
        self,
        assistant: JarvisAssistant,
        stt: Optional[Any] = None,
        recorder: Optional[Any] = None,
        tts: Optional[Any] = None,
        voice_reply_enabled: Optional[bool] = None,
    ) -> None:
        self._assistant = assistant
        self._stt = stt
        self._recorder = recorder
        self._tts = tts
        if voice_reply_enabled is not None:
            self._voice_reply_enabled = voice_reply_enabled
        else:
            self._voice_reply_enabled = (
                os.environ.get("BOB_VOICE_REPLY")
                or os.environ.get("JARVIS_VOICE_REPLY", "false")
            ).lower() in ("true", "1")

    def run(self) -> None:
        """Start the blocking interactive read-eval-print loop."""
        self._print_banner()
        self._print_model_info()

        while True:
            try:
                user_input = self._read_input()
            except (EOFError, KeyboardInterrupt):
                self._print_goodbye()
                break

            if user_input is None:
                # Empty line – just loop again.
                continue

            cmd = user_input.lower().strip()

            if cmd in _EXIT_COMMANDS:
                self._print_goodbye()
                break

            if cmd in _RESET_COMMANDS:
                self._assistant.reset()
                print(_coloured("  ↺  Conversation history cleared.\n", YELLOW))
                continue

            if cmd in _VOICE_REPLY_COMMANDS:
                self._voice_reply_enabled = not self._voice_reply_enabled
                status = "ENABLED" if self._voice_reply_enabled else "DISABLED"
                print(_coloured(f"  Voice reply output: {status}\n", YELLOW, BOLD))
                continue

            if cmd.startswith("/tts ") or cmd.startswith("/speak "):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    self._speak_reply(parts[1].strip())
                continue

            if cmd in _VOICE_COMMANDS:
                self._handle_voice_input()
                continue

            self._call_assistant(user_input)

    def _handle_voice_input(self) -> None:
        """
        Continuous voice conversation mode.

        Repeatedly: listen → transcribe → JarvisAssistant → speak reply,
        until the user says an exit phrase (e.g. "stop listening") or presses
        Ctrl+C.  Uses the same JarvisAssistant instance (and history) as text mode.
        """
        from app.voice.audio_recorder import AudioRecorder
        from app.voice.stt import WhisperSTT

        try:
            if self._stt is None:
                print(_coloured("  Loading Whisper STT model...", DIM))
                self._stt = WhisperSTT()
                self._stt.load()

            if self._recorder is None:
                self._recorder = AudioRecorder()
        except KeyboardInterrupt:
            print(_coloured("\n  Voice mode cancelled. Back to text mode.\n", YELLOW))
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialise voice input.")
            print(_coloured(f"  [Voice Error] Could not start voice mode: {exc}\n", RED))
            return

        print(_coloured(
            '  Voice mode ON. Say "stop listening" (or press Ctrl+C) to return to text.\n',
            CYAN, BOLD,
        ))

        consecutive_errors = 0
        while True:
            try:
                result = self._voice_turn()
            except KeyboardInterrupt:
                print(_coloured("\n  Voice mode interrupted.", YELLOW))
                break

            if result == "exit":
                break
            if result == "error":
                consecutive_errors += 1
                if consecutive_errors >= _MAX_CONSECUTIVE_VOICE_ERRORS:
                    print(_coloured(
                        "  [Voice Error] Too many consecutive voice errors; leaving voice mode.",
                        RED,
                    ))
                    break
            else:
                consecutive_errors = 0

        print(_coloured("  Voice mode OFF. Back to text mode.\n", YELLOW, BOLD))

    def _voice_turn(self) -> str:
        """
        Run one listen → transcribe → reply cycle.

        Returns "exit" when an exit phrase was spoken, "error" on a recording /
        transcription failure, and "ok" otherwise (including silence).
        KeyboardInterrupt is propagated to the caller.
        """
        from app.voice.base import AudioRecordingError, NoSpeechDetectedError, TranscriptionError

        print(_coloured("  [•] Listening... Speak into your microphone.", CYAN))
        try:
            audio = self._recorder.record_phrase(
                on_speech_start=lambda: print(_coloured("  [🎙️] Speech detected! Recording...", GREEN)),
                on_speech_end=lambda: print(_coloured("  [✓] Silence detected. Transcribing...", YELLOW)),
            )
            transcription = self._stt.transcribe(audio)
        except NoSpeechDetectedError as exc:
            print(_coloured(f"  [-] {exc}", DIM))
            return "ok"
        except (AudioRecordingError, TranscriptionError) as exc:
            print(_coloured(f"  [Voice Error] {exc}\n", RED))
            return "error"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during voice capture.")
            print(_coloured(f"  [Voice Error] {exc}\n", RED))
            return "error"

        if not transcription or not transcription.strip():
            print(_coloured("  [-] No speech transcribed.", DIM))
            return "ok"

        print(_coloured(f"\n  You (voice) › {transcription}", GREEN, BOLD))

        if _is_voice_exit_phrase(transcription):
            return "exit"

        # Voice input always speaks the reply aloud
        self._call_assistant(transcription, speak=True)
        return "ok"

    def _speak_reply(self, text: str) -> None:
        """Synthesize and play speech via Piper TTS."""
        if not text:
            return
        try:
            from app.voice.tts import PiperTTS
            if self._tts is None:
                self._tts = PiperTTS()
                self._tts.load()
            self._tts.speak(text, blocking=True)
        except Exception as exc:
            logger.warning("TTS speech output failed: %s", exc)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        print(_BANNER)

    def _print_model_info(self) -> None:
        """Show which model, host, and tools are active."""
        llm = self._assistant._llm  # type: ignore[attr-defined]
        reg = self._assistant.tool_registry
        tools_str = ", ".join(reg.list_tools()) if reg else "none"
        if hasattr(llm, "model") and hasattr(llm, "host"):
            print(
                _coloured(
                    f"  Model : {llm.model}\n"
                    f"  Host  : {llm.host}\n"
                    f"  Tools : {tools_str}\n",
                    DIM,
                )
            )

    def _read_input(self) -> Optional[str]:
        """Read one line of input.  Returns None for blank lines."""
        raw = input(_coloured("You › ", GREEN, BOLD))
        stripped = raw.strip()
        return stripped if stripped else None

    def _call_assistant(self, message: str, speak: bool = False) -> Optional[str]:
        """Send message to assistant, print reply, and optionally speak aloud."""
        print(_coloured("\nBOB › ", CYAN, BOLD), end="", flush=True)
        reply = None
        try:
            reply = self._assistant.chat(message)
            print(reply)
        except OllamaConnectionError as exc:
            print(_coloured(f"\n[Connection error] {exc}", RED))
            print(_coloured("  Is Ollama running? Try: ollama serve", DIM))
        except OllamaResponseError as exc:
            print(_coloured(f"\n[Response error] {exc}", RED))
        except NvidiaConnectionError as exc:
            print(_coloured(f"\n[Connection error] {exc}", RED))
        except NvidiaResponseError as exc:
            print(_coloured(f"\n[Response error] {exc}", RED))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during chat.")
            print(_coloured(f"\n[Error] {exc}", RED))
        print()  # blank line after every exchange

        if reply and (speak or self._voice_reply_enabled):
            self._speak_reply(reply)

        return reply

    def _print_goodbye(self) -> None:
        print(_coloured("\n  Goodbye. BOB offline.\n", CYAN, BOLD))


# ─── Public entry point ───────────────────────────────────────────────────────

def _configure_stdout_utf8() -> None:
    """
    Reconfigure sys.stdout and sys.stderr to use UTF-8 encoding.

    On Windows the default console/pipe encoding is often cp1252, which
    cannot encode Unicode box-drawing characters used in the banner.
    This is a process-local change and does not affect other programs.
    """
    import io
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        else:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        else:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace"
            )
    except Exception:  # noqa: BLE001
        pass  # Non-fatal; worst case the banner shows replacement chars


def run_cli() -> None:
    """
    Build the BOB stack and start the CLI session.
    This is the only place where the LLM client and ToolRegistry are wired up.
    The LLM backend is chosen by LLM_PROVIDER (see app/llm/factory.py).
    """
    # Force UTF-8 output so Unicode banner chars work on Windows cp1252 terminals.
    _configure_stdout_utf8()

    # Enable ANSI on Windows (no-op on modern Windows 10+ with VT enabled)
    if sys.platform == "win32":
        import ctypes
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:  # noqa: BLE001
            pass  # Non-fatal – colours just won't render

    from app.core.laya_router import build_router_from_env
    from app.llm.factory import create_llm_client
    from app.llm.nvidia_client import NvidiaClient as _NvidiaClient
    from app.tools.registry import build_default_registry

    llm_client   = create_llm_client()
    tool_registry = build_default_registry()

    # Optional Laya request router: loaded once here, then reused for every message.
    router = build_router_from_env()
    if router is not None:
        print(_coloured("  Loading Laya request router...", DIM), flush=True)
        if router.load():
            print(_coloured(
                f"  Laya router ready on {router.device} ({router.load_seconds:.1f}s).", DIM
            ))
        else:
            print(_coloured(
                f"  Laya router unavailable ({router.failure_reason}); continuing without routing.",
                YELLOW,
            ))
            router = None

    # When the primary provider is NVIDIA, wire up OllamaClient as a silent
    # fallback for transient connection failures and rate-limit errors.
    fallback_llm = None
    if isinstance(llm_client, _NvidiaClient):
        fallback_llm = OllamaClient()

    assistant    = JarvisAssistant(
        llm_client=llm_client, tool_registry=tool_registry, router=router,
        fallback_llm=fallback_llm,
    )
    session      = CLISession(assistant=assistant)
    session.run()
