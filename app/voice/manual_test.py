"""
app/voice/manual_test.py
────────────────────────
Manual Speech-to-Text (STT) verification tool for BOB.

Tests the pipeline:
    Microphone → AudioRecorder → WhisperSTT → Text

IMPORTANT:
    This tool intentionally DOES NOT send anything to Ollama or the LLM.
    It isolates and verifies local audio capture and speech recognition.

Usage:
    python -m app.voice.manual_test
"""

from __future__ import annotations

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.voice.audio_recorder import AudioRecorder
from app.voice.base import AudioRecordingError, NoSpeechDetectedError, TranscriptionError
from app.voice.stt import WhisperSTT

# ─── Terminal styling ─────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"


def _coloured(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


def run_manual_test() -> None:
    """Run interactive microphone -> faster-whisper -> text verification."""
    # Ensure UTF-8 console output on Windows
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)  # type: ignore[attr-defined]
        except Exception:
            pass

    print()
    print(_coloured("╔══════════════════════════════════════════════════════════╗", CYAN, BOLD))
    print(_coloured("║      BOB Phase 3: Speech-to-Text (STT) Manual Test        ║", CYAN, BOLD))
    print(_coloured("╚══════════════════════════════════════════════════════════╝", CYAN, BOLD))
    print(_coloured("  Testing: Microphone → AudioRecorder → WhisperSTT → Text", DIM))
    print(_coloured("  Note: Audio and transcriptions are NOT sent to Ollama.\n", DIM))

    # 1. Inspect audio hardware
    devices = AudioRecorder.list_input_devices()
    print(_coloured("1. Microphone Status:", BOLD))
    if not devices:
        print(_coloured("  [!] Warning: No audio input devices found.", RED))
    else:
        print(_coloured(f"  Found {len(devices)} input device(s):", GREEN))
        for dev in devices[:5]:
            print(f"    - [{dev['index']}] {dev['name']} ({dev['channels']} ch, {int(dev['default_samplerate'])} Hz)")
        if len(devices) > 5:
            print(f"    ... and {len(devices) - 5} more.")

    # 2. Initialize Whisper STT engine
    print()
    print(_coloured("2. Initializing Local Whisper STT Engine...", BOLD))
    stt = WhisperSTT()
    stt.load()

    print(_coloured("  STT Engine Ready:", GREEN))
    print(f"    Model        : {stt.model_name}")
    print(f"    Active Device: {stt.device}")
    print(f"    Compute Type : {stt.compute_type}")
    print(f"    Language     : {stt.language or 'auto-detect'}")

    recorder = AudioRecorder()

    print()
    print(_coloured("3. Interactive Voice Test Loop:", BOLD))
    print(_coloured("  Press [Enter] to start speaking, or type 'exit' / 'q' to quit.", DIM))

    while True:
        try:
            prompt = input(_coloured("\nReady › Press [Enter] to speak: ", GREEN, BOLD))
        except (EOFError, KeyboardInterrupt):
            print(_coloured("\nExiting manual test.", YELLOW))
            break

        if prompt.strip().lower() in ("exit", "quit", "q"):
            print(_coloured("\nExiting manual test.", YELLOW))
            break

        print(_coloured("  [•] Listening... Speak into your microphone now.", CYAN))

        def on_speech_start() -> None:
            print(_coloured("  [🎙️] Speech detected! Recording...", GREEN))

        def on_speech_end() -> None:
            print(_coloured("  [✓] Silence detected. Transcribing with faster-whisper...", YELLOW))

        try:
            audio = recorder.record_phrase(
                on_speech_start=on_speech_start,
                on_speech_end=on_speech_end,
            )
        except NoSpeechDetectedError as exc:
            print(_coloured(f"  [-] {exc}", YELLOW))
            continue
        except AudioRecordingError as exc:
            print(_coloured(f"  [X] Audio recording error: {exc}", RED))
            continue
        except KeyboardInterrupt:
            print(_coloured("\n  Recording cancelled by user.", YELLOW))
            continue

        # Transcribe
        t0 = time.perf_counter()
        try:
            transcription = stt.transcribe(audio)
            elapsed = time.perf_counter() - t0
        except TranscriptionError as exc:
            print(_coloured(f"  [X] Transcription error: {exc}", RED))
            continue

        print()
        print(_coloured("  ┌────────────────────────────────────────────────────────┐", CYAN))
        print(_coloured("  │ Transcribed Output:                                    │", CYAN))
        print(f"  │ {_coloured(transcription or '(silence / unvoiced)', BOLD):<54} │")
        print(_coloured("  └────────────────────────────────────────────────────────┘", CYAN))
        print(_coloured(f"  (Processed in {elapsed:.2f}s | Audio duration: {len(audio)/recorder.sample_rate:.2f}s)", DIM))


if __name__ == "__main__":
    run_manual_test()
