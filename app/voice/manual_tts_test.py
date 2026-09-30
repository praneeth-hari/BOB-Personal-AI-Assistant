"""
app/voice/manual_tts_test.py
────────────────────────────
Manual Text-to-Speech (TTS) verification tool for BOB.

Tests the pipeline:
    Text → PiperTTS (synthesis) → sounddevice (playback) → Speaker

Usage:
    # Interactive loop:
    python -m app.voice.manual_tts_test

    # One-shot test:
    python -m app.voice.manual_tts_test --text "Hello, I am BOB. Speech synthesis is working."
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.voice.base import AudioPlaybackError, SynthesisError, TTSError
from app.voice.tts import PiperTTS

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"


def _coloured(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


def run_manual_tts_test(text: str | None = None) -> None:
    """Run interactive or one-shot TTS verification."""
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
    print(_coloured("║      BOB Phase 4: Text-to-Speech (TTS) Manual Test       ║", CYAN, BOLD))
    print(_coloured("╚══════════════════════════════════════════════════════════╝", CYAN, BOLD))
    print(_coloured("  Testing: Text → Piper Neural TTS → sounddevice → Speakers\n", DIM))

    # Initialize Piper
    print(_coloured("1. Initializing Local Piper TTS Engine...", BOLD))
    t0 = time.perf_counter()
    tts = PiperTTS()
    try:
        tts.load()
    except TTSError as exc:
        print(_coloured(f"  [X] Failed to load Piper TTS: {exc}", RED))
        return

    init_time = time.perf_counter() - t0
    print(_coloured("  TTS Engine Ready:", GREEN))
    print(f"    Voice Model : {tts.voice_name}")
    print(f"    Voices Dir  : {tts.voices_dir}")
    print(f"    Sample Rate : {tts._sample_rate} Hz")
    print(f"    Load Time   : {init_time:.2f}s\n")

    def _speak_and_report(prompt_text: str) -> None:
        print(_coloured(f'  Synthesizing: "{prompt_text}"...', CYAN))
        t_synth = time.perf_counter()
        try:
            audio, sample_rate = tts.synthesize(prompt_text)
            elapsed_synth = time.perf_counter() - t_synth
            audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

            print(_coloured(f"  [✓] Synthesized in {elapsed_synth:.2f}s ({audio_duration:.2f}s of audio)", GREEN))
            print(_coloured("  [▶] Playing audio through default speaker...", YELLOW))

            t_play = time.perf_counter()
            tts.play_audio(audio, sample_rate, blocking=True)
            elapsed_play = time.perf_counter() - t_play
            print(_coloured(f"  [✓] Playback complete ({elapsed_play:.2f}s)\n", GREEN))
        except (SynthesisError, AudioPlaybackError) as exc:
            print(_coloured(f"  [X] TTS Error: {exc}\n", RED))

    # One-shot mode
    if text:
        _speak_and_report(text)
        return

    # Interactive loop
    print(_coloured("2. Interactive TTS Test Loop:", BOLD))
    print(_coloured("  Type any sentence and press [Enter] to hear it.", DIM))
    print(_coloured("  Type 'exit' / 'q' to quit.\n", DIM))

    # Speak initial greeting
    _speak_and_report("Hello, I am BOB. Local speech synthesis is online.")

    while True:
        try:
            user_input = input(_coloured("Speak › ", GREEN, BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print(_coloured("\nExiting TTS manual test.", YELLOW))
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print(_coloured("\nExiting TTS manual test.", YELLOW))
            break

        _speak_and_report(user_input)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual Piper TTS verification for BOB.")
    parser.add_argument("--text", type=str, default=None, help="Optional text string to synthesize and speak once.")
    args = parser.parse_args()
    run_manual_tts_test(text=args.text)


if __name__ == "__main__":
    main()
