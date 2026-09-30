"""
app/voice/base.py
─────────────────
Base interfaces and exception classes for BOB voice processing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
import numpy as np


# ─── Exceptions ───────────────────────────────────────────────────────────────

class VoiceError(Exception):
    """Base exception for all voice-related errors."""


class AudioRecordingError(VoiceError):
    """Raised when audio capture from the microphone fails (e.g. no device, permission denied)."""


class NoSpeechDetectedError(VoiceError):
    """Raised when no speech was detected in the recorded audio (only silence/ambient noise)."""


class TranscriptionError(VoiceError):
    """Raised when speech-to-text transcription fails."""


class TTSError(VoiceError):
    """Base exception for all text-to-speech errors."""


class SynthesisError(TTSError):
    """Raised when voice audio synthesis fails."""


class AudioPlaybackError(TTSError):
    """Raised when audio playback through output speakers fails."""


# ─── Protocols / Interfaces ───────────────────────────────────────────────────

@runtime_checkable
class SpeechToText(Protocol):
    """
    Protocol for local speech-to-text engines.

    Any STT engine (faster-whisper, whisper.cpp, etc.) can be plugged into BOB
    by adhering to this contract.
    """

    @property
    def model_name(self) -> str:
        """The identifier of the loaded STT model (e.g. 'base', 'tiny')."""
        ...

    @property
    def device(self) -> str:
        """The device currently in use ('cuda' or 'cpu')."""
        ...

    @property
    def compute_type(self) -> str:
        """The compute precision in use ('float16', 'int8', etc.)."""
        ...

    def is_available(self) -> bool:
        """Return True if the engine is initialized and ready for transcription."""
        ...

    def transcribe(self, audio: np.ndarray | str) -> str:
        """
        Transcribe audio to text.

        Parameters:
            audio: 1D float32 numpy array (sampled at 16kHz) or file path to audio.

        Returns:
            The transcribed text string.

        Raises:
            TranscriptionError: If transcription fails unexpectedly.
        """
        ...


@runtime_checkable
class TextToSpeech(Protocol):
    """
    Protocol for local text-to-speech engines.

    Any TTS engine (Piper, Coqui, pyttsx3, etc.) can be plugged into BOB
    by adhering to this contract.
    """

    @property
    def voice_name(self) -> str:
        """The identifier of the voice model in use."""
        ...

    def is_available(self) -> bool:
        """Return True if the TTS engine is loaded and ready to synthesize."""
        ...

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize text into raw audio samples.

        Parameters:
            text: The text string to speak.

        Returns:
            A tuple of (1D float32 audio array, sample_rate).

        Raises:
            SynthesisError: If synthesis fails.
        """
        ...

    def speak(self, text: str, blocking: bool = True) -> None:
        """
        Synthesize text and play it through the default audio output.

        Parameters:
            text: The text to speak aloud.
            blocking: If True, wait until playback finishes before returning.

        Raises:
            TTSError: If synthesis or playback fails.
        """
        ...
