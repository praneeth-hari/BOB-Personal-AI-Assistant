"""
app/voice/tts.py
────────────────
Local text-to-speech implementation using Piper.

Key features:
  - 100% offline, local neural speech synthesis via ONNX Runtime.
  - Configurable voice model, speech rate, and audio output via environment variables.
  - Automatic download of missing voice files on first use.
  - Text normalization: cleans markdown syntax (headers, asterisks, links, code blocks)
    so LLM responses are spoken naturally.
  - Audio playback through Windows default audio device via sounddevice.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

from app.voice.base import AudioPlaybackError, SynthesisError, TTSError, TextToSpeech

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "en_US-lessac-medium"
DEFAULT_VOICES_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "piper"
DEFAULT_SAMPLE_RATE = 22050


def clean_text_for_speech(text: str) -> str:
    """
    Clean markdown formatting and symbols from text for natural speech synthesis.

    Removes code blocks, bold/italic markers, markdown links, headers, and bullet points.
    """
    if not text:
        return ""

    # Replace code blocks with brief note
    cleaned = re.sub(r"```[\s\S]*?```", " [code block omitted] ", text)
    # Remove inline code backticks
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Replace markdown links [label](url) with just label
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # Remove bold / italics (*, **, _, __)
    cleaned = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", cleaned)
    # Remove header markers (e.g. "### Heading")
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    # Remove blockquote markers
    cleaned = re.sub(r"^>\s*", "", cleaned, flags=re.MULTILINE)
    # Remove bullet markers (*, -, +) at line start
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    # Remove numbered lists at line start (e.g. "1. ")
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    # Replace markdown table pipes with comma
    cleaned = re.sub(r"\|", " ", cleaned)
    # Normalize multiple whitespace / newlines to single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


class PiperTTS(TextToSpeech):
    """
    Local Text-to-Speech engine powered by Piper.

    Implements the TextToSpeech protocol.
    """

    def __init__(
        self,
        voice_name: Optional[str] = None,
        voices_dir: Optional[Union[str, Path]] = None,
        use_cuda: Optional[bool] = None,
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        lazy_load: bool = True,
    ) -> None:
        self._voice_name = voice_name or os.environ.get("PIPER_VOICE", DEFAULT_VOICE)

        custom_dir = voices_dir or os.environ.get("PIPER_VOICES_DIR")
        self._voices_dir = Path(custom_dir) if custom_dir else DEFAULT_VOICES_DIR

        if use_cuda is not None:
            self._use_cuda = use_cuda
        else:
            self._use_cuda = os.environ.get("PIPER_USE_CUDA", "false").lower() in ("true", "1")

        if speaker_id is not None:
            self._speaker_id = speaker_id
        elif "PIPER_SPEAKER_ID" in os.environ:
            try:
                self._speaker_id = int(os.environ["PIPER_SPEAKER_ID"])
            except ValueError:
                self._speaker_id = None
        else:
            self._speaker_id = None

        self._length_scale = length_scale or float(
            os.environ.get("PIPER_LENGTH_SCALE", "1.0")
        )
        self._noise_scale = noise_scale or float(
            os.environ.get("PIPER_NOISE_SCALE", "0.667")
        )
        self._noise_w = noise_w or float(
            os.environ.get("PIPER_NOISE_W", "0.8")
        )

        self._voice: Optional[Any] = None
        self._sample_rate: int = DEFAULT_SAMPLE_RATE

        if not lazy_load:
            self.load()

    # ── Protocol properties ───────────────────────────────────────────────────

    @property
    def voice_name(self) -> str:
        return self._voice_name

    @property
    def voices_dir(self) -> Path:
        return self._voices_dir

    def is_available(self) -> bool:
        """Return True if model is initialized and ready for synthesis."""
        return self._voice is not None

    # ── Model loading ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load the Piper voice model into memory.
        If model files do not exist locally in voices_dir, download them automatically.
        """
        if self._voice is not None:
            return

        model_path = self._voices_dir / f"{self._voice_name}.onnx"
        config_path = self._voices_dir / f"{self._voice_name}.onnx.json"

        # Check if files need downloading
        if not model_path.exists() or not config_path.exists():
            logger.info("Voice files for '%s' not found. Downloading to '%s'...", self._voice_name, self._voices_dir)
            self._download_voice_model()

        try:
            from piper import PiperVoice

            logger.info("Loading Piper voice model '%s' (CUDA=%s)...", self._voice_name, self._use_cuda)
            self._voice = PiperVoice.load(
                model_path=model_path,
                config_path=config_path,
                use_cuda=self._use_cuda,
            )
            if hasattr(self._voice, "config") and hasattr(self._voice.config, "sample_rate"):
                self._sample_rate = self._voice.config.sample_rate

            logger.info("Piper voice '%s' loaded successfully (sample_rate=%d).", self._voice_name, self._sample_rate)
        except Exception as exc:
            logger.error("Failed to load Piper voice model: %s", exc, exc_info=True)
            raise TTSError(f"Failed to load Piper voice model '{self._voice_name}': {exc}") from exc

    def _download_voice_model(self) -> None:
        """Download voice model and json config via piper.download_voices."""
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        try:
            from piper.download_voices import download_voice
            download_voice(self._voice_name, self._voices_dir)
        except Exception as exc:
            raise TTSError(
                f"Failed to download Piper voice model '{self._voice_name}': {exc}"
            ) from exc

    # ── Speech synthesis ──────────────────────────────────────────────────────

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize text into a single float32 NumPy audio array.

        Parameters:
            text: The text to speak.

        Returns:
            Tuple of (1D float32 audio array, sample_rate).
        """
        if self._voice is None:
            self.load()

        assert self._voice is not None

        clean_text = clean_text_for_speech(text)
        if not clean_text:
            return np.array([], dtype=np.float32), self._sample_rate

        try:
            from piper.config import SynthesisConfig

            syn_config = SynthesisConfig(
                speaker_id=self._speaker_id,
                length_scale=self._length_scale,
                noise_scale=self._noise_scale,
                noise_w_scale=self._noise_w,
            )

            chunks = list(self._voice.synthesize(clean_text, syn_config=syn_config))
            if not chunks:
                return np.array([], dtype=np.float32), self._sample_rate

            sample_rate = chunks[0].sample_rate
            audio_arrays = [chunk.audio_float_array for chunk in chunks if len(chunk.audio_float_array) > 0]

            if not audio_arrays:
                return np.array([], dtype=np.float32), sample_rate

            full_audio = np.concatenate(audio_arrays, axis=0).astype(np.float32)
            return full_audio, sample_rate
        except Exception as exc:
            logger.error("Piper synthesis error: %s", exc, exc_info=True)
            raise SynthesisError(f"Speech synthesis failed: {exc}") from exc

    # ── Audio playback ────────────────────────────────────────────────────────

    def play_audio(self, audio: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
        """
        Play raw float32 audio array through the default Windows speaker via sounddevice.

        Parameters:
            audio: 1D float32 array in [-1.0, 1.0].
            sample_rate: Audio sampling frequency in Hz.
            blocking: If True, blocks until audio finishes playing.
        """
        if len(audio) == 0:
            return

        try:
            import sounddevice as sd
            sd.play(audio, samplerate=sample_rate)
            if blocking:
                sd.wait()
        except Exception as exc:
            logger.error("Audio playback error: %s", exc, exc_info=True)
            raise AudioPlaybackError(f"Audio playback failed: {exc}") from exc

    # ── High-level speak method ───────────────────────────────────────────────

    def speak(self, text: str, blocking: bool = True) -> None:
        """
        Convenience method: synthesize text and immediately play it through speakers.

        Parameters:
            text: Text to speak aloud.
            blocking: If True, wait until playback finishes before returning.
        """
        audio, sample_rate = self.synthesize(text)
        if len(audio) > 0:
            self.play_audio(audio, sample_rate, blocking=blocking)
