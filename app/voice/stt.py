"""
app/voice/stt.py
────────────────
Local speech-to-text implementation using faster-whisper.

Key features:
  - 100% offline, local inference via CTranslate2.
  - Configurable model, device, compute precision, and language via environment variables.
  - Automatic GPU (CUDA) detection with a safe, verified CPU fallback.
  - Lazy loading: model weights are loaded on demand to keep startup instantaneous.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional, Tuple, Union

import numpy as np

from app.voice.base import SpeechToText, TranscriptionError

logger = logging.getLogger(__name__)

# Suppress HuggingFace symlinks warning on Windows if not already set
if "HF_HUB_DISABLE_SYMLINKS_WARNING" not in os.environ:
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Sensible defaults for RTX 3050 (4 GB VRAM) / 16 GB RAM setup
DEFAULT_MODEL = "base"
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE_TYPE = "auto"


class WhisperSTT(SpeechToText):
    """
    Local Speech-to-Text engine powered by faster-whisper.

    Implements the SpeechToText protocol.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        download_root: Optional[str] = None,
        lazy_load: bool = True,
    ) -> None:
        self._model_name = model_name or os.environ.get("WHISPER_MODEL", DEFAULT_MODEL)
        self._requested_device = (
            device or os.environ.get("WHISPER_DEVICE", DEFAULT_DEVICE)
        ).lower()
        self._requested_compute_type = (
            compute_type or os.environ.get("WHISPER_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE)
        ).lower()
        self._language = language or os.environ.get("WHISPER_LANGUAGE")
        self._download_root = download_root or os.environ.get("WHISPER_DOWNLOAD_ROOT")

        self._model: Optional[Any] = None
        self._active_device: Optional[str] = None
        self._active_compute_type: Optional[str] = None

        if not lazy_load:
            self.load()

    # ── Properties adhering to SpeechToText protocol ──────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._active_device or self._requested_device

    @property
    def compute_type(self) -> str:
        return self._active_compute_type or self._requested_compute_type

    @property
    def language(self) -> Optional[str]:
        return self._language

    def is_available(self) -> bool:
        """Return True if model is loaded and ready."""
        return self._model is not None

    # ── Model loading with CUDA -> CPU fallback ───────────────────────────────

    def load(self) -> None:
        """Load the faster-whisper model, attempting GPU first if applicable."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel
        import ctranslate2

        target_device = self._requested_device
        target_compute = self._requested_compute_type

        # Decide whether to attempt CUDA
        try_cuda = False
        if target_device in ("auto", "cuda"):
            try:
                cuda_count = ctranslate2.get_cuda_device_count()
                try_cuda = cuda_count > 0
            except Exception as exc:
                logger.debug("Failed checking CUDA devices: %s", exc)
                try_cuda = False

        if try_cuda:
            compute_cuda = (
                "float16" if target_compute in ("auto", "float16") else target_compute
            )
            try:
                logger.info(
                    "Attempting to load faster-whisper model '%s' on CUDA (%s)...",
                    self._model_name,
                    compute_cuda,
                )
                candidate_model = WhisperModel(
                    self._model_name,
                    device="cuda",
                    compute_type=compute_cuda,
                    download_root=self._download_root,
                )
                # Verification probe: test a 100ms silence vector to verify cuBLAS/CUDA runtime DLLs
                dummy_probe = np.zeros(1600, dtype=np.float32)
                list(candidate_model.transcribe(dummy_probe, vad_filter=False)[0])

                self._model = candidate_model
                self._active_device = "cuda"
                self._active_compute_type = compute_cuda
                logger.info("Successfully loaded faster-whisper on CUDA.")
                return
            except Exception as exc:
                logger.warning(
                    "CUDA acceleration failed (%s). Falling back to CPU with int8 quantization.",
                    exc,
                )

        # Fallback to CPU
        compute_cpu = "int8" if target_compute == "auto" else target_compute
        logger.info(
            "Loading faster-whisper model '%s' on CPU (%s)...",
            self._model_name,
            compute_cpu,
        )
        self._model = WhisperModel(
            self._model_name,
            device="cpu",
            compute_type=compute_cpu,
            download_root=self._download_root,
        )
        self._active_device = "cpu"
        self._active_compute_type = compute_cpu
        logger.info("Successfully loaded faster-whisper on CPU.")

    # ── Transcription ─────────────────────────────────────────────────────────

    def transcribe(self, audio: Union[np.ndarray, str]) -> str:
        """
        Transcribe audio using faster-whisper.

        Parameters:
            audio: 1D float32 numpy array or path to an audio file.

        Returns:
            Clean transcribed text string (or empty string if only silence).

        Raises:
            TranscriptionError: If faster-whisper encounters an unrecoverable error.
        """
        # Ensure model is loaded
        if self._model is None:
            self.load()

        if self._model is None:
            raise TranscriptionError("Failed to initialize Whisper STT model.")

        # Quick check for empty or zero-length audio
        if isinstance(audio, np.ndarray):
            if audio.size == 0:
                return ""
            if np.all(audio == 0):
                return ""
            # Ensure float32 1D
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.flatten()
        elif isinstance(audio, str):
            if not os.path.isfile(audio):
                raise TranscriptionError(f"Audio file does not exist: {audio}")

        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=5,
                language=self._language,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            text_parts = [segment.text.strip() for segment in segments if segment.text]
            transcribed = " ".join(text_parts).strip()
            return transcribed
        except Exception as exc:
            logger.error("STT transcription error: %s", exc, exc_info=True)
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
