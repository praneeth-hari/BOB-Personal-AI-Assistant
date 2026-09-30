"""
app/voice/audio_recorder.py
───────────────────────────
Microphone capture with adaptive energy-based voice activity detection (VAD).

Records 16kHz mono float32 audio suitable for Whisper transcription:
  1. Listens for speech above an RMS energy threshold.
  2. Keeps a small pre-roll buffer so the beginning of the first word isn't clipped.
  3. Records while the user speaks.
  4. Automatically stops when a period of silence is detected following speech.
  5. Enforces safety limits: initial silence timeout and maximum total duration.
"""

from __future__ import annotations

import collections
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.voice.base import AudioRecordingError, NoSpeechDetectedError

logger = logging.getLogger(__name__)

# ─── Default Audio Configuration ──────────────────────────────────────────────

DEFAULT_SAMPLE_RATE = 16000          # Whisper standard sample rate
DEFAULT_CHANNELS = 1                 # Mono
DEFAULT_CHUNK_DURATION = 0.1         # Read in 100ms blocks
DEFAULT_SILENCE_THRESHOLD = 0.015    # RMS threshold to detect voice activity
DEFAULT_SILENCE_DURATION = 1.2       # Seconds of silence after speech to stop
DEFAULT_INITIAL_TIMEOUT = 7.0        # Seconds to wait for first speech before timeout
DEFAULT_MAX_DURATION = 30.0          # Max total recording duration safety limit
DEFAULT_MIN_SPEECH_DURATION = 0.4    # Minimum speech duration to qualify as valid utterance
DEFAULT_PRE_ROLL_DURATION = 0.3      # Pre-speech audio buffer to preserve word onsets


class AudioRecorder:
    """
    Microphone recorder with energy-based silence detection.

    All audio captured is 1-channel, 16kHz float32 in [-1.0, 1.0].
    """

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        silence_threshold: Optional[float] = None,
        silence_duration: Optional[float] = None,
        initial_timeout: Optional[float] = None,
        max_duration: Optional[float] = None,
        min_speech_duration: Optional[float] = None,
        pre_roll_duration: Optional[float] = None,
        device_index: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate or int(
            os.environ.get("AUDIO_SAMPLE_RATE", DEFAULT_SAMPLE_RATE)
        )
        self.silence_threshold = silence_threshold or float(
            os.environ.get("AUDIO_SILENCE_THRESHOLD", DEFAULT_SILENCE_THRESHOLD)
        )
        self.silence_duration = silence_duration or float(
            os.environ.get("AUDIO_SILENCE_DURATION", DEFAULT_SILENCE_DURATION)
        )
        self.initial_timeout = initial_timeout or float(
            os.environ.get("AUDIO_INITIAL_TIMEOUT", DEFAULT_INITIAL_TIMEOUT)
        )
        self.max_duration = max_duration or float(
            os.environ.get("AUDIO_MAX_DURATION", DEFAULT_MAX_DURATION)
        )
        self.min_speech_duration = min_speech_duration or float(
            os.environ.get("AUDIO_MIN_SPEECH_DURATION", DEFAULT_MIN_SPEECH_DURATION)
        )
        self.pre_roll_duration = pre_roll_duration or float(
            os.environ.get("AUDIO_PRE_ROLL_DURATION", DEFAULT_PRE_ROLL_DURATION)
        )
        self.device_index = device_index
        if self.device_index is None and "AUDIO_INPUT_DEVICE" in os.environ:
            try:
                self.device_index = int(os.environ["AUDIO_INPUT_DEVICE"])
            except ValueError:
                logger.warning("Invalid AUDIO_INPUT_DEVICE env var: %r", os.environ["AUDIO_INPUT_DEVICE"])

        self.chunk_size = int(self.sample_rate * DEFAULT_CHUNK_DURATION)

    # ── Device inspection ─────────────────────────────────────────────────────

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """
        List all available audio input devices.
        Returns a list of dictionaries with device index, name, and channels.
        """
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = []
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    input_devices.append({
                        "index": idx,
                        "name": dev.get("name", f"Device {idx}"),
                        "channels": dev.get("max_input_channels", 1),
                        "default_samplerate": dev.get("default_samplerate", 16000),
                    })
            return input_devices
        except Exception as exc:
            logger.warning("Failed to list audio devices: %s", exc)
            return []

    # ── Energy computation ───────────────────────────────────────────────────

    @staticmethod
    def compute_rms(chunk: np.ndarray) -> float:
        """Calculate Root-Mean-Square energy of an audio chunk."""
        if len(chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(chunk**2)))

    # ── Main recording method ─────────────────────────────────────────────────

    def record_phrase(
        self,
        on_listening: Optional[Callable[[], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ) -> np.ndarray:
        """
        Listen to the microphone and record a single utterance until silence is detected.

        Callbacks can be supplied for UI/CLI feedback:
          - on_listening(): Fired when microphone is open and waiting for user to speak.
          - on_speech_start(): Fired as soon as speech energy crosses the threshold.
          - on_speech_end(): Fired when silence duration triggers completion.

        Returns:
            1D float32 numpy array normalized to [-1.0, 1.0] at self.sample_rate.

        Raises:
            AudioRecordingError: If microphone stream fails to open or read.
            NoSpeechDetectedError: If no speech was detected before initial timeout.
        """
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioRecordingError(
                "sounddevice is not installed. Install it with: pip install sounddevice"
            ) from exc

        num_pre_roll_chunks = max(1, int(self.pre_roll_duration / DEFAULT_CHUNK_DURATION))
        pre_roll_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=num_pre_roll_chunks)

        speech_chunks: List[np.ndarray] = []
        is_speech_active = False
        consecutive_silence_time = 0.0

        if on_listening:
            on_listening()

        start_time = time.monotonic()
        speech_start_time: Optional[float] = None

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=DEFAULT_CHANNELS,
                dtype="float32",
                device=self.device_index,
            ) as stream:
                while True:
                    # Read one chunk (blocking for chunk_duration ~100ms)
                    raw_chunk, overflowed = stream.read(self.chunk_size)
                    if overflowed:
                        logger.debug("Audio input buffer overflowed")

                    # Flatten to 1D
                    chunk = raw_chunk.flatten()
                    rms = self.compute_rms(chunk)
                    now = time.monotonic()

                    if not is_speech_active:
                        pre_roll_buffer.append(chunk)

                        if rms >= self.silence_threshold:
                            # Speech has started!
                            is_speech_active = True
                            speech_start_time = now
                            consecutive_silence_time = 0.0
                            # Include the pre-roll buffer so the start of words is not clipped
                            speech_chunks.extend(pre_roll_buffer)
                            pre_roll_buffer.clear()
                            if on_speech_start:
                                on_speech_start()
                        elif now - start_time >= self.initial_timeout:
                            # User remained silent past the initial timeout
                            logger.info("No speech detected within %.1fs initial timeout", self.initial_timeout)
                            raise NoSpeechDetectedError(
                                f"No speech detected within {self.initial_timeout:.1f} seconds."
                            )
                    else:
                        # Speech is currently being recorded
                        speech_chunks.append(chunk)

                        if rms < self.silence_threshold:
                            consecutive_silence_time += DEFAULT_CHUNK_DURATION
                            if consecutive_silence_time >= self.silence_duration:
                                # Silence threshold reached after speech
                                if on_speech_end:
                                    on_speech_end()
                                break
                        else:
                            # User spoke again within silence window; reset silence accumulator
                            consecutive_silence_time = 0.0

                        # Check maximum duration safety limit
                        if speech_start_time and (now - speech_start_time >= self.max_duration):
                            logger.info("Maximum recording duration (%.1fs) reached", self.max_duration)
                            if on_speech_end:
                                on_speech_end()
                            break

        except NoSpeechDetectedError:
            raise
        except Exception as exc:
            logger.error("Audio recording error: %s", exc, exc_info=True)
            raise AudioRecordingError(f"Failed to record audio from microphone: {exc}") from exc

        if not speech_chunks:
            raise NoSpeechDetectedError("No audio data was captured.")

        full_audio = np.concatenate(speech_chunks, axis=0).astype(np.float32)
        total_duration = len(full_audio) / self.sample_rate

        if total_duration < self.min_speech_duration:
            logger.info("Recorded audio was too short (%.2fs < %.2fs)", total_duration, self.min_speech_duration)
            raise NoSpeechDetectedError(
                f"Audio too short to contain speech ({total_duration:.2f}s)."
            )

        logger.debug("Captured %.2fs of audio (%d samples)", total_duration, len(full_audio))
        return full_audio
