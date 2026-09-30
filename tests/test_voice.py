"""
tests/test_voice.py
───────────────────
Unit tests for the BOB Voice / STT module (Phase 3).

All tests use mocks — no physical microphone, GPU, or model downloads required.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.voice.audio_recorder import AudioRecorder, DEFAULT_CHUNK_DURATION
from app.voice.base import (
    AudioRecordingError,
    NoSpeechDetectedError,
    SpeechToText,
    TranscriptionError,
    VoiceError,
)
from app.voice.stt import WhisperSTT
from app.interfaces.cli import CLISession


# ─── 1. Protocol & Exception Hierarchy Tests ─────────────────────────────────

class TestVoiceBase:
    def test_voice_exception_hierarchy(self):
        assert issubclass(AudioRecordingError, VoiceError)
        assert issubclass(NoSpeechDetectedError, VoiceError)
        assert issubclass(TranscriptionError, VoiceError)
        assert issubclass(VoiceError, Exception)

    def test_whisper_stt_implements_protocol(self):
        stt = WhisperSTT(lazy_load=True)
        assert isinstance(stt, SpeechToText)


# ─── 2. AudioRecorder Tests ──────────────────────────────────────────────────

class TestAudioRecorderConfig:
    def test_default_config(self):
        rec = AudioRecorder()
        assert rec.sample_rate == 16000
        assert rec.silence_threshold == 0.015
        assert rec.silence_duration == 1.2
        assert rec.initial_timeout == 7.0
        assert rec.max_duration == 30.0
        assert rec.min_speech_duration == 0.4
        assert rec.device_index is None

    def test_custom_config(self):
        rec = AudioRecorder(
            sample_rate=8000,
            silence_threshold=0.03,
            silence_duration=2.0,
            initial_timeout=5.0,
            max_duration=15.0,
            min_speech_duration=1.0,
            device_index=3,
        )
        assert rec.sample_rate == 8000
        assert rec.silence_threshold == 0.03
        assert rec.silence_duration == 2.0
        assert rec.initial_timeout == 5.0
        assert rec.max_duration == 15.0
        assert rec.min_speech_duration == 1.0
        assert rec.device_index == 3

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("AUDIO_SAMPLE_RATE", "22050")
        monkeypatch.setenv("AUDIO_SILENCE_THRESHOLD", "0.025")
        monkeypatch.setenv("AUDIO_SILENCE_DURATION", "1.8")
        monkeypatch.setenv("AUDIO_INITIAL_TIMEOUT", "4.5")
        monkeypatch.setenv("AUDIO_MAX_DURATION", "20.0")
        monkeypatch.setenv("AUDIO_MIN_SPEECH_DURATION", "0.8")
        monkeypatch.setenv("AUDIO_INPUT_DEVICE", "2")

        rec = AudioRecorder()
        assert rec.sample_rate == 22050
        assert rec.silence_threshold == 0.025
        assert rec.silence_duration == 1.8
        assert rec.initial_timeout == 4.5
        assert rec.max_duration == 20.0
        assert rec.min_speech_duration == 0.8
        assert rec.device_index == 2

    def test_invalid_device_env_var(self, monkeypatch):
        monkeypatch.setenv("AUDIO_INPUT_DEVICE", "not_a_number")
        rec = AudioRecorder()
        assert rec.device_index is None


class TestAudioRecorderRMS:
    def test_rms_of_zeros(self):
        chunk = np.zeros(1600, dtype=np.float32)
        assert AudioRecorder.compute_rms(chunk) == 0.0

    def test_rms_of_empty_chunk(self):
        chunk = np.array([], dtype=np.float32)
        assert AudioRecorder.compute_rms(chunk) == 0.0

    def test_rms_of_constant_signal(self):
        chunk = np.full(1000, 0.5, dtype=np.float32)
        rms = AudioRecorder.compute_rms(chunk)
        assert pytest.approx(rms, rel=1e-5) == 0.5


class TestAudioRecorderDevices:
    def test_list_input_devices_filters_inputs(self):
        mock_devices = [
            {"name": "Mic 1", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "Speaker 1", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "Mic 2", "max_input_channels": 1, "default_samplerate": 16000},
        ]
        with patch("sounddevice.query_devices", return_value=mock_devices):
            devices = AudioRecorder.list_input_devices()
            assert len(devices) == 2
            assert devices[0]["name"] == "Mic 1"
            assert devices[0]["index"] == 0
            assert devices[1]["name"] == "Mic 2"
            assert devices[1]["index"] == 2

    def test_list_input_devices_exception_returns_empty(self):
        with patch("sounddevice.query_devices", side_effect=RuntimeError("PortAudio error")):
            devices = AudioRecorder.list_input_devices()
            assert devices == []


class TestAudioRecorderCapture:
    def _create_mock_stream(self, chunks):
        """Helper to create a mock sounddevice.InputStream that yields pre-defined chunks."""
        chunk_iter = iter(chunks)
        mock_stream = MagicMock()

        def mock_read(chunk_size):
            try:
                return next(chunk_iter), False
            except StopIteration:
                # Return silence forever if called past end
                return np.zeros((chunk_size, 1), dtype=np.float32), False

        mock_stream.read.side_effect = mock_read
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = None
        return mock_stream

    def test_record_phrase_successful(self):
        rec = AudioRecorder(
            sample_rate=16000,
            silence_threshold=0.02,
            silence_duration=0.3,   # Stop after 3 silent chunks (0.3s)
            min_speech_duration=0.2, # At least 2 speech chunks
            initial_timeout=5.0,
        )
        chunk_len = rec.chunk_size

        silence_chunk = np.zeros((chunk_len, 1), dtype=np.float32)
        speech_chunk = np.full((chunk_len, 1), 0.1, dtype=np.float32)

        # 2 silent chunks -> 3 speech chunks -> 4 silent chunks (triggers stop)
        chunks = [
            silence_chunk,
            silence_chunk,
            speech_chunk,
            speech_chunk,
            speech_chunk,
            silence_chunk,
            silence_chunk,
            silence_chunk,
        ]

        started = []
        ended = []

        mock_stream = self._create_mock_stream(chunks)
        with patch("sounddevice.InputStream", return_value=mock_stream):
            audio = rec.record_phrase(
                on_speech_start=lambda: started.append(True),
                on_speech_end=lambda: ended.append(True),
            )

        assert len(started) == 1
        assert len(ended) == 1
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
        # Speech was present, so RMS > 0
        assert np.max(np.abs(audio)) >= 0.1

    def test_record_phrase_initial_timeout_raises_no_speech(self):
        rec = AudioRecorder(
            sample_rate=16000,
            silence_threshold=0.05,
            initial_timeout=0.01, # Instant timeout
        )
        silence_chunk = np.zeros((rec.chunk_size, 1), dtype=np.float32)
        mock_stream = self._create_mock_stream([silence_chunk] * 10)

        with patch("sounddevice.InputStream", return_value=mock_stream):
            with pytest.raises(NoSpeechDetectedError, match="No speech detected"):
                rec.record_phrase()

    def test_record_phrase_too_short_raises_no_speech(self):
        rec = AudioRecorder(
            sample_rate=16000,
            silence_threshold=0.02,
            silence_duration=0.1,
            min_speech_duration=1.0, # Needs 1.0s, we give only 0.1s
            initial_timeout=5.0,
        )
        chunk_len = rec.chunk_size
        silence_chunk = np.zeros((chunk_len, 1), dtype=np.float32)
        speech_chunk = np.full((chunk_len, 1), 0.1, dtype=np.float32)

        chunks = [speech_chunk, silence_chunk, silence_chunk]
        mock_stream = self._create_mock_stream(chunks)

        with patch("sounddevice.InputStream", return_value=mock_stream):
            with pytest.raises(NoSpeechDetectedError, match="Audio too short"):
                rec.record_phrase()

    def test_record_phrase_portaudio_error_raises_audio_recording_error(self):
        rec = AudioRecorder()
        with patch("sounddevice.InputStream", side_effect=RuntimeError("Cannot open device")):
            with pytest.raises(AudioRecordingError, match="Failed to record audio"):
                rec.record_phrase()


# ─── 3. WhisperSTT Tests ─────────────────────────────────────────────────────

class TestWhisperSTTConfig:
    def test_defaults(self):
        stt = WhisperSTT(lazy_load=True)
        assert stt.model_name == "base"
        assert stt.device == "auto"
        assert stt.compute_type == "auto"
        assert stt.language is None
        assert not stt.is_available()

    def test_custom_parameters(self):
        stt = WhisperSTT(
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            lazy_load=True,
        )
        assert stt.model_name == "tiny"
        assert stt.device == "cpu"
        assert stt.compute_type == "int8"
        assert stt.language == "en"

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL", "small.en")
        monkeypatch.setenv("WHISPER_DEVICE", "cpu")
        monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
        monkeypatch.setenv("WHISPER_LANGUAGE", "fr")

        stt = WhisperSTT(lazy_load=True)
        assert stt.model_name == "small.en"
        assert stt.device == "cpu"
        assert stt.compute_type == "int8"
        assert stt.language == "fr"


class TestWhisperSTTLoading:
    def test_explicit_cpu_loads_cpu(self):
        stt = WhisperSTT(device="cpu", compute_type="int8", lazy_load=True)
        mock_model = MagicMock()

        with patch("faster_whisper.WhisperModel", return_value=mock_model) as mock_cls:
            stt.load()
            mock_cls.assert_called_once_with(
                "base",
                device="cpu",
                compute_type="int8",
                download_root=None,
            )
            assert stt.device == "cpu"
            assert stt.compute_type == "int8"
            assert stt.is_available()

    def test_auto_device_cuda_probe_success(self):
        stt = WhisperSTT(device="auto", lazy_load=True)
        mock_model = MagicMock()
        # Probe transcription succeeds
        mock_model.transcribe.return_value = ([SimpleNamespace(text="probe")], SimpleNamespace())

        with patch("ctranslate2.get_cuda_device_count", return_value=1), \
             patch("faster_whisper.WhisperModel", return_value=mock_model) as mock_cls:
            stt.load()
            mock_cls.assert_called_once_with(
                "base",
                device="cuda",
                compute_type="float16",
                download_root=None,
            )
            assert stt.device == "cuda"
            assert stt.compute_type == "float16"

    def test_auto_device_cuda_probe_failure_falls_back_to_cpu(self):
        stt = WhisperSTT(device="auto", lazy_load=True)
        cuda_model = MagicMock()
        # CUDA probe fails due to missing DLL
        cuda_model.transcribe.side_effect = RuntimeError("cublas64_12.dll not found")
        cpu_model = MagicMock()

        models_instantiated = []

        def side_effect(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                models_instantiated.append("cuda")
                return cuda_model
            else:
                models_instantiated.append("cpu")
                return cpu_model

        with patch("ctranslate2.get_cuda_device_count", return_value=1), \
             patch("faster_whisper.WhisperModel", side_effect=side_effect):
            stt.load()
            assert models_instantiated == ["cuda", "cpu"]
            assert stt.device == "cpu"
            assert stt.compute_type == "int8"
            assert stt.is_available()


class TestWhisperSTTTranscription:
    def test_transcribe_empty_audio_returns_empty_string(self):
        stt = WhisperSTT(lazy_load=True)
        stt._model = MagicMock()
        result = stt.transcribe(np.array([], dtype=np.float32))
        assert result == ""
        # Underlying transcribe should not even be called
        stt._model.transcribe.assert_not_called()

    def test_transcribe_all_zeros_returns_empty_string(self):
        stt = WhisperSTT(lazy_load=True)
        stt._model = MagicMock()
        result = stt.transcribe(np.zeros(16000, dtype=np.float32))
        assert result == ""
        stt._model.transcribe.assert_not_called()

    def test_transcribe_successful(self):
        stt = WhisperSTT(lazy_load=True)
        mock_model = MagicMock()
        segments = [
            SimpleNamespace(text=" Hello"),
            SimpleNamespace(text=" BOB,"),
            SimpleNamespace(text=" what is the weather? "),
        ]
        info = SimpleNamespace(language="en")
        mock_model.transcribe.return_value = (segments, info)
        stt._model = mock_model

        audio = np.random.randn(16000).astype(np.float32)
        text = stt.transcribe(audio)
        assert text == "Hello BOB, what is the weather?"

    def test_transcribe_nonexistent_file_raises_transcription_error(self):
        stt = WhisperSTT(lazy_load=True)
        stt._model = MagicMock()
        with pytest.raises(TranscriptionError, match="does not exist"):
            stt.transcribe("non_existent_audio_file.wav")

    def test_transcribe_exception_wrapped_in_transcription_error(self):
        stt = WhisperSTT(lazy_load=True)
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Whisper decoder crashed")
        stt._model = mock_model

        audio = np.random.randn(16000).astype(np.float32)
        with pytest.raises(TranscriptionError, match="Whisper decoder crashed"):
            stt.transcribe(audio)


# ─── 4. CLI Voice Integration Tests ──────────────────────────────────────────

class TestCLIVoiceIntegration:
    def test_voice_command_invokes_recorder_and_stt(self):
        mock_assistant = MagicMock()
        mock_assistant.chat.return_value = "I heard you clearly."
        mock_assistant._llm = SimpleNamespace(model="qwen", host="http://localhost")
        mock_assistant.tool_registry = None

        mock_stt = MagicMock()
        mock_stt.transcribe.side_effect = ["What time is it?", "stop listening"]

        mock_recorder = MagicMock()
        mock_recorder.record_phrase.return_value = np.zeros(16000, dtype=np.float32)

        mock_tts = MagicMock()

        session = CLISession(
            assistant=mock_assistant,
            stt=mock_stt,
            recorder=mock_recorder,
            tts=mock_tts,
        )

        with patch("builtins.input", side_effect=["voice", "exit"]), \
             patch("builtins.print") as mock_print:
            session.run()

        assert mock_recorder.record_phrase.call_count == 2
        assert mock_stt.transcribe.call_count == 2
        mock_assistant.chat.assert_called_once_with("What time is it?")

    def test_voice_command_handles_no_speech_gracefully(self):
        mock_assistant = MagicMock()
        mock_assistant._llm = SimpleNamespace(model="qwen", host="http://localhost")
        mock_assistant.tool_registry = None

        mock_recorder = MagicMock()
        mock_recorder.record_phrase.side_effect = [
            NoSpeechDetectedError("No speech detected."),
            np.zeros(16000, dtype=np.float32),
        ]
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "stop listening"

        session = CLISession(
            assistant=mock_assistant,
            stt=mock_stt,
            recorder=mock_recorder,
        )

        with patch("builtins.input", side_effect=["/voice", "exit"]), \
             patch("builtins.print"):
            session.run()

        # Assistant should NOT be called when no speech detected
        mock_assistant.chat.assert_not_called()
