"""
tests/test_tts.py
─────────────────
Unit tests for the BOB Text-to-Speech (TTS) module (Phase 4).

All tests use mocks — no physical speakers, GPU, or model downloads required.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.voice.base import (
    AudioPlaybackError,
    SynthesisError,
    TTSError,
    TextToSpeech,
    VoiceError,
)
from app.voice.tts import PiperTTS, clean_text_for_speech
from app.interfaces.cli import CLISession


# ─── 1. Protocol & Exception Hierarchy Tests ─────────────────────────────────

class TestTTSBase:
    def test_tts_exception_hierarchy(self):
        assert issubclass(TTSError, VoiceError)
        assert issubclass(SynthesisError, TTSError)
        assert issubclass(AudioPlaybackError, TTSError)
        assert issubclass(VoiceError, Exception)

    def test_piper_tts_implements_protocol(self):
        tts = PiperTTS(lazy_load=True)
        assert isinstance(tts, TextToSpeech)


# ─── 2. Text Normalization for Speech ────────────────────────────────────────

class TestCleanTextForSpeech:
    def test_empty_and_none(self):
        assert clean_text_for_speech("") == ""
        assert clean_text_for_speech("   ") == ""

    def test_removes_markdown_bold_and_italic(self):
        assert clean_text_for_speech("This is **bold** and *italic* text.") == "This is bold and italic text."
        assert clean_text_for_speech("Using __underline__ and _emphasis_.") == "Using underline and emphasis."

    def test_removes_headers(self):
        text = "# Heading 1\n## Heading 2\n### Heading 3\nActual text."
        assert clean_text_for_speech(text) == "Heading 1 Heading 2 Heading 3 Actual text."

    def test_replaces_code_blocks(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        assert clean_text_for_speech(text) == "Here is code: [code block omitted] Done."

    def test_removes_inline_code(self):
        assert clean_text_for_speech("Run `python -m app.main` now.") == "Run python -m app.main now."

    def test_removes_markdown_links(self):
        assert clean_text_for_speech("Check [the documentation](https://example.com) for details.") == "Check the documentation for details."

    def test_removes_bullet_points_and_numbers(self):
        text = "- First item\n* Second item\n1. Third item"
        assert clean_text_for_speech(text) == "First item Second item Third item"

    def test_replaces_table_pipes(self):
        text = "| Name | Value |\n| RAM | 16 GB |"
        cleaned = clean_text_for_speech(text)
        assert "|" not in cleaned
        assert "RAM" in cleaned and "16 GB" in cleaned


# ─── 3. PiperTTS Configuration Tests ─────────────────────────────────────────

class TestPiperTTSConfig:
    def test_default_config(self):
        tts = PiperTTS(lazy_load=True)
        assert tts.voice_name == "en_US-lessac-medium"
        assert tts._use_cuda is False
        assert tts._speaker_id is None
        assert tts._length_scale == 1.0
        assert tts._noise_scale == 0.667
        assert tts._noise_w == 0.8
        assert not tts.is_available()

    def test_custom_parameters(self, tmp_path):
        tts = PiperTTS(
            voice_name="en_US-amy-medium",
            voices_dir=tmp_path,
            use_cuda=True,
            speaker_id=2,
            length_scale=0.9,
            noise_scale=0.5,
            noise_w=0.7,
            lazy_load=True,
        )
        assert tts.voice_name == "en_US-amy-medium"
        assert tts.voices_dir == tmp_path
        assert tts._use_cuda is True
        assert tts._speaker_id == 2
        assert tts._length_scale == 0.9
        assert tts._noise_scale == 0.5
        assert tts._noise_w == 0.7

    def test_env_var_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PIPER_VOICE", "en_GB-alan-medium")
        monkeypatch.setenv("PIPER_VOICES_DIR", str(tmp_path))
        monkeypatch.setenv("PIPER_USE_CUDA", "true")
        monkeypatch.setenv("PIPER_SPEAKER_ID", "1")
        monkeypatch.setenv("PIPER_LENGTH_SCALE", "1.1")
        monkeypatch.setenv("PIPER_NOISE_SCALE", "0.6")
        monkeypatch.setenv("PIPER_NOISE_W", "0.75")

        tts = PiperTTS(lazy_load=True)
        assert tts.voice_name == "en_GB-alan-medium"
        assert tts.voices_dir == tmp_path
        assert tts._use_cuda is True
        assert tts._speaker_id == 1
        assert tts._length_scale == 1.1
        assert tts._noise_scale == 0.6
        assert tts._noise_w == 0.75

    def test_invalid_speaker_id_env_var(self, monkeypatch):
        monkeypatch.setenv("PIPER_SPEAKER_ID", "not_a_number")
        tts = PiperTTS(lazy_load=True)
        assert tts._speaker_id is None


# ─── 4. PiperTTS Loading Tests ───────────────────────────────────────────────

class TestPiperTTSLoading:
    def test_load_when_files_exist(self, tmp_path):
        onnx_file = tmp_path / "en_US-lessac-medium.onnx"
        json_file = tmp_path / "en_US-lessac-medium.onnx.json"
        onnx_file.write_text("dummy onnx")
        json_file.write_text("{}")

        tts = PiperTTS(voices_dir=tmp_path, lazy_load=True)
        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050

        with patch("piper.PiperVoice.load", return_value=mock_voice) as mock_load, \
             patch.object(tts, "_download_voice_model") as mock_download:
            tts.load()
            mock_download.assert_not_called()
            mock_load.assert_called_once_with(
                model_path=onnx_file,
                config_path=json_file,
                use_cuda=False,
            )
            assert tts.is_available()
            assert tts._sample_rate == 22050

    def test_load_downloads_when_files_missing(self, tmp_path):
        tts = PiperTTS(voices_dir=tmp_path, lazy_load=True)
        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050

        # Simulate files created by download
        def fake_download():
            (tmp_path / "en_US-lessac-medium.onnx").write_text("onnx")
            (tmp_path / "en_US-lessac-medium.onnx.json").write_text("{}")

        with patch.object(tts, "_download_voice_model", side_effect=fake_download) as mock_dl, \
             patch("piper.PiperVoice.load", return_value=mock_voice):
            tts.load()
            mock_dl.assert_called_once()
            assert tts.is_available()

    def test_load_failure_raises_tts_error(self, tmp_path):
        onnx_file = tmp_path / "en_US-lessac-medium.onnx"
        json_file = tmp_path / "en_US-lessac-medium.onnx.json"
        onnx_file.write_text("onnx")
        json_file.write_text("{}")

        tts = PiperTTS(voices_dir=tmp_path, lazy_load=True)
        with patch("piper.PiperVoice.load", side_effect=RuntimeError("Corrupt ONNX model")):
            with pytest.raises(TTSError, match="Failed to load Piper voice model"):
                tts.load()

    def test_download_failure_raises_tts_error(self, tmp_path):
        tts = PiperTTS(voices_dir=tmp_path, lazy_load=True)
        with patch("piper.download_voices.download_voice", side_effect=RuntimeError("Network down")):
            with pytest.raises(TTSError, match="Failed to download Piper voice model"):
                tts.load()


# ─── 5. Speech Synthesis Tests ───────────────────────────────────────────────

class TestPiperTTSSynthesis:
    def test_synthesize_empty_text_returns_empty_array(self):
        tts = PiperTTS(lazy_load=True)
        tts._voice = MagicMock()
        audio, sample_rate = tts.synthesize("")
        assert len(audio) == 0
        tts._voice.synthesize.assert_not_called()

    def test_synthesize_whitespace_returns_empty_array(self):
        tts = PiperTTS(lazy_load=True)
        tts._voice = MagicMock()
        audio, sample_rate = tts.synthesize("   \n\t  ")
        assert len(audio) == 0
        tts._voice.synthesize.assert_not_called()

    def test_synthesize_successful(self):
        tts = PiperTTS(lazy_load=True)
        mock_voice = MagicMock()

        chunk1 = SimpleNamespace(
            sample_rate=22050,
            audio_float_array=np.full(1000, 0.1, dtype=np.float32),
        )
        chunk2 = SimpleNamespace(
            sample_rate=22050,
            audio_float_array=np.full(1000, 0.2, dtype=np.float32),
        )
        mock_voice.synthesize.return_value = [chunk1, chunk2]
        tts._voice = mock_voice

        audio, sample_rate = tts.synthesize("Hello world")
        assert sample_rate == 22050
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 2000
        assert audio.dtype == np.float32

    def test_synthesize_error_raises_synthesis_error(self):
        tts = PiperTTS(lazy_load=True)
        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = RuntimeError("Synthesis engine error")
        tts._voice = mock_voice

        with pytest.raises(SynthesisError, match="Speech synthesis failed"):
            tts.synthesize("Test phrase")


# ─── 6. Audio Playback Tests ─────────────────────────────────────────────────

class TestPiperTTSPlayback:
    def test_play_audio_empty_array_is_noop(self):
        tts = PiperTTS(lazy_load=True)
        with patch("sounddevice.play") as mock_play:
            tts.play_audio(np.array([], dtype=np.float32), 22050)
            mock_play.assert_not_called()

    def test_play_audio_blocking(self):
        tts = PiperTTS(lazy_load=True)
        audio = np.full(1000, 0.05, dtype=np.float32)

        with patch("sounddevice.play") as mock_play, \
             patch("sounddevice.wait") as mock_wait:
            tts.play_audio(audio, 22050, blocking=True)
            mock_play.assert_called_once_with(audio, samplerate=22050)
            mock_wait.assert_called_once()

    def test_play_audio_non_blocking(self):
        tts = PiperTTS(lazy_load=True)
        audio = np.full(1000, 0.05, dtype=np.float32)

        with patch("sounddevice.play") as mock_play, \
             patch("sounddevice.wait") as mock_wait:
            tts.play_audio(audio, 22050, blocking=False)
            mock_play.assert_called_once_with(audio, samplerate=22050)
            mock_wait.assert_not_called()

    def test_play_audio_error_raises_audio_playback_error(self):
        tts = PiperTTS(lazy_load=True)
        audio = np.full(1000, 0.05, dtype=np.float32)

        with patch("sounddevice.play", side_effect=RuntimeError("PortAudio playback failed")):
            with pytest.raises(AudioPlaybackError, match="Audio playback failed"):
                tts.play_audio(audio, 22050)


# ─── 7. Speak Convenience Method Tests ───────────────────────────────────────

class TestPiperTTSSpeak:
    def test_speak_calls_synthesize_and_play(self):
        tts = PiperTTS(lazy_load=True)
        audio_data = np.full(1000, 0.1, dtype=np.float32)

        with patch.object(tts, "synthesize", return_value=(audio_data, 22050)) as mock_synth, \
             patch.object(tts, "play_audio") as mock_play:
            tts.speak("Good morning.")
            mock_synth.assert_called_once_with("Good morning.")
            mock_play.assert_called_once_with(audio_data, 22050, blocking=True)

    def test_speak_empty_text_skips_playback(self):
        tts = PiperTTS(lazy_load=True)
        with patch.object(tts, "synthesize", return_value=(np.array([], dtype=np.float32), 22050)), \
             patch.object(tts, "play_audio") as mock_play:
            tts.speak("")
            mock_play.assert_not_called()


# ─── 8. CLI Voice & TTS Integration Tests ────────────────────────────────────

class TestCLITTSIntegration:
    def test_voice_command_speaks_assistant_reply(self):
        mock_assistant = MagicMock()
        mock_assistant.chat.return_value = "Hello from BOB."
        mock_assistant._llm = SimpleNamespace(model="qwen", host="http://localhost")
        mock_assistant.tool_registry = None

        mock_stt = MagicMock()
        mock_stt.transcribe.side_effect = ["Hello", "stop listening"]

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
             patch("builtins.print"):
            session.run()

        # STT transcribed (question + exit phrase)
        assert mock_stt.transcribe.call_count == 2
        # Assistant responded
        mock_assistant.chat.assert_called_once_with("Hello")
        # TTS spoke the assistant reply aloud!
        mock_tts.speak.assert_called_once_with("Hello from BOB.", blocking=True)

    def test_tts_command_speaks_custom_text(self):
        mock_assistant = MagicMock()
        mock_assistant._llm = SimpleNamespace(model="qwen", host="http://localhost")
        mock_assistant.tool_registry = None
        mock_tts = MagicMock()

        session = CLISession(
            assistant=mock_assistant,
            tts=mock_tts,
        )

        with patch("builtins.input", side_effect=["/tts Testing speech synthesis", "exit"]), \
             patch("builtins.print"):
            session.run()

        mock_tts.speak.assert_called_once_with("Testing speech synthesis", blocking=True)
        # Assistant chat should not have been called
        mock_assistant.chat.assert_not_called()

    def test_voice_reply_toggle_command(self):
        mock_assistant = MagicMock()
        mock_assistant.chat.return_value = "Typed response."
        mock_assistant._llm = SimpleNamespace(model="qwen", host="http://localhost")
        mock_assistant.tool_registry = None
        mock_tts = MagicMock()

        session = CLISession(
            assistant=mock_assistant,
            tts=mock_tts,
            voice_reply_enabled=False,
        )

        # 1. First typed input without voice-reply: does NOT speak
        # 2. Toggle /voice-reply
        # 3. Second typed input with voice-reply: DOES speak
        with patch("builtins.input", side_effect=["First input", "/voice-reply", "Second input", "exit"]), \
             patch("builtins.print"):
            session.run()

        # Only the second input should have triggered tts.speak
        mock_tts.speak.assert_called_once_with("Typed response.", blocking=True)
