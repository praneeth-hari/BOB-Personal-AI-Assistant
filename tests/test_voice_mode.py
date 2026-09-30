"""
tests/test_voice_mode.py
────────────────────────
Unit tests for continuous voice mode in the CLI.

Recorder, STT, TTS, and the assistant are all mocked — no microphone,
models, or Ollama required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.interfaces.cli import CLISession, _is_voice_exit_phrase
from app.voice.base import AudioRecordingError, NoSpeechDetectedError, TranscriptionError


def _audio() -> np.ndarray:
    return np.zeros(16000, dtype=np.float32)


def _make_session(transcripts, record_side_effect=None, replies=None):
    assistant = MagicMock()
    assistant._llm = SimpleNamespace(model="qwen", host="http://localhost")
    assistant.tool_registry = None
    if replies is not None:
        assistant.chat.side_effect = replies
    else:
        assistant.chat.return_value = "Reply."

    stt = MagicMock()
    stt.transcribe.side_effect = transcripts

    recorder = MagicMock()
    if record_side_effect is not None:
        recorder.record_phrase.side_effect = record_side_effect
    else:
        recorder.record_phrase.return_value = _audio()

    tts = MagicMock()
    session = CLISession(assistant=assistant, stt=stt, recorder=recorder, tts=tts,
                         voice_reply_enabled=False)
    return session, assistant, stt, recorder, tts


def _run(session, inputs):
    with patch("builtins.input", side_effect=inputs), patch("builtins.print"):
        session.run()


class TestExitPhraseDetection:
    @pytest.mark.parametrize("text", [
        "stop listening",
        "Stop listening.",
        "Exit voice mode!",
        "stop voice",
        "Quit voice.",
        "Go back to text.",
        "Okay, stop listening please.",
        "Jarvis, stop listening.",
        "Bob, stop listening.",
    ])
    def test_exit_phrases_detected(self, text):
        assert _is_voice_exit_phrase(text)

    @pytest.mark.parametrize("text", [
        "",
        "What time is it?",
        "How do I stop listening to podcasts on my phone every night?",
        "listening stop",
    ])
    def test_non_exit_phrases(self, text):
        assert not _is_voice_exit_phrase(text)


class TestContinuousVoiceMode:
    def test_multiple_questions_without_leaving_voice_mode(self):
        session, assistant, stt, recorder, tts = _make_session(
            ["What time is it?", "What's the weather?", "stop listening"],
            replies=["It is noon.", "Sunny."],
        )
        _run(session, ["voice", "exit"])

        assert recorder.record_phrase.call_count == 3
        assert [c.args[0] for c in assistant.chat.call_args_list] == [
            "What time is it?", "What's the weather?",
        ]
        assert [c.args[0] for c in tts.speak.call_args_list] == ["It is noon.", "Sunny."]

    def test_stop_listening_exits_without_calling_llm(self):
        session, assistant, stt, recorder, tts = _make_session(["Stop listening."])
        _run(session, ["/voice", "exit"])

        assistant.chat.assert_not_called()
        tts.speak.assert_not_called()
        recorder.record_phrase.assert_called_once()

    def test_returns_to_text_mode_with_same_assistant(self):
        session, assistant, *_ = _make_session(["Hello", "go back to text"])
        _run(session, ["voice", "typed question", "exit"])

        # Voice and text both go through the same assistant instance.
        assert [c.args[0] for c in assistant.chat.call_args_list] == ["Hello", "typed question"]

    def test_ctrl_c_while_listening_returns_to_text_cli(self):
        session, assistant, stt, recorder, tts = _make_session(
            [], record_side_effect=KeyboardInterrupt,
        )
        _run(session, ["voice", "typed after voice", "exit"])

        # Text CLI keeps working after the interrupt.
        assistant.chat.assert_called_once_with("typed after voice")
        stt.transcribe.assert_not_called()

    def test_ctrl_c_during_reply_returns_to_text_cli(self):
        session, assistant, *_ = _make_session(
            ["question"], replies=[KeyboardInterrupt, "Text reply."],
        )
        _run(session, ["voice", "typed", "exit"])

        assert [c.args[0] for c in assistant.chat.call_args_list] == ["question", "typed"]

    def test_silence_timeout_keeps_listening(self):
        session, assistant, stt, recorder, tts = _make_session(
            ["Hi there", "stop listening"],
            record_side_effect=[
                NoSpeechDetectedError("No speech detected."),
                NoSpeechDetectedError("No speech detected."),
                _audio(),
                NoSpeechDetectedError("No speech detected."),
                _audio(),
            ],
        )
        _run(session, ["voice", "exit"])

        assert recorder.record_phrase.call_count == 5
        assistant.chat.assert_called_once_with("Hi there")

    def test_empty_transcription_keeps_listening(self):
        session, assistant, *_ = _make_session(["", "   ", "Hello", "stop voice"])
        _run(session, ["voice", "exit"])

        assistant.chat.assert_called_once_with("Hello")

    def test_transient_stt_and_mic_errors_do_not_crash(self):
        session, assistant, stt, recorder, tts = _make_session(
            [TranscriptionError("decode failed"), "Hello", "quit voice"],
            record_side_effect=[
                _audio(),
                AudioRecordingError("device busy"),
                _audio(),
                _audio(),
            ],
        )
        _run(session, ["voice", "exit"])

        assistant.chat.assert_called_once_with("Hello")

    def test_persistent_mic_error_leaves_voice_mode(self):
        session, assistant, stt, recorder, tts = _make_session(
            [], record_side_effect=AudioRecordingError("no microphone"),
        )
        _run(session, ["voice", "typed", "exit"])

        # Gives up after a few consecutive failures instead of looping forever.
        assert recorder.record_phrase.call_count == 3
        assistant.chat.assert_called_once_with("typed")

    def test_stt_load_failure_returns_to_text_cli(self):
        assistant = MagicMock()
        assistant._llm = SimpleNamespace(model="qwen", host="http://localhost")
        assistant.tool_registry = None
        assistant.chat.return_value = "ok"
        session = CLISession(assistant=assistant, recorder=MagicMock(), voice_reply_enabled=False)

        with patch("app.voice.stt.WhisperSTT") as mock_cls:
            mock_cls.return_value.load.side_effect = RuntimeError("model missing")
            _run(session, ["voice", "typed", "exit"])

        assistant.chat.assert_called_once_with("typed")
