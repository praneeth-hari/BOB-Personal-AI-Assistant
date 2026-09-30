"""
app/voice/__init__.py
─────────────────────
BOB Local Voice, Speech-to-Text (STT) and Text-to-Speech (TTS) module.
"""

from app.voice.audio_recorder import AudioRecorder
from app.voice.base import (
    AudioPlaybackError,
    AudioRecordingError,
    NoSpeechDetectedError,
    SpeechToText,
    SynthesisError,
    TextToSpeech,
    TranscriptionError,
    TTSError,
    VoiceError,
)
from app.voice.stt import WhisperSTT
from app.voice.tts import PiperTTS, clean_text_for_speech

__all__ = [
    "AudioPlaybackError",
    "AudioRecorder",
    "AudioRecordingError",
    "NoSpeechDetectedError",
    "PiperTTS",
    "SpeechToText",
    "SynthesisError",
    "TextToSpeech",
    "TranscriptionError",
    "TTSError",
    "VoiceError",
    "WhisperSTT",
    "clean_text_for_speech",
]
