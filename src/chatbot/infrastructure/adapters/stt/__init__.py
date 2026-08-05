"""Adaptadores de speech-to-text."""

from chatbot.infrastructure.adapters.stt.faster_whisper_adapter import FasterWhisperAdapter
from chatbot.infrastructure.adapters.stt.litellm_transcription_adapter import (
    LiteLLMTranscriptionAdapter,
)
from chatbot.infrastructure.adapters.stt.stt_factory import STTFactory

__all__ = [
    "FasterWhisperAdapter",
    "LiteLLMTranscriptionAdapter",
    "STTFactory",
]
