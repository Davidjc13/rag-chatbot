"""Factory de adaptadores STT."""

from __future__ import annotations

import logging
from typing import Literal

from chatbot.core.env import Env
from chatbot.domain.exceptions import ConfigurationError
from chatbot.domain.ports import TranscriptionPort
from chatbot.infrastructure.adapters.stt.faster_whisper_adapter import FasterWhisperAdapter
from chatbot.infrastructure.adapters.stt.litellm_transcription_adapter import (
    LiteLLMTranscriptionAdapter,
)

logger = logging.getLogger(__name__)

STTProviderName = Literal["faster_whisper", "litellm"]


class STTFactory:
    """Crea la implementación concreta de TranscriptionPort según configuración."""

    @staticmethod
    def create(env: Env | None = None) -> TranscriptionPort:
        settings = env or Env.get_instance()
        provider = settings.stt_provider
        logger.info(
            "Creando adaptador STT",
            extra={"provider": provider, "model": settings.stt_model},
        )

        if provider == "faster_whisper":
            return FasterWhisperAdapter(model=settings.stt_model)

        if provider == "litellm":
            return LiteLLMTranscriptionAdapter(
                model=settings.stt_model,
                api_key=settings.litellm_api_key,
                api_base=settings.litellm_api_base,
                timeout_seconds=settings.http_timeout_seconds,
            )

        raise ConfigurationError(f"Proveedor STT no soportado: {provider}")
