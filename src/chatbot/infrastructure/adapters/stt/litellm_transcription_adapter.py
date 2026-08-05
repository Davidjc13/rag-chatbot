"""Adaptador STT vía LiteLLM (p. ej. OpenAI whisper-1)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from litellm import atranscription
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from chatbot.domain.exceptions import TranscriptionUnavailableError
from chatbot.domain.ports import TranscriptionPort

logger = logging.getLogger(__name__)

_MIME_TO_SUFFIX: dict[str, str] = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
}


class LiteLLMTranscriptionAdapter(TranscriptionPort):
    """Transcripción remota mediante litellm.atranscription."""

    def __init__(
        self,
        *,
        model: str = "whisper-1",
        api_key: str | None = None,
        api_base: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._timeout_seconds = timeout_seconds

    @property
    def model_name(self) -> str:
        return self._model

    async def transcribe(self, audio: bytes, *, mime: str, language: str) -> str:
        suffix = _MIME_TO_SUFFIX.get(mime, ".webm")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio)
            tmp_path = Path(tmp.name)
        try:
            with tmp_path.open("rb") as audio_file:
                kwargs: dict[str, object] = {
                    "model": self._model,
                    "file": audio_file,
                    "timeout": self._timeout_seconds,
                }
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                if self._api_base:
                    kwargs["api_base"] = self._api_base
                if language:
                    kwargs["language"] = language
                response = await atranscription(**kwargs)
        except (
            APIConnectionError,
            APIError,
            AuthenticationError,
            RateLimitError,
            ServiceUnavailableError,
            Timeout,
        ) as exc:
            logger.error("LiteLLM STT falló: %s", exc)
            raise TranscriptionUnavailableError(
                f"Transcripción no disponible: {exc}",
                provider="litellm",
            ) from exc
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            logger.exception("Error inesperado en LiteLLM STT")
            raise TranscriptionUnavailableError(
                f"Error al transcribir audio: {exc}",
                provider="litellm",
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        text = getattr(response, "text", None) or ""
        return str(text).strip()
