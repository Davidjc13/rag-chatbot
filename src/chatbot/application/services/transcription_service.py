"""Servicio de transcripción de voz a texto."""

from __future__ import annotations

from chatbot.core.env import Env
from chatbot.domain.exceptions import TranscriptionUnavailableError, ValidationError
from chatbot.domain.ports import TranscriptionPort
from chatbot.infrastructure.adapters.api.mime_validation import normalize_mime

_ALLOWED_AUDIO_MIMES = frozenset(
    {
        "audio/webm",
        "audio/ogg",
        "audio/wav",
        "audio/x-wav",
        "audio/mp4",
        "audio/mpeg",
        "audio/mp3",
    }
)

# ~5 MB; suficiente para ~60 s de audio comprimido en webm.
_MAX_AUDIO_BYTES = 5 * 1024 * 1024


class TranscriptionService:
    """Valida audio de entrada y delega la transcripción al puerto STT."""

    def __init__(
        self,
        *,
        stt: TranscriptionPort,
        env: Env | None = None,
    ) -> None:
        self._stt = stt
        self._env = env or Env.get_instance()

    @property
    def model_name(self) -> str:
        return self._stt.model_name

    async def transcribe(self, *, data: bytes, content_type: str | None) -> str:
        if not self._env.stt_enabled:
            raise TranscriptionUnavailableError(
                "La transcripción por voz está deshabilitada",
                provider=self._env.stt_provider,
            )
        if not data:
            raise ValidationError("El audio está vacío")
        if len(data) > _MAX_AUDIO_BYTES:
            raise ValidationError(
                f"El audio supera el límite de {_MAX_AUDIO_BYTES // (1024 * 1024)} MB"
            )

        mime = normalize_mime(content_type)
        if mime not in _ALLOWED_AUDIO_MIMES:
            raise ValidationError(
                f"Tipo de audio no soportado: {mime or '(desconocido)'}. "
                f"Permitidos: {', '.join(sorted(_ALLOWED_AUDIO_MIMES))}"
            )

        text = await self._stt.transcribe(
            data,
            mime=mime,
            language=self._env.stt_language,
        )
        normalized = text.strip()
        if not normalized:
            raise ValidationError("No se detectó habla en el audio")
        return normalized
