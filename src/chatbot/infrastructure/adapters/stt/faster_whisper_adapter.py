"""Adaptador STT local con faster-whisper."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
from pathlib import Path

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


class FasterWhisperAdapter(TranscriptionPort):
    """Transcripción offline con faster-whisper (Whisper local)."""

    def __init__(self, *, model: str = "base") -> None:
        self._model_name = model
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return f"faster-whisper/{self._model_name}"

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel  # pylint: disable=import-outside-toplevel
            except ImportError as exc:
                raise TranscriptionUnavailableError(
                    "faster-whisper no está instalado",
                    provider="faster_whisper",
                ) from exc
            logger.info("Cargando modelo faster-whisper: %s", self._model_name)
            self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
            return self._model

    def _transcribe_sync(self, audio: bytes, *, mime: str, language: str) -> str:
        model = self._load_model()
        suffix = _MIME_TO_SUFFIX.get(mime, ".webm")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio)
            tmp_path = Path(tmp.name)
        try:
            segments, _info = model.transcribe(  # type: ignore[union-attr]
                str(tmp_path),
                language=language or None,
                vad_filter=True,
            )
            parts = [segment.text.strip() for segment in segments if segment.text.strip()]
            return " ".join(parts).strip()
        except TranscriptionUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            raise TranscriptionUnavailableError(
                f"Error al transcribir audio: {exc}",
                provider="faster_whisper",
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

    async def transcribe(self, audio: bytes, *, mime: str, language: str) -> str:
        return await asyncio.to_thread(
            self._transcribe_sync,
            audio,
            mime=mime,
            language=language,
        )
