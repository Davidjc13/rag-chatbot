"""Tests del servicio de transcripción."""

from __future__ import annotations

import pytest

from chatbot.application.services.transcription_service import TranscriptionService
from chatbot.core.env import Env
from chatbot.domain.exceptions import TranscriptionUnavailableError, ValidationError
from chatbot.domain.ports import TranscriptionPort


class MockTranscriptionPort(TranscriptionPort):
    def __init__(self, *, text: str = "hola mundo") -> None:
        self._text = text
        self.last_audio: bytes | None = None
        self.last_mime: str | None = None
        self.last_language: str | None = None

    @property
    def model_name(self) -> str:
        return "mock-stt"

    async def transcribe(self, audio: bytes, *, mime: str, language: str) -> str:
        self.last_audio = audio
        self.last_mime = mime
        self.last_language = language
        return self._text


@pytest.fixture(autouse=True)
def _reset_env() -> None:
    Env.reset()
    yield
    Env.reset()


@pytest.mark.asyncio
async def test_transcribe_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STT_ENABLED", "true")
    monkeypatch.setenv("STT_LANGUAGE", "es")
    stt = MockTranscriptionPort(text="  pregunta de prueba  ")
    service = TranscriptionService(stt=stt, env=Env.get_instance())

    text = await service.transcribe(data=b"audio-bytes", content_type="audio/webm")

    assert text == "pregunta de prueba"
    assert stt.last_audio == b"audio-bytes"
    assert stt.last_mime == "audio/webm"
    assert stt.last_language == "es"


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_audio() -> None:
    service = TranscriptionService(stt=MockTranscriptionPort(), env=Env.get_instance())

    with pytest.raises(ValidationError, match="vacío"):
        await service.transcribe(data=b"", content_type="audio/webm")


@pytest.mark.asyncio
async def test_transcribe_rejects_unsupported_mime() -> None:
    service = TranscriptionService(stt=MockTranscriptionPort(), env=Env.get_instance())

    with pytest.raises(ValidationError, match="no soportado"):
        await service.transcribe(data=b"1234", content_type="video/mp4")


@pytest.mark.asyncio
async def test_transcribe_rejects_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STT_ENABLED", "false")
    service = TranscriptionService(stt=MockTranscriptionPort(), env=Env.get_instance())

    with pytest.raises(TranscriptionUnavailableError, match="deshabilitada"):
        await service.transcribe(data=b"1234", content_type="audio/webm")


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_result() -> None:
    service = TranscriptionService(stt=MockTranscriptionPort(text="   "), env=Env.get_instance())

    with pytest.raises(ValidationError, match="No se detectó habla"):
        await service.transcribe(data=b"1234", content_type="audio/wav")
