"""Tests de la ruta POST /api/v1/transcribe."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from chatbot.application.services.transcription_service import TranscriptionService
from chatbot.core.env import Env
from chatbot.domain.ports import TranscriptionPort
from chatbot.infrastructure.adapters.api.exception_handlers import register_exception_handlers
from chatbot.infrastructure.adapters.api.routes import router


class MockTranscriptionPort(TranscriptionPort):
    @property
    def model_name(self) -> str:
        return "mock-stt"

    async def transcribe(self, audio: bytes, *, mime: str, language: str) -> str:
        assert audio == b"fake-audio"
        assert mime == "audio/webm"
        return "texto transcrito"


@pytest.fixture(autouse=True)
def _reset_env() -> None:
    Env.reset()
    yield
    Env.reset()


@pytest.fixture
def app() -> FastAPI:
    fastapi_app = FastAPI()
    register_exception_handlers(fastapi_app)
    fastapi_app.include_router(router, prefix="/api/v1")
    fastapi_app.state.transcription_service = TranscriptionService(
        stt=MockTranscriptionPort(),
        env=Env.get_instance(),
    )
    return fastapi_app


@pytest.mark.asyncio
async def test_transcribe_route_returns_text(app: FastAPI) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("voice.webm", b"fake-audio", "audio/webm")}
        response = await client.post("/api/v1/transcribe", files=files)

    assert response.status_code == 200
    assert response.json() == {"text": "texto transcrito"}
