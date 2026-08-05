"""Tests del endpoint GET /api/v1/models."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from chatbot.domain.ports import LLMPort
from chatbot.infrastructure.adapters.api.exception_handlers import register_exception_handlers
from chatbot.infrastructure.adapters.api.routes import router


class StubLLM(LLMPort):
    @property
    def model_name(self) -> str:
        return "qwen3:4b"

    async def generate(self, messages, *, system_prompt=None, model=None):
        raise NotImplementedError

    async def generate_stream(self, messages, *, system_prompt=None, model=None):
        raise NotImplementedError
        yield  # pragma: no cover

    async def list_models(self) -> list[str]:
        return ["qwen3:4b", "nomic-embed-text"]

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def app() -> FastAPI:
    fastapi_app = FastAPI()
    register_exception_handlers(fastapi_app)
    fastapi_app.include_router(router, prefix="/api/v1")
    fastapi_app.state.llm = StubLLM()
    return fastapi_app


@pytest.mark.asyncio
async def test_list_models_route(app: FastAPI) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": ["qwen3:4b", "nomic-embed-text"],
        "active": "qwen3:4b",
    }
