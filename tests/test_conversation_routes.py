"""Tests HTTP de listado y borrado de conversaciones."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from chatbot.application.services.chat_service import ChatService
from chatbot.infrastructure.adapters.api.exception_handlers import register_exception_handlers
from chatbot.infrastructure.adapters.api.routes import router
from chatbot.infrastructure.adapters.llm.mock_adapter import MockLLMAdapter
from chatbot.infrastructure.adapters.persistence.memory_repository import (
    InMemoryConversationRepository,
)
from tests.prompt_fixtures import default_prompt_repo


@pytest.fixture
def app() -> FastAPI:
    service = ChatService(
        llm=MockLLMAdapter(model="test-model"),
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(),
    )
    fastapi_app = FastAPI()
    register_exception_handlers(fastapi_app)
    fastapi_app.include_router(router, prefix="/api/v1")
    fastapi_app.state.chat_service = service
    return fastapi_app


@pytest.mark.asyncio
async def test_conversation_list_and_delete(app: FastAPI) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        chat = await client.post("/api/v1/chat", json={"message": "Hola titular"})
        assert chat.status_code == 200
        cid = chat.json()["conversation_id"]

        listed = await client.get("/api/v1/conversations")
        assert listed.status_code == 200
        ids = [item["id"] for item in listed.json()["conversations"]]
        assert cid in ids

        deleted = await client.delete(f"/api/v1/conversations/{cid}")
        assert deleted.status_code == 204

        missing = await client.get(f"/api/v1/conversations/{cid}")
        assert missing.status_code == 404
