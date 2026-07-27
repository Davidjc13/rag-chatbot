"""Tests del servicio de chat."""

from __future__ import annotations

from uuid import uuid4

import pytest

from chatbot.application.services.chat_service import ChatService
from chatbot.domain.documents import DocumentChunk
from chatbot.domain.exceptions import ValidationError
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.llm.mock_adapter import MockLLMAdapter
from chatbot.infrastructure.adapters.persistence.memory_repository import (
    InMemoryConversationRepository,
)
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore
from tests.prompt_fixtures import default_prompt_repo


@pytest.fixture
async def chat_service() -> ChatService:
    embeddings = MockEmbeddingAdapter()
    store = InMemoryVectorStore()
    vectors = await embeddings.embed(["Las devoluciones se aceptan en 30 días."])
    await store.upsert(
        [
            DocumentChunk(
                document_id="doc-1",
                content="Las devoluciones se aceptan en 30 días.",
                metadata={"filename": "policy.docx", "format": "docx"},
                embedding=vectors[0],
            )
        ]
    )
    return ChatService(
        llm=MockLLMAdapter(model="test-model"),
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(system="Eres un bot de prueba.\n\n{context}"),
        embeddings=embeddings,
        vector_store=store,
        rag_top_k=2,
    )


@pytest.mark.asyncio
async def test_chat_creates_conversation(chat_service: ChatService) -> None:
    reply = await chat_service.chat("Hola")
    assert reply.conversation_id
    assert "Hola" in reply.message.content
    assert reply.model == "test-model"


@pytest.mark.asyncio
async def test_chat_continues_conversation(chat_service: ChatService) -> None:
    first = await chat_service.chat("Uno")
    second = await chat_service.chat("Dos", conversation_id=first.conversation_id)
    assert second.conversation_id == first.conversation_id

    conversation = await chat_service.get_conversation(first.conversation_id)
    assert len(conversation.messages) == 4


@pytest.mark.asyncio
async def test_empty_message_raises(chat_service: ChatService) -> None:
    with pytest.raises(ValidationError):
        await chat_service.chat("   ")


@pytest.mark.asyncio
async def test_chat_accepts_client_generated_conversation_id(
    chat_service: ChatService,
) -> None:
    cid = str(uuid4())
    reply = await chat_service.chat("Hola", conversation_id=cid)
    assert reply.conversation_id == cid
    conversation = await chat_service.get_conversation(cid)
    assert conversation.messages[0].content == "Hola"


@pytest.mark.asyncio
async def test_invalid_conversation_id_raises(chat_service: ChatService) -> None:
    with pytest.raises(ValidationError):
        await chat_service.chat("Hola", conversation_id="no-es-uuid")


@pytest.mark.asyncio
async def test_persisted_history_keeps_raw_question(chat_service: ChatService) -> None:
    reply = await chat_service.chat("pregunta cruda")
    conversation = await chat_service.get_conversation(reply.conversation_id)
    assert conversation.messages[0].content == "pregunta cruda"
