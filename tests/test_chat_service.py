"""Tests del servicio de chat."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from chatbot.application.services.chat_service import (
    ChatService,
    StreamCancelled,
    StreamDone,
    StreamToken,
)
from chatbot.domain.documents import DocumentChunk
from chatbot.domain.entities import Message, Role
from chatbot.domain.exceptions import ValidationError
from chatbot.domain.llm_stream import LLMDelta
from chatbot.domain.ports import LLMPort
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.llm.mock_adapter import MockLLMAdapter
from chatbot.infrastructure.adapters.persistence.memory_repository import (
    InMemoryConversationRepository,
)
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore
from tests.prompt_fixtures import default_prompt_repo


class SlowStreamingLLM(LLMPort):
    def __init__(self) -> None:
        self._model = "slow-model"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Message:
        return Message(role=Role.ASSISTANT, content="respuesta completa")

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        for word in ("uno", "dos", "tres", "cuatro"):
            yield LLMDelta(text=f"{word} ", kind="content")

    async def list_models(self) -> list[str]:
        return [self._model]

    async def health_check(self) -> bool:
        return True


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


@pytest.mark.asyncio
async def test_chat_stream_cancelled_saves_partial() -> None:
    service = ChatService(
        llm=SlowStreamingLLM(),
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(system="Bot\n\n{context}"),
    )
    token_count = 0

    async def is_cancelled() -> bool:
        nonlocal token_count
        return token_count >= 2

    events = []
    async for event in service.chat_stream(
        "Hola",
        is_cancelled=is_cancelled,
    ):
        events.append(event)
        if isinstance(event, StreamToken):
            token_count += 1

    assert isinstance(events[-1], StreamCancelled)
    tokens = [e.content for e in events if isinstance(e, StreamToken)]
    assert tokens == ["uno ", "dos "]

    conversation = await service.get_conversation(events[0].conversation_id)
    assert conversation.messages[0].content == "Hola"
    assert conversation.messages[1].content == "uno dos"
