"""Tests del retrieval RAG en el chat."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from chatbot.application.services.chat_service import ChatService
from chatbot.domain.documents import DocumentChunk
from chatbot.domain.entities import Message, Role
from chatbot.domain.ports import LLMPort
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.persistence.memory_repository import (
    InMemoryConversationRepository,
)
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore


class CapturingLLM(LLMPort):
    def __init__(self) -> None:
        self.last_system_prompt: str | None = None

    @property
    def model_name(self) -> str:
        return "capture-model"

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> Message:
        self.last_system_prompt = system_prompt
        return Message(role=Role.ASSISTANT, content="ok")

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        self.last_system_prompt = system_prompt
        yield "ok"

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_chat_adds_retrieved_chunks_to_system_prompt() -> None:
    embeddings = MockEmbeddingAdapter()
    store = InMemoryVectorStore()
    content = "Las devoluciones se aceptan en 30 días."
    vectors = await embeddings.embed([content])
    await store.upsert(
        [
            DocumentChunk(
                document_id="doc-1",
                content=content,
                metadata={"filename": "policy.docx", "format": "docx"},
                embedding=vectors[0],
            )
        ]
    )
    llm = CapturingLLM()
    service = ChatService(
        llm=llm,
        repository=InMemoryConversationRepository(),
        system_prompt="Eres un bot.",
        embeddings=embeddings,
        vector_store=store,
        rag_top_k=2,
    )

    await service.chat("¿Cuál es el plazo de devolución?")
    assert llm.last_system_prompt is not None
    assert "30 días" in llm.last_system_prompt
    assert "policy.docx" in llm.last_system_prompt
    assert "Fragmento" in llm.last_system_prompt
    assert "<index = 1, source=rag, title=policy.docx, id=doc-1>" in llm.last_system_prompt
    assert "cita=" in llm.last_system_prompt


@pytest.mark.asyncio
async def test_chat_stream_meta_includes_sources() -> None:
    embeddings = MockEmbeddingAdapter()
    store = InMemoryVectorStore()
    content = "Las devoluciones se aceptan en 30 días."
    vectors = await embeddings.embed([content])
    await store.upsert(
        [
            DocumentChunk(
                document_id="doc-1",
                content=content,
                metadata={"filename": "policy.docx", "format": "docx"},
                embedding=vectors[0],
            )
        ]
    )
    service = ChatService(
        llm=CapturingLLM(),
        repository=InMemoryConversationRepository(),
        system_prompt="Eres un bot.",
        embeddings=embeddings,
        vector_store=store,
        rag_top_k=2,
    )

    from chatbot.application.services.chat_service import StreamMeta

    events = [event async for event in service.chat_stream("¿Plazo?")]
    meta = next(e for e in events if isinstance(e, StreamMeta))
    assert meta.sources
    assert meta.sources[0]["title"] == "policy.docx"
    assert meta.sources[0]["id"] == "doc-1"
    assert meta.sources[0]["index"] == 1
    assert meta.sources[0]["source"] == "rag"
