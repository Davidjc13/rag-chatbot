"""Tests de guardarraíles y chat en streaming."""

from __future__ import annotations

import pytest

from chatbot.application.services.chat_service import (
    ChatService,
    StreamDone,
    StreamMeta,
    StreamToken,
)
from chatbot.application.services.guardrails import RuleBasedGuardrail
from chatbot.domain.documents import DocumentChunk
from chatbot.domain.exceptions import GuardrailBlockedError
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.llm.mock_adapter import MockLLMAdapter
from chatbot.infrastructure.adapters.persistence.memory_repository import (
    InMemoryConversationRepository,
)
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore
from tests.prompt_fixtures import default_prompt_repo


def test_guardrail_blocks_toxic_input() -> None:
    guard = RuleBasedGuardrail(min_score=0.1)
    with pytest.raises(GuardrailBlockedError) as exc:
        guard.check_input("Eres un gilipollas")
    assert exc.value.reason == "toxic_input"


def test_guardrail_scope_requires_min_score() -> None:
    guard = RuleBasedGuardrail(min_score=0.5)
    assert guard.is_in_scope([]) is False
    assert guard.is_in_scope([0.2, 0.4]) is False
    assert guard.is_in_scope([0.2, 0.6]) is True


@pytest.fixture
async def rag_chat_service() -> ChatService:
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
    return ChatService(
        llm=MockLLMAdapter(model="stream-mock"),
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(system="Eres un bot de prueba.\n\n{context}"),
        embeddings=embeddings,
        vector_store=store,
        guardrails=RuleBasedGuardrail(min_score=0.01),
        rag_top_k=2,
    )


@pytest.mark.asyncio
async def test_chat_blocks_toxic_before_llm(rag_chat_service: ChatService) -> None:
    with pytest.raises(GuardrailBlockedError):
        await rag_chat_service.chat("Qué puto desastre")


@pytest.mark.asyncio
async def test_chat_out_of_scope_when_no_documents() -> None:
    service = ChatService(
        llm=MockLLMAdapter(),
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(system="Bot\n\n{context}"),
        embeddings=MockEmbeddingAdapter(),
        vector_store=InMemoryVectorStore(),
        guardrails=RuleBasedGuardrail(min_score=0.25),
    )
    reply = await service.chat("¿Qué hora es en Tokio?")
    assert "documentos indexados" in reply.message.content.lower()


@pytest.mark.asyncio
async def test_chat_stream_yields_tokens(rag_chat_service: ChatService) -> None:
    events = [
        event
        async for event in rag_chat_service.chat_stream("¿Plazo de devolución?")
    ]
    assert isinstance(events[0], StreamMeta)
    assert any(isinstance(e, StreamToken) for e in events)
    assert isinstance(events[-1], StreamDone)
    text = "".join(e.content for e in events if isinstance(e, StreamToken))
    assert "devolución" in text.lower() or "mock" in text.lower()


@pytest.mark.asyncio
async def test_mock_generate_stream_chunks() -> None:
    adapter = MockLLMAdapter(model="m")
    from chatbot.domain.entities import Message, Role

    chunks = [
        chunk
        async for chunk in adapter.generate_stream(
            [Message(role=Role.USER, content="hola")]
        )
    ]
    assert len(chunks) >= 2
    assert "hola" in "".join(chunk.text for chunk in chunks)
