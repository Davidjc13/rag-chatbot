"""Tests del retrieval RAG en el chat."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from chatbot.application.services.chat_service import ChatService
from chatbot.domain.documents import DocumentChunk
from chatbot.domain.entities import Message, Role
from chatbot.domain.exceptions import ConfigurationError
from chatbot.domain.llm_stream import LLMDelta
from chatbot.domain.ports import LLMPort
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.persistence.memory_repository import (
    InMemoryConversationRepository,
)
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore
from chatbot.infrastructure.adapters.persistence.routed_vector_store import RoutedVectorStore
from tests.prompt_fixtures import default_prompt_repo


class CapturingLLM(LLMPort):
    def __init__(self) -> None:
        self.last_system_prompt: str | None = None
        self.last_messages: list[Message] = []

    @property
    def model_name(self) -> str:
        return "capture-model"

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Message:
        self.last_system_prompt = system_prompt
        self.last_messages = list(messages)
        return Message(role=Role.ASSISTANT, content="ok")

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        self.last_system_prompt = system_prompt
        self.last_messages = list(messages)
        yield LLMDelta(text="ok", kind="content")

    async def list_models(self) -> list[str]:
        return [self.model_name]

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
        prompts=default_prompt_repo(
            system="Eres un bot.\n\n{context}",
            user_message="Pregunta:\n{question}",
        ),
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
    assert llm.last_messages[-1].content == "Pregunta:\n¿Cuál es el plazo de devolución?"


@pytest.mark.asyncio
async def test_chat_renders_question_template_but_stores_raw() -> None:
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
    repo = InMemoryConversationRepository()
    service = ChatService(
        llm=llm,
        repository=repo,
        prompts=default_prompt_repo(user_message="Q: {question}"),
        embeddings=embeddings,
        vector_store=store,
        rag_top_k=2,
    )
    reply = await service.chat("¿Plazo?")
    assert llm.last_messages[-1].content == "Q: ¿Plazo?"
    conversation = await service.get_conversation(reply.conversation_id)
    assert conversation.messages[0].content == "¿Plazo?"


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
        prompts=default_prompt_repo(),
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


@pytest.mark.asyncio
async def test_sources_dedupe_same_document() -> None:
    embeddings = MockEmbeddingAdapter()
    store = InMemoryVectorStore()
    vectors = await embeddings.embed(["chunk a sobre plazos", "chunk b sobre plazos"])
    await store.upsert(
        [
            DocumentChunk(
                document_id="doc-1",
                content="chunk a sobre plazos",
                metadata={"filename": "policy.docx", "format": "docx"},
                embedding=vectors[0],
            ),
            DocumentChunk(
                document_id="doc-1",
                content="chunk b sobre plazos",
                metadata={"filename": "policy.docx", "format": "docx"},
                embedding=vectors[1],
            ),
        ]
    )
    service = ChatService(
        llm=CapturingLLM(),
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(),
        embeddings=embeddings,
        vector_store=store,
        rag_top_k=4,
    )

    from chatbot.application.services.chat_service import StreamMeta

    events = [event async for event in service.chat_stream("¿Plazo?")]
    meta = next(e for e in events if isinstance(e, StreamMeta))
    assert len(meta.sources) == 1
    assert meta.sources[0]["id"] == "doc-1"


@pytest.mark.asyncio
async def test_chat_can_choose_neo4j_flow() -> None:
    embeddings = MockEmbeddingAdapter()
    postgres = InMemoryVectorStore()
    neo4j = InMemoryVectorStore()
    pg_vector = (await embeddings.embed(["contenido desde postgres"]))[0]
    neo_vector = (await embeddings.embed(["contenido desde neo4j"]))[0]
    await postgres.upsert(
        [
            DocumentChunk(
                document_id="doc-pg",
                content="contenido desde postgres",
                metadata={"filename": "pg.docx", "format": "docx"},
                embedding=pg_vector,
            )
        ]
    )
    await neo4j.upsert(
        [
            DocumentChunk(
                document_id="doc-neo",
                content="contenido desde neo4j",
                metadata={"filename": "neo.docx", "format": "docx"},
                embedding=neo_vector,
            )
        ]
    )
    llm = CapturingLLM()
    service = ChatService(
        llm=llm,
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(),
        embeddings=embeddings,
        vector_store=RoutedVectorStore(primary=postgres, neo4j=neo4j),
        rag_top_k=1,
    )

    await service.chat("contenido desde neo4j", retrieval_backend="neo4j")

    assert llm.last_system_prompt is not None
    assert "neo.docx" in llm.last_system_prompt
    assert "pg.docx" not in llm.last_system_prompt


@pytest.mark.asyncio
async def test_chat_rejects_unavailable_neo4j_flow() -> None:
    embeddings = MockEmbeddingAdapter()
    store = RoutedVectorStore(primary=InMemoryVectorStore(), neo4j=None)
    service = ChatService(
        llm=CapturingLLM(),
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(),
        embeddings=embeddings,
        vector_store=store,
        rag_top_k=1,
    )

    with pytest.raises(ConfigurationError):
        await service.chat("hola", retrieval_backend="neo4j")
