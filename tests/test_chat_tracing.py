"""Tests de trazas en ChatService."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from chatbot.application.services.chat_service import ChatService, StreamDone
from chatbot.domain.documents import DocumentChunk
from chatbot.domain.ports import ChatGenerationTrace, TracingPort
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.llm.mock_adapter import MockLLMAdapter
from chatbot.infrastructure.adapters.persistence.memory_repository import (
    InMemoryConversationRepository,
)
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore
from tests.prompt_fixtures import default_prompt_repo


@dataclass
class RecordingTracer(TracingPort):
    traces: list[ChatGenerationTrace] = field(default_factory=list)

    def record_chat_generation(self, trace: ChatGenerationTrace) -> None:
        self.traces.append(trace)


@pytest.fixture
async def traced_chat_service() -> tuple[ChatService, RecordingTracer]:
    embeddings = MockEmbeddingAdapter()
    store = InMemoryVectorStore()
    vectors = await embeddings.embed(["Las devoluciones se aceptan en 30 días."])
    chunk = DocumentChunk(
        id="chunk-abc",
        document_id="doc-1",
        content="Las devoluciones se aceptan en 30 días.",
        metadata={"filename": "policy.docx", "format": "docx"},
        embedding=vectors[0],
    )
    await store.upsert([chunk])
    tracer = RecordingTracer()
    service = ChatService(
        llm=MockLLMAdapter(model="test-model"),
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(system="Eres un bot de prueba.\n\n{context}"),
        embeddings=embeddings,
        vector_store=store,
        tracer=tracer,
        rag_top_k=2,
    )
    return service, tracer


@pytest.mark.asyncio
async def test_chat_stream_records_trace_after_done(
    traced_chat_service: tuple[ChatService, RecordingTracer],
) -> None:
    service, tracer = traced_chat_service
    events = [event async for event in service.chat_stream("¿Plazo de devolución?")]
    assert isinstance(events[-1], StreamDone)
    assert len(tracer.traces) == 1

    trace = tracer.traces[0]
    assert trace.user_query == "¿Plazo de devolución?"
    assert trace.chunk_ids == ("chunk-abc",)
    assert trace.model == "test-model"
    assert trace.input_tokens == 42
    assert trace.output_tokens is not None
    assert trace.duration_ms >= 0
    assert trace.retrieval_duration_ms >= 0
    assert trace.mode == "stream"
    assert trace.model_response


@pytest.mark.asyncio
async def test_chat_sync_records_trace(
    traced_chat_service: tuple[ChatService, RecordingTracer],
) -> None:
    service, tracer = traced_chat_service
    reply = await service.chat("¿Plazo de devolución?")
    assert len(tracer.traces) == 1

    trace = tracer.traces[0]
    assert trace.user_query == "¿Plazo de devolución?"
    assert trace.chunk_ids == ("chunk-abc",)
    assert trace.mode == "sync"
    assert trace.conversation_id == reply.conversation_id
    assert trace.model_response
