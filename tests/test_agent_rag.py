"""Tests del modo RAG agéntico (ReAct + LangGraph)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from langchain_core.messages import AIMessage, ToolCall

from chatbot.application.services.chat_service import (
    ChatService,
    StreamMeta,
    StreamToken,
    StreamTool,
)
from chatbot.application.services.guardrails import RuleBasedGuardrail
from chatbot.domain.documents import DocumentChunk
from chatbot.domain.entities import Message, Role
from chatbot.domain.exceptions import ConfigurationError
from chatbot.domain.llm_stream import LLMDelta
from chatbot.domain.ports import LLMPort
from chatbot.infrastructure.adapters.agent.chat_model_factory import BindableFakeChatModel
from chatbot.infrastructure.adapters.agent.langgraph_react import LangGraphReactAgent
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.llm.mock_adapter import MockLLMAdapter
from chatbot.infrastructure.adapters.persistence.memory_repository import (
    InMemoryConversationRepository,
)
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore
from tests.prompt_fixtures import default_prompt_repo

_DOC_CONTENT = "Las devoluciones se aceptan en 30 días."


class RecordingEmbeddings(MockEmbeddingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.embedded: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return await super().embed(texts)


class CapturingLLM(LLMPort):
    def __init__(self) -> None:
        self.last_system_prompt: str | None = None
        self.calls = 0

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
        self.calls += 1
        self.last_system_prompt = system_prompt
        return Message(role=Role.ASSISTANT, content="ok-rag")

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        self.calls += 1
        self.last_system_prompt = system_prompt
        yield LLMDelta(text="ok-rag", kind="content")

    async def list_models(self) -> list[str]:
        return [self.model_name]

    async def health_check(self) -> bool:
        return True


def _scripted_model(*messages: AIMessage) -> BindableFakeChatModel:
    return BindableFakeChatModel(responses=list(messages))


async def _seed_store(store: InMemoryVectorStore, embeddings: MockEmbeddingAdapter) -> None:
    vectors = await embeddings.embed([_DOC_CONTENT])
    await store.upsert(
        [
            DocumentChunk(
                document_id="doc-1",
                content=_DOC_CONTENT,
                metadata={"filename": "policy.docx", "format": "docx"},
                embedding=vectors[0],
            )
        ]
    )


def _service(
    *,
    store: InMemoryVectorStore,
    embeddings: MockEmbeddingAdapter,
    chat_model: BindableFakeChatModel,
    llm: LLMPort | None = None,
    guardrails: RuleBasedGuardrail | None = None,
    rag_min_score: float = 0.25,
    rag_top_k: int = 4,
) -> ChatService:
    agent = LangGraphReactAgent(
        embeddings=embeddings,
        vector_store=store,
        prompts=default_prompt_repo(),
        rag_top_k=rag_top_k,
        rag_min_score=rag_min_score,
        max_steps=4,
        chat_model=chat_model,
    )
    return ChatService(
        llm=llm or MockLLMAdapter(model="classic-unused"),
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(),
        embeddings=embeddings,
        vector_store=store,
        guardrails=guardrails,
        agent=agent,
        rag_top_k=rag_top_k,
    )


@pytest.mark.asyncio
async def test_agent_searches_then_answers() -> None:
    embeddings = RecordingEmbeddings()
    store = InMemoryVectorStore()
    await _seed_store(store, embeddings)
    embeddings.embedded.clear()

    model = _scripted_model(
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="search_documents",
                    args={"query": "plazo de devolución"},
                    id="call-1",
                )
            ],
        ),
        AIMessage(
            content=(
                "El plazo es de 30 días "
                "<index = 1, source=rag, title=policy.docx, id=doc-1>."
            )
        ),
    )
    service = _service(store=store, embeddings=embeddings, chat_model=model)

    reply = await service.chat("¿Cuál es el plazo de devolución?", mode="agent")
    assert "30 días" in reply.message.content
    assert "plazo de devolución" in embeddings.embedded
    conversation = await service.get_conversation(reply.conversation_id)
    roles = [msg.role for msg in conversation.messages]
    assert roles == [Role.USER, Role.ASSISTANT]


@pytest.mark.asyncio
async def test_agent_stream_emits_tool_and_sources() -> None:
    embeddings = MockEmbeddingAdapter()
    store = InMemoryVectorStore()
    await _seed_store(store, embeddings)
    model = _scripted_model(
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="search_documents",
                    args={"query": "devoluciones"},
                    id="call-1",
                )
            ],
        ),
        AIMessage(content="Se aceptan en 30 días."),
    )
    service = _service(store=store, embeddings=embeddings, chat_model=model)

    events = [
        event
        async for event in service.chat_stream("¿Plazo?", mode="agent")
    ]
    tools = [e for e in events if isinstance(e, StreamTool)]
    assert any(e.status == "start" and e.query == "devoluciones" for e in tools)
    assert any(e.status == "end" for e in tools)
    metas = [e for e in events if isinstance(e, StreamMeta)]
    sourced = [e for e in metas if e.sources]
    assert sourced
    assert sourced[-1].sources[0]["id"] == "doc-1"
    tokens = "".join(e.content for e in events if isinstance(e, StreamToken))
    assert "30 días" in tokens


@pytest.mark.asyncio
async def test_agent_reformulates_when_below_threshold() -> None:
    embeddings = RecordingEmbeddings()
    store = InMemoryVectorStore()
    await _seed_store(store, embeddings)
    embeddings.embedded.clear()

    model = _scripted_model(
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="search_documents",
                    args={"query": "zzzz-no-match"},
                    id="call-1",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="search_documents",
                    args={"query": _DOC_CONTENT},
                    id="call-2",
                )
            ],
        ),
        AIMessage(content="Tras reformular: 30 días."),
    )
    service = _service(
        store=store,
        embeddings=embeddings,
        chat_model=model,
        rag_min_score=0.9,
    )

    reply = await service.chat("¿Plazo?", mode="agent")
    assert "30 días" in reply.message.content
    assert embeddings.embedded == ["zzzz-no-match", _DOC_CONTENT]


@pytest.mark.asyncio
async def test_agent_out_of_scope_when_no_chunks() -> None:
    embeddings = MockEmbeddingAdapter()
    store = InMemoryVectorStore()
    model = _scripted_model(
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="search_documents",
                    args={"query": "cualquier cosa"},
                    id="call-1",
                )
            ],
        ),
        AIMessage(content="Inventaría una respuesta."),
    )
    service = _service(
        store=store,
        embeddings=embeddings,
        chat_model=model,
        guardrails=RuleBasedGuardrail(min_score=0.25),
    )

    reply = await service.chat("¿Hay algo?", mode="agent")
    assert "Inventaría" not in reply.message.content
    assert "documentos indexados" in reply.message.content


@pytest.mark.asyncio
async def test_rag_mode_still_retrieves_once_when_agent_wired() -> None:
    embeddings = RecordingEmbeddings()
    store = InMemoryVectorStore()
    await _seed_store(store, embeddings)
    embeddings.embedded.clear()
    llm = CapturingLLM()
    unused_agent_model = _scripted_model(
        AIMessage(content="esto no debería usarse"),
    )
    service = _service(
        store=store,
        embeddings=embeddings,
        chat_model=unused_agent_model,
        llm=llm,
    )

    reply = await service.chat("¿Cuál es el plazo de devolución?", mode="rag")
    assert reply.message.content == "ok-rag"
    assert llm.calls == 1
    assert llm.last_system_prompt is not None
    assert "30 días" in llm.last_system_prompt
    assert embeddings.embedded == ["¿Cuál es el plazo de devolución?"]
    assert unused_agent_model.i == 0


@pytest.mark.asyncio
async def test_agent_mode_requires_port() -> None:
    service = ChatService(
        llm=MockLLMAdapter(),
        repository=InMemoryConversationRepository(),
        prompts=default_prompt_repo(),
    )
    with pytest.raises(ConfigurationError):
        await service.chat("hola", mode="agent")
