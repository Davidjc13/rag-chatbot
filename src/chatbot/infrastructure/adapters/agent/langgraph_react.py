"""Adaptador ReAct con LangGraph: search_documents como única herramienta."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from chatbot.application.services.rag_context import build_context, document_index_map
from chatbot.application.services.rag_retrieval import retrieve_chunks
from chatbot.domain.documents import RetrievedChunk
from chatbot.domain.entities import Message, Role
from chatbot.domain.exceptions import ConfigurationError, LLMProviderError
from chatbot.domain.ports import (
    AgentRagPort,
    AgentRagResult,
    AgentSourcesEvent,
    AgentStreamEvent,
    AgentTokenEvent,
    AgentToolEvent,
    EmbeddingPort,
    PromptRepositoryPort,
    VectorStorePort,
)
from chatbot.domain.prompts import PROMPT_AGENT_SYSTEM
from chatbot.infrastructure.adapters.agent.chat_model_factory import ChatModelFactory

_FALLBACK_AGENT_SYSTEM = (
    "Eres un asistente de documentos con acceso a search_documents. "
    "Usa la herramienta antes de afirmar hechos. Puedes buscar varias veces "
    "reformulando la consulta. Cita las fuentes con el formato indicado en "
    "los resultados. No inventes."
)
_OUTPUT_PREVIEW_CHARS = 280


@dataclass
class _RunState:
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    mapping: dict[str, tuple[int, str]] = field(default_factory=dict)
    retrieval_ms: int = 0


class LangGraphReactAgent(AgentRagPort):  # pylint: disable=too-many-instance-attributes
    """Agente ReAct (LangGraph) que busca en el almacén vectorial."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        embeddings: EmbeddingPort,
        vector_store: VectorStorePort,
        prompts: PromptRepositoryPort,
        rag_top_k: int,
        rag_min_score: float,
        max_steps: int,
        chat_model_factory: ChatModelFactory | None = None,
        chat_model: BaseChatModel | None = None,
        rag_hybrid: bool = False,
        rag_candidates: int = 20,
        rag_hybrid_alpha: float = 0.7,
    ) -> None:
        if chat_model_factory is None and chat_model is None:
            raise ConfigurationError(
                "El agente ReAct requiere chat_model_factory o chat_model"
            )
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._prompts = prompts
        self._rag_top_k = rag_top_k
        self._rag_min_score = rag_min_score
        self._max_steps = max(1, max_steps)
        self._chat_model_factory = chat_model_factory
        self._chat_model = chat_model
        self._rag_hybrid = rag_hybrid
        self._rag_candidates = rag_candidates
        self._rag_hybrid_alpha = rag_hybrid_alpha

    def _resolve_model(self, model: str | None) -> BaseChatModel:
        if self._chat_model is not None:
            return self._chat_model
        assert self._chat_model_factory is not None
        return self._chat_model_factory.create(model=model)

    async def _system_prompt(self) -> str:
        raw = await self._prompts.get(PROMPT_AGENT_SYSTEM) or _FALLBACK_AGENT_SYSTEM
        return raw.replace("{context}", "").strip()

    def _recursion_limit(self) -> int:
        return 2 * self._max_steps + 2

    def _make_search_tool(self, state: _RunState, retrieval_backend: str | None) -> StructuredTool:
        min_score = self._rag_min_score
        top_k = self._rag_top_k
        embeddings = self._embeddings
        vector_store = self._vector_store

        async def search_documents(query: str) -> str:
            """Busca fragmentos relevantes en los documentos indexados."""
            started = time.perf_counter()
            retrieved = await retrieve_chunks(
                query,
                embeddings=embeddings,
                vector_store=vector_store,
                retrieval_backend=retrieval_backend,
                top_k=top_k,
                hybrid=self._rag_hybrid,
                candidates=self._rag_candidates,
                hybrid_alpha=self._rag_hybrid_alpha,
            )
            state.retrieval_ms += int((time.perf_counter() - started) * 1000)
            if not retrieved:
                return (
                    "No se encontraron fragmentos. Reformula la consulta "
                    "e inténtalo de nuevo."
                )
            state.mapping = document_index_map(retrieved, existing=state.mapping)
            state.retrieved.extend(retrieved)
            context = build_context(retrieved, mapping=state.mapping)
            best = max(item.score for item in retrieved)
            if best < min_score:
                return (
                    f"Los fragmentos recuperados tienen baja relevancia "
                    f"(máx. {best:.3f}, umbral {min_score}). "
                    f"Reformula la consulta e inténtalo de nuevo.\n\n{context}"
                )
            return context

        return StructuredTool.from_function(
            coroutine=search_documents,
            name="search_documents",
            description=(
                "Busca fragmentos relevantes en los documentos indexados. "
                "Pasa una consulta en lenguaje natural; reformúlala si la "
                "búsqueda anterior no fue suficiente."
            ),
        )

    def _build_graph(self, model: BaseChatModel, tool: StructuredTool, system_prompt: str):
        return create_react_agent(model, [tool], prompt=system_prompt)

    async def run(  # pylint: disable=too-many-locals
        self,
        messages: list[Message],
        *,
        retrieval_backend: str | None = None,
        model: str | None = None,
    ) -> AgentRagResult:
        started = time.perf_counter()
        state = _RunState()
        chat_model = self._resolve_model(model)
        system_prompt = await self._system_prompt()
        tool = self._make_search_tool(state, retrieval_backend)
        graph = self._build_graph(chat_model, tool, system_prompt)
        try:
            result = await graph.ainvoke(
                {"messages": _to_lc_messages(messages)},
                {"recursion_limit": self._recursion_limit()},
            )
        except GraphRecursionError as exc:
            raise LLMProviderError(
                "El agente superó el número máximo de búsquedas.",
                provider="langgraph",
            ) from exc
        lc_messages = result.get("messages") or []
        text = _final_assistant_text(lc_messages)
        duration_ms = int((time.perf_counter() - started) * 1000)
        input_tokens, output_tokens = _usage_from_messages(lc_messages)
        return AgentRagResult(
            text=text,
            retrieved=tuple(state.retrieved),
            retrieval_duration_ms=state.retrieval_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )

    async def run_stream(  # pylint: disable=too-many-locals
        self,
        messages: list[Message],
        *,
        retrieval_backend: str | None = None,
        model: str | None = None,
        is_cancelled: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        started = time.perf_counter()
        state = _RunState()
        chat_model = self._resolve_model(model)
        system_prompt = await self._system_prompt()
        tool = self._make_search_tool(state, retrieval_backend)
        graph = self._build_graph(chat_model, tool, system_prompt)
        collected: list[BaseMessage] = []
        emitted_sources = 0

        try:
            async for event in graph.astream_events(
                {"messages": _to_lc_messages(messages)},
                version="v2",
                config={"recursion_limit": self._recursion_limit()},
            ):
                if is_cancelled is not None and await is_cancelled():
                    break
                kind = event.get("event")
                data = event.get("data") or {}
                if kind == "on_tool_start" and event.get("name") == "search_documents":
                    query = _tool_query(data.get("input"))
                    yield AgentToolEvent(
                        name="search_documents",
                        status="start",
                        query=query,
                    )
                    continue
                if kind == "on_tool_end" and event.get("name") == "search_documents":
                    output_text = _as_text(data.get("output"))
                    yield AgentToolEvent(
                        name="search_documents",
                        status="end",
                        query=_tool_query(data.get("input")),
                        output_preview=_preview(output_text),
                    )
                    if len(state.retrieved) > emitted_sources:
                        emitted_sources = len(state.retrieved)
                        yield AgentSourcesEvent(retrieved=tuple(state.retrieved))
                    continue
                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    for token in _tokens_from_chunk(chunk):
                        yield token
                    continue
                if kind == "on_chain_end" and event.get("name") == "LangGraph":
                    output = data.get("output") or {}
                    if isinstance(output, dict):
                        collected = list(output.get("messages") or [])
        except GraphRecursionError as exc:
            raise LLMProviderError(
                "El agente superó el número máximo de búsquedas.",
                provider="langgraph",
            ) from exc

        if not collected:
            # Fallback: el grafo puede no emitir on_chain_end LangGraph al cancelar.
            collected = []
        text = _final_assistant_text(collected)
        duration_ms = int((time.perf_counter() - started) * 1000)
        input_tokens, output_tokens = _usage_from_messages(collected)
        yield AgentRagResult(
            text=text,
            retrieved=tuple(state.retrieved),
            retrieval_duration_ms=state.retrieval_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )


def _to_lc_messages(messages: list[Message]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        if message.role == Role.USER:
            converted.append(HumanMessage(content=message.content))
        elif message.role == Role.ASSISTANT:
            converted.append(AIMessage(content=message.content))
    return converted


def _final_assistant_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        text = _as_text(message.content).strip()
        if text:
            return text
    return ""


def _usage_from_messages(
    messages: list[BaseMessage],
) -> tuple[int | None, int | None]:
    for message in reversed(messages):
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, dict):
            continue
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        in_val = int(input_tokens) if input_tokens is not None else None
        out_val = int(output_tokens) if output_tokens is not None else None
        if in_val is not None or out_val is not None:
            return in_val, out_val
    return None, None


def _tool_query(raw: object) -> str:
    if isinstance(raw, dict):
        value = raw.get("query") or raw.get("input") or ""
        return str(value)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(parsed, dict):
            return str(parsed.get("query") or parsed.get("input") or raw)
        return raw
    return ""


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _blocks_to_text(content, include_thinking=False)
    return str(value)


def _blocks_to_text(blocks: list[object], *, include_thinking: bool) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        if isinstance(block, dict):
            kind = str(block.get("type") or "")
            text = str(block.get("text") or block.get("content") or "")
            if kind in {"thinking", "reasoning"} and not include_thinking:
                continue
            parts.append(text)
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _preview(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= _OUTPUT_PREVIEW_CHARS:
        return stripped
    return stripped[:_OUTPUT_PREVIEW_CHARS].rstrip() + "…"


def _tokens_from_chunk(chunk: object) -> list[AgentTokenEvent]:
    events: list[AgentTokenEvent] = []
    if chunk is None:
        return events
    if getattr(chunk, "tool_call_chunks", None):
        return events
    additional = getattr(chunk, "additional_kwargs", None) or {}
    if isinstance(additional, dict):
        for key in ("reasoning_content", "thinking", "reasoning"):
            value = additional.get(key)
            if value:
                events.append(AgentTokenEvent(content=str(value), kind="thinking"))
    content = getattr(chunk, "content", None)
    if isinstance(content, list):
        thinking_blocks = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") in {"thinking", "reasoning"}
        ]
        thinking = _blocks_to_text(thinking_blocks, include_thinking=True)
        if thinking:
            events.append(AgentTokenEvent(content=thinking, kind="thinking"))
        text = _blocks_to_text(content, include_thinking=False)
        if text:
            events.append(AgentTokenEvent(content=text, kind="content"))
        return events
    if isinstance(content, str) and content:
        events.append(AgentTokenEvent(content=content, kind="content"))
    return events
