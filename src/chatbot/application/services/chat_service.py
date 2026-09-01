"""Casos de uso de chat con retrieval RAG y streaming."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from chatbot.application.services.rag_context import (
    build_context,
    sources_payload,
)
from chatbot.application.services.rag_retrieval import retrieve_chunks
from chatbot.domain.documents import RetrievedChunk
from chatbot.domain.entities import ChatReply, Conversation, Message, Role
from chatbot.domain.exceptions import (
    ConfigurationError,
    ConversationNotFoundError,
    ValidationError,
)
from chatbot.domain.ports import (
    AgentRagPort,
    AgentRagResult,
    AgentSourcesEvent,
    AgentTokenEvent,
    AgentToolEvent,
    ChatGenerationTrace,
    ConversationRepositoryPort,
    EmbeddingPort,
    GuardrailPort,
    LLMPort,
    PromptRepositoryPort,
    TracingPort,
    VectorStorePort,
)
from chatbot.domain.prompts import (
    PROMPT_STUDY_TUTOR_SYSTEM,
    PROMPT_SYSTEM,
    PROMPT_USER_MESSAGE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StreamMeta:
    conversation_id: str
    model: str
    sources: tuple[dict[str, str | int], ...] = ()


@dataclass(frozen=True, slots=True)
class StreamToken:
    content: str


@dataclass(frozen=True, slots=True)
class StreamThinking:
    content: str


@dataclass(frozen=True, slots=True)
class StreamDone:
    conversation_id: str


@dataclass(frozen=True, slots=True)
class StreamCancelled:
    conversation_id: str


@dataclass(frozen=True, slots=True)
class StreamTool:
    name: str
    status: Literal["start", "end"]
    query: str
    output_preview: str = ""


StreamEvent = (
    StreamMeta | StreamToken | StreamThinking | StreamTool | StreamDone | StreamCancelled
)
ChatMode = Literal["rag", "study", "agent"]

_FALLBACK_SYSTEM = (
    "Eres un asistente de documentos. Responde solo con el contexto RAG. "
    "Si razonas internamente, hazlo de forma breve (pocas frases) y luego "
    "da la respuesta final al usuario.\n\n{context}"
)
_FALLBACK_USER = "{question}"
_STUDY_TOP_K = 8


class ChatService:  # pylint: disable=too-many-instance-attributes
    """Servicio de aplicación: orquesta conversación + contexto RAG."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        llm: LLMPort,
        repository: ConversationRepositoryPort,
        prompts: PromptRepositoryPort,
        *,
        embeddings: EmbeddingPort | None = None,
        vector_store: VectorStorePort | None = None,
        guardrails: GuardrailPort | None = None,
        tracer: TracingPort | None = None,
        agent: AgentRagPort | None = None,
        rag_top_k: int = 4,
    ) -> None:
        self._llm = llm
        self._repository = repository
        self._prompts = prompts
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._guardrails = guardrails
        self._tracer = tracer
        self._agent = agent
        self._rag_top_k = rag_top_k

    def _resolve_model(self, model: str | None) -> str:
        return (model or "").strip() or self._llm.model_name

    async def chat(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        user_message: str,
        *,
        conversation_id: str | None = None,
        retrieval_backend: str | None = None,
        model: str | None = None,
        mode: ChatMode = "rag",
        document_id: str | None = None,
    ) -> ChatReply:
        content = (user_message or "").strip()
        if not content:
            raise ValidationError("El mensaje del usuario no puede estar vacío")

        selected_model = self._resolve_model(model)

        if self._guardrails is not None:
            self._guardrails.check_input(content)

        conversation = await self._resolve_conversation(conversation_id)
        conversation.add_message(Message(role=Role.USER, content=content))

        if mode == "agent":
            return await self._chat_with_agent(
                conversation,
                content,
                retrieval_backend=retrieval_backend,
                model=model,
                selected_model=selected_model,
            )

        retrieved, retrieval_duration_ms = await self._retrieve_timed(
            content,
            retrieval_backend=retrieval_backend,
            document_id=document_id if mode == "study" else None,
            top_k=_STUDY_TOP_K if mode == "study" else self._rag_top_k,
        )
        if mode == "study" and document_id and not retrieved and self._vector_store:
            chunks = await self._vector_store.get_chunks_by_document(document_id)
            retrieved = [
                RetrievedChunk(chunk=chunk, score=1.0) for chunk in chunks[:_STUDY_TOP_K]
            ]
        if self._guardrails is not None and not self._is_in_scope(
            retrieved,
            mode=mode,
            document_id=document_id,
        ):
            refusal = self._guardrails.out_of_scope_message
            assistant_message = Message(role=Role.ASSISTANT, content=refusal)
            conversation.add_message(assistant_message)
            await self._repository.save(conversation)
            self._record_chat_trace(
                user_query=content,
                retrieved=retrieved,
                model_response=refusal,
                conversation_id=conversation.id,
                duration_ms=0,
                retrieval_duration_ms=retrieval_duration_ms,
                mode="sync",
                retrieval_backend=retrieval_backend,
            )
            return ChatReply(
                conversation_id=conversation.id,
                message=assistant_message,
                model=selected_model,
            )

        system_prompt, llm_messages = await self._build_llm_payload(
            conversation, retrieved, mode=mode
        )

        logger.info(
            "Generando respuesta",
            extra={
                "conversation_id": conversation.id,
                "model": selected_model,
                "history_size": len(conversation.messages),
                "rag_chunks": len(retrieved),
            },
        )

        started = time.perf_counter()
        assistant_message = await self._llm.generate(
            llm_messages,
            system_prompt=system_prompt,
            model=model,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        if self._guardrails is not None:
            self._guardrails.check_output(assistant_message.content)

        conversation.add_message(assistant_message)
        await self._repository.save(conversation)

        logger.info(
            "Respuesta generada",
            extra={"conversation_id": conversation.id, "model": selected_model},
        )

        self._record_chat_trace(
            user_query=content,
            retrieved=retrieved,
            model_response=assistant_message.content,
            conversation_id=conversation.id,
            duration_ms=duration_ms,
            retrieval_duration_ms=retrieval_duration_ms,
            mode="sync",
            retrieval_backend=retrieval_backend,
        )

        return ChatReply(
            conversation_id=conversation.id,
            message=assistant_message,
            model=selected_model,
        )

    async def chat_stream(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
        self,
        user_message: str,
        *,
        conversation_id: str | None = None,
        retrieval_backend: str | None = None,
        model: str | None = None,
        mode: ChatMode = "rag",
        document_id: str | None = None,
        is_cancelled: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        content = (user_message or "").strip()
        if not content:
            raise ValidationError("El mensaje del usuario no puede estar vacío")

        selected_model = self._resolve_model(model)

        if self._guardrails is not None:
            self._guardrails.check_input(content)

        conversation = await self._resolve_conversation(conversation_id)
        conversation.add_message(Message(role=Role.USER, content=content))

        if mode == "agent":
            async for event in self._chat_stream_agent(
                conversation,
                content,
                retrieval_backend=retrieval_backend,
                model=model,
                selected_model=selected_model,
                is_cancelled=is_cancelled,
            ):
                yield event
            return

        retrieved, retrieval_duration_ms = await self._retrieve_timed(
            content,
            retrieval_backend=retrieval_backend,
            document_id=document_id if mode == "study" else None,
            top_k=_STUDY_TOP_K if mode == "study" else self._rag_top_k,
        )
        sources = self._sources_payload(retrieved)
        yield StreamMeta(
            conversation_id=conversation.id,
            model=selected_model,
            sources=sources,
        )

        if self._guardrails is not None and not self._is_in_scope(
            retrieved,
            mode=mode,
            document_id=document_id,
        ):
            refusal = self._guardrails.out_of_scope_message
            yield StreamToken(content=refusal)
            conversation.add_message(Message(role=Role.ASSISTANT, content=refusal))
            await self._repository.save(conversation)
            self._record_chat_trace(
                user_query=content,
                retrieved=retrieved,
                model_response=refusal,
                conversation_id=conversation.id,
                duration_ms=0,
                retrieval_duration_ms=retrieval_duration_ms,
                mode="stream",
                retrieval_backend=retrieval_backend,
            )
            yield StreamDone(conversation_id=conversation.id)
            return

        system_prompt, llm_messages = await self._build_llm_payload(
            conversation, retrieved, mode=mode
        )
        logger.info(
            "Generando respuesta (stream)",
            extra={
                "conversation_id": conversation.id,
                "model": selected_model,
                "history_size": len(conversation.messages),
                "rag_chunks": len(retrieved),
            },
        )

        parts: list[str] = []
        input_tokens: int | None = None
        output_tokens: int | None = None
        provider_duration_ms: int | None = None
        started = time.perf_counter()
        cancelled = False
        async for delta in self._llm.generate_stream(
            llm_messages,
            system_prompt=system_prompt,
            model=model,
        ):
            if is_cancelled is not None and await is_cancelled():
                cancelled = True
                break
            if delta.input_tokens is not None:
                input_tokens = delta.input_tokens
            if delta.output_tokens is not None:
                output_tokens = delta.output_tokens
            if delta.duration_ms is not None:
                provider_duration_ms = delta.duration_ms
            if not delta.text:
                continue
            if delta.kind == "thinking":
                yield StreamThinking(content=delta.text)
                continue
            parts.append(delta.text)
            yield StreamToken(content=delta.text)

        duration_ms = (
            provider_duration_ms
            if provider_duration_ms is not None
            else int((time.perf_counter() - started) * 1000)
        )
        full_text = "".join(parts).strip()

        if cancelled:
            if full_text:
                conversation.add_message(Message(role=Role.ASSISTANT, content=full_text))
            await self._repository.save(conversation)
            if full_text:
                self._record_chat_trace(
                    user_query=content,
                    retrieved=retrieved,
                    model_response=full_text,
                    conversation_id=conversation.id,
                    duration_ms=duration_ms,
                    retrieval_duration_ms=retrieval_duration_ms,
                    mode="stream",
                    retrieval_backend=retrieval_backend,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            yield StreamCancelled(conversation_id=conversation.id)
            return

        if not full_text:
            raise ValidationError("El modelo devolvió una respuesta vacía")

        if self._guardrails is not None:
            self._guardrails.check_output(full_text)

        conversation.add_message(Message(role=Role.ASSISTANT, content=full_text))
        await self._repository.save(conversation)
        self._record_chat_trace(
            user_query=content,
            retrieved=retrieved,
            model_response=full_text,
            conversation_id=conversation.id,
            duration_ms=duration_ms,
            retrieval_duration_ms=retrieval_duration_ms,
            mode="stream",
            retrieval_backend=retrieval_backend,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        yield StreamDone(conversation_id=conversation.id)

    async def get_conversation(self, conversation_id: str) -> Conversation:
        conversation = await self._repository.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    async def _resolve_conversation(self, conversation_id: str | None) -> Conversation:
        if conversation_id is None or not conversation_id.strip():
            return Conversation()

        cid = conversation_id.strip()
        try:
            UUID(cid)
        except ValueError as exc:
            raise ValidationError(
                "conversation_id debe ser un UUID válido generado por el cliente"
            ) from exc

        conversation = await self._repository.get(cid)
        if conversation is None:
            # El frontend genera el id; si aún no existe en BD, se crea.
            return Conversation(id=cid)
        return conversation

    async def _retrieve(
        self,
        query: str,
        *,
        retrieval_backend: str | None = None,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        effective_top_k = top_k if top_k is not None else self._rag_top_k
        return await retrieve_chunks(
            query,
            embeddings=self._embeddings,
            vector_store=self._vector_store,
            retrieval_backend=retrieval_backend,
            top_k=effective_top_k,
            document_id=document_id,
        )

    async def _retrieve_timed(
        self,
        query: str,
        *,
        retrieval_backend: str | None = None,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], int]:
        started = time.perf_counter()
        retrieved = await self._retrieve(
            query,
            retrieval_backend=retrieval_backend,
            document_id=document_id,
            top_k=top_k,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        return retrieved, duration_ms

    def _is_in_scope(
        self,
        retrieved: list[RetrievedChunk],
        *,
        mode: ChatMode,
        document_id: str | None,
    ) -> bool:
        if mode == "study" and document_id:
            return bool(retrieved) or self._guardrails is None
        if self._guardrails is None:
            return True
        return self._guardrails.is_in_scope([item.score for item in retrieved])

    async def _build_llm_payload(
        self,
        conversation: Conversation,
        retrieved: list[RetrievedChunk],
        *,
        mode: ChatMode = "rag",
    ) -> tuple[str, list[Message]]:
        context = build_context(retrieved)
        if mode == "study":
            system_template = (
                await self._prompts.get(PROMPT_STUDY_TUTOR_SYSTEM)
                or _FALLBACK_SYSTEM
            )
        else:
            system_template = await self._prompts.get(PROMPT_SYSTEM) or _FALLBACK_SYSTEM
        user_template = await self._prompts.get(PROMPT_USER_MESSAGE) or _FALLBACK_USER

        system_prompt = system_template.replace("{context}", context)
        if mode != "study" and "razonamiento breve" not in system_prompt.lower():
            system_prompt = (
                system_prompt.rstrip()
                + "\n\nSi razonas internamente antes de responder, "
                "hazlo de forma breve (pocas frases) y luego escribe la respuesta."
            )
        question = ""
        if conversation.messages and conversation.messages[-1].role == Role.USER:
            question = conversation.messages[-1].content
        user_rendered = user_template.replace("{question}", question)

        history = conversation.history()
        if history and history[-1].role == Role.USER:
            history[-1] = Message(
                role=Role.USER,
                content=user_rendered,
                created_at=history[-1].created_at,
            )
        return system_prompt, history

    @staticmethod
    def _sources_payload(
        retrieved: list[RetrievedChunk],
    ) -> tuple[dict[str, str | int], ...]:
        return sources_payload(retrieved)

    async def _chat_with_agent(  # pylint: disable=too-many-arguments
        self,
        conversation: Conversation,
        content: str,
        *,
        retrieval_backend: str | None,
        model: str | None,
        selected_model: str,
    ) -> ChatReply:
        if self._agent is None:
            raise ConfigurationError("El modo agente no está disponible")

        started = time.perf_counter()
        result = await self._agent.run(
            list(conversation.messages),
            retrieval_backend=retrieval_backend,
            model=model,
        )
        duration_ms = result.duration_ms or int((time.perf_counter() - started) * 1000)
        retrieved = list(result.retrieved)
        reply_text, used_retrieved = self._agent_final_text(result.text, retrieved)

        assistant_message = Message(role=Role.ASSISTANT, content=reply_text)
        conversation.add_message(assistant_message)
        await self._repository.save(conversation)
        self._record_chat_trace(
            user_query=content,
            retrieved=used_retrieved,
            model_response=reply_text,
            conversation_id=conversation.id,
            duration_ms=duration_ms,
            retrieval_duration_ms=result.retrieval_duration_ms,
            mode="sync",
            retrieval_backend=retrieval_backend,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        return ChatReply(
            conversation_id=conversation.id,
            message=assistant_message,
            model=selected_model,
        )

    async def _chat_stream_agent(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
        self,
        conversation: Conversation,
        content: str,
        *,
        retrieval_backend: str | None,
        model: str | None,
        selected_model: str,
        is_cancelled: Callable[[], Awaitable[bool]] | None,
    ) -> AsyncIterator[StreamEvent]:
        if self._agent is None:
            raise ConfigurationError("El modo agente no está disponible")

        yield StreamMeta(
            conversation_id=conversation.id,
            model=selected_model,
            sources=(),
        )

        retrieved: list[RetrievedChunk] = []
        buffered: list[str] = []
        streamed: list[str] = []
        result: AgentRagResult | None = None
        cancelled = False
        started = time.perf_counter()

        async for event in self._agent.run_stream(
            list(conversation.messages),
            retrieval_backend=retrieval_backend,
            model=model,
            is_cancelled=is_cancelled,
        ):
            if is_cancelled is not None and await is_cancelled():
                cancelled = True
                break
            if isinstance(event, AgentToolEvent):
                yield StreamTool(
                    name=event.name,
                    status=event.status,
                    query=event.query,
                    output_preview=event.output_preview,
                )
                continue
            if isinstance(event, AgentSourcesEvent):
                retrieved = list(event.retrieved)
                yield StreamMeta(
                    conversation_id=conversation.id,
                    model=selected_model,
                    sources=self._sources_payload(retrieved),
                )
                if retrieved and buffered:
                    flushed = "".join(buffered)
                    buffered.clear()
                    streamed.append(flushed)
                    yield StreamToken(content=flushed)
                continue
            if isinstance(event, AgentTokenEvent):
                if not event.content:
                    continue
                if event.kind == "thinking":
                    yield StreamThinking(content=event.content)
                    continue
                if retrieved:
                    streamed.append(event.content)
                    yield StreamToken(content=event.content)
                else:
                    buffered.append(event.content)
                continue
            if isinstance(event, AgentRagResult):
                result = event
                if event.retrieved:
                    retrieved = list(event.retrieved)

        duration_ms = (
            result.duration_ms
            if result is not None and result.duration_ms is not None
            else int((time.perf_counter() - started) * 1000)
        )
        retrieval_duration_ms = (
            result.retrieval_duration_ms if result is not None else 0
        )
        input_tokens = result.input_tokens if result is not None else None
        output_tokens = result.output_tokens if result is not None else None

        if cancelled:
            full_text = "".join(streamed).strip()
            if full_text:
                conversation.add_message(Message(role=Role.ASSISTANT, content=full_text))
            await self._repository.save(conversation)
            if full_text:
                self._record_chat_trace(
                    user_query=content,
                    retrieved=retrieved,
                    model_response=full_text,
                    conversation_id=conversation.id,
                    duration_ms=duration_ms,
                    retrieval_duration_ms=retrieval_duration_ms,
                    mode="stream",
                    retrieval_backend=retrieval_backend,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            yield StreamCancelled(conversation_id=conversation.id)
            return

        raw_text = "".join(streamed) + "".join(buffered)
        if result is not None and result.text.strip():
            raw_text = result.text
        reply_text, used_retrieved = self._agent_final_text(raw_text, retrieved)
        if not "".join(streamed).strip():
            yield StreamToken(content=reply_text)

        conversation.add_message(Message(role=Role.ASSISTANT, content=reply_text))
        await self._repository.save(conversation)
        self._record_chat_trace(
            user_query=content,
            retrieved=used_retrieved,
            model_response=reply_text,
            conversation_id=conversation.id,
            duration_ms=duration_ms,
            retrieval_duration_ms=retrieval_duration_ms,
            mode="stream",
            retrieval_backend=retrieval_backend,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        yield StreamDone(conversation_id=conversation.id)

    def _agent_final_text(
        self,
        raw_text: str,
        retrieved: list[RetrievedChunk],
    ) -> tuple[str, list[RetrievedChunk]]:
        if not retrieved:
            refusal = (
                self._guardrails.out_of_scope_message
                if self._guardrails is not None
                else "No he encontrado contexto suficiente para esta pregunta."
            )
            return refusal, []
        text = (raw_text or "").strip()
        if not text:
            raise ValidationError("El modelo devolvió una respuesta vacía")
        if self._guardrails is not None:
            self._guardrails.check_output(text)
        return text, retrieved

    def _record_chat_trace(  # pylint: disable=too-many-arguments
        self,
        *,
        user_query: str,
        retrieved: list[RetrievedChunk],
        model_response: str,
        conversation_id: str,
        duration_ms: int,
        retrieval_duration_ms: int,
        mode: Literal["stream", "sync"],
        retrieval_backend: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        if self._tracer is None:
            return
        chunk_ids = tuple(item.chunk.id for item in retrieved)
        chunk_scores = tuple(item.score for item in retrieved)
        self._tracer.record_chat_generation(
            ChatGenerationTrace(
                user_query=user_query,
                chunk_ids=chunk_ids,
                chunk_scores=chunk_scores,
                model_response=model_response,
                model=self._llm.model_name,
                conversation_id=conversation_id,
                duration_ms=duration_ms,
                retrieval_duration_ms=retrieval_duration_ms,
                mode=mode,
                retrieval_backend=retrieval_backend,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
