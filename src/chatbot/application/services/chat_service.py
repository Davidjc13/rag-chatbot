"""Casos de uso de chat con retrieval RAG y streaming."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from chatbot.domain.documents import RetrievedChunk
from chatbot.domain.entities import ChatReply, Conversation, Message, Role
from chatbot.domain.exceptions import ConversationNotFoundError, ValidationError
from chatbot.domain.ports import (
    ChatGenerationTrace,
    ConversationRepositoryPort,
    EmbeddingPort,
    GuardrailPort,
    LLMPort,
    PromptRepositoryPort,
    TracingPort,
    VectorStorePort,
)
from chatbot.domain.prompts import PROMPT_SYSTEM, PROMPT_USER_MESSAGE

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


StreamEvent = StreamMeta | StreamToken | StreamThinking | StreamDone

_CITATION_INSTRUCTIONS = (
    "Cuando uses información de un documento, cítalo en línea una sola vez "
    "con exactamente este formato (sin alterar los campos ni añadir espacios "
    "extra alrededor del signo = salvo el de index): "
    "<index = N, source=rag, title=NOMBRE_ARCHIVO, id=DOCUMENT_ID>. "
    "Cada documento tiene un único índice: no repitas la misma cita ni cites "
    "el mismo id más de una vez. No inventes índices, títulos ni ids."
)

_FALLBACK_SYSTEM = (
    "Eres un asistente de documentos. Responde solo con el contexto RAG. "
    "Si razonas internamente, hazlo de forma breve (pocas frases) y luego "
    "da la respuesta final al usuario.\n\n{context}"
)
_FALLBACK_USER = "{question}"


class ChatService:
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
        rag_top_k: int = 4,
    ) -> None:
        self._llm = llm
        self._repository = repository
        self._prompts = prompts
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._guardrails = guardrails
        self._tracer = tracer
        self._rag_top_k = rag_top_k

    def _resolve_model(self, model: str | None) -> str:
        return (model or "").strip() or self._llm.model_name

    async def chat(
        self,
        user_message: str,
        *,
        conversation_id: str | None = None,
        retrieval_backend: str | None = None,
        model: str | None = None,
    ) -> ChatReply:
        content = (user_message or "").strip()
        if not content:
            raise ValidationError("El mensaje del usuario no puede estar vacío")

        selected_model = self._resolve_model(model)

        if self._guardrails is not None:
            self._guardrails.check_input(content)

        conversation = await self._resolve_conversation(conversation_id)
        conversation.add_message(Message(role=Role.USER, content=content))

        retrieved, retrieval_duration_ms = await self._retrieve_timed(
            content,
            retrieval_backend=retrieval_backend,
        )
        if self._guardrails is not None and not self._guardrails.is_in_scope(
            [item.score for item in retrieved]
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

        system_prompt, llm_messages = await self._build_llm_payload(conversation, retrieved)

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

    async def chat_stream(
        self,
        user_message: str,
        *,
        conversation_id: str | None = None,
        retrieval_backend: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        content = (user_message or "").strip()
        if not content:
            raise ValidationError("El mensaje del usuario no puede estar vacío")

        selected_model = self._resolve_model(model)

        if self._guardrails is not None:
            self._guardrails.check_input(content)

        conversation = await self._resolve_conversation(conversation_id)
        conversation.add_message(Message(role=Role.USER, content=content))

        retrieved, retrieval_duration_ms = await self._retrieve_timed(
            content,
            retrieval_backend=retrieval_backend,
        )
        sources = self._sources_payload(retrieved)
        yield StreamMeta(
            conversation_id=conversation.id,
            model=selected_model,
            sources=sources,
        )

        if self._guardrails is not None and not self._guardrails.is_in_scope(
            [item.score for item in retrieved]
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

        system_prompt, llm_messages = await self._build_llm_payload(conversation, retrieved)
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
        async for delta in self._llm.generate_stream(
            llm_messages,
            system_prompt=system_prompt,
            model=model,
        ):
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
    ) -> list[RetrievedChunk]:
        if self._embeddings is None or self._vector_store is None:
            return []
        vectors = await self._embeddings.embed([query])
        if not vectors:
            return []
        search_backend = getattr(self._vector_store, "search_backend", None)
        if callable(search_backend) and retrieval_backend:
            return await search_backend(
                retrieval_backend,
                vectors[0],
                top_k=self._rag_top_k,
            )
        return await self._vector_store.search(vectors[0], top_k=self._rag_top_k)

    async def _retrieve_timed(
        self,
        query: str,
        *,
        retrieval_backend: str | None = None,
    ) -> tuple[list[RetrievedChunk], int]:
        started = time.perf_counter()
        retrieved = await self._retrieve(query, retrieval_backend=retrieval_backend)
        duration_ms = int((time.perf_counter() - started) * 1000)
        return retrieved, duration_ms

    async def _build_llm_payload(
        self,
        conversation: Conversation,
        retrieved: list[RetrievedChunk],
    ) -> tuple[str, list[Message]]:
        context = self._build_context(retrieved)
        system_template = await self._prompts.get(PROMPT_SYSTEM) or _FALLBACK_SYSTEM
        user_template = await self._prompts.get(PROMPT_USER_MESSAGE) or _FALLBACK_USER

        system_prompt = system_template.replace("{context}", context)
        if "razonamiento breve" not in system_prompt.lower():
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
    def _document_index_map(
        retrieved: list[RetrievedChunk],
    ) -> dict[str, tuple[int, str]]:
        """Asigna un índice único por document_id (orden de primera aparición)."""
        mapping: dict[str, tuple[int, str]] = {}
        next_index = 1
        for item in retrieved:
            doc_id = item.chunk.document_id
            if doc_id in mapping:
                continue
            filename = str(item.chunk.metadata.get("filename", "documento"))
            mapping[doc_id] = (next_index, filename)
            next_index += 1
        return mapping

    @staticmethod
    def _build_context(retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return "(Sin fragmentos relevantes recuperados.)"

        doc_map = ChatService._document_index_map(retrieved)
        parts = [
            _CITATION_INSTRUCTIONS,
            "",
            "Documentos (cita cada uno como máximo una vez):",
        ]
        for doc_id, (index, filename) in doc_map.items():
            citation = (
                f"<index = {index}, source=rag, title={filename}, id={doc_id}>"
            )
            parts.append(f"- {citation}")
        parts.append("")

        for item in retrieved:
            doc_id = item.chunk.document_id
            index, filename = doc_map[doc_id]
            citation = (
                f"<index = {index}, source=rag, title={filename}, id={doc_id}>"
            )
            parts.append(
                f"[Fragmento doc={index} | score={item.score:.3f} | cita={citation}]"
            )
            parts.append(item.chunk.content)
            parts.append("")
        return "\n".join(parts).strip()

    @staticmethod
    def _sources_payload(
        retrieved: list[RetrievedChunk],
    ) -> tuple[dict[str, str | int], ...]:
        sources: list[dict[str, str | int]] = []
        for doc_id, (index, filename) in ChatService._document_index_map(
            retrieved
        ).items():
            sources.append(
                {
                    "index": index,
                    "source": "rag",
                    "title": filename,
                    "id": doc_id,
                }
            )
        return tuple(sources)

    def _record_chat_trace(
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
