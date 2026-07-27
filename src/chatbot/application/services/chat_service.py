"""Casos de uso de chat con retrieval RAG y streaming."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from chatbot.domain.documents import RetrievedChunk
from chatbot.domain.entities import ChatReply, Conversation, Message, Role
from chatbot.domain.exceptions import ConversationNotFoundError, ValidationError
from chatbot.domain.ports import (
    ConversationRepositoryPort,
    EmbeddingPort,
    GuardrailPort,
    LLMPort,
    VectorStorePort,
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
class StreamDone:
    conversation_id: str


StreamEvent = StreamMeta | StreamToken | StreamDone

_CITATION_INSTRUCTIONS = (
    "Cuando uses información de un fragmento, cita la fuente en línea con "
    "exactamente este formato (sin alterar los campos ni añadir espacios extra "
    "alrededor del signo = salvo el de index): "
    "<index = N, source=rag, title=NOMBRE_ARCHIVO, id=DOCUMENT_ID>. "
    "Usa el índice y los metadatos indicados en cada fragmento. "
    "No inventes índices, títulos ni ids."
)


class ChatService:
    """Servicio de aplicación: orquesta conversación + contexto RAG."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        llm: LLMPort,
        repository: ConversationRepositoryPort,
        *,
        system_prompt: str,
        embeddings: EmbeddingPort | None = None,
        vector_store: VectorStorePort | None = None,
        guardrails: GuardrailPort | None = None,
        rag_top_k: int = 4,
    ) -> None:
        self._llm = llm
        self._repository = repository
        self._system_prompt = system_prompt
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._guardrails = guardrails
        self._rag_top_k = rag_top_k

    async def chat(
        self,
        user_message: str,
        *,
        conversation_id: str | None = None,
    ) -> ChatReply:
        content = (user_message or "").strip()
        if not content:
            raise ValidationError("El mensaje del usuario no puede estar vacío")

        if self._guardrails is not None:
            self._guardrails.check_input(content)

        conversation = await self._resolve_conversation(conversation_id)
        conversation.add_message(Message(role=Role.USER, content=content))

        retrieved = await self._retrieve(content)
        if self._guardrails is not None and not self._guardrails.is_in_scope(
            [item.score for item in retrieved]
        ):
            assistant_message = Message(
                role=Role.ASSISTANT,
                content=self._guardrails.out_of_scope_message,
            )
            conversation.add_message(assistant_message)
            await self._repository.save(conversation)
            return ChatReply(
                conversation_id=conversation.id,
                message=assistant_message,
                model=self._llm.model_name,
            )

        system_prompt = self._build_system_prompt(retrieved)

        logger.info(
            "Generando respuesta",
            extra={
                "conversation_id": conversation.id,
                "model": self._llm.model_name,
                "history_size": len(conversation.messages),
                "rag_chunks": len(retrieved),
            },
        )

        assistant_message = await self._llm.generate(
            conversation.history(),
            system_prompt=system_prompt,
        )
        if self._guardrails is not None:
            self._guardrails.check_output(assistant_message.content)

        conversation.add_message(assistant_message)
        await self._repository.save(conversation)

        logger.info(
            "Respuesta generada",
            extra={"conversation_id": conversation.id, "model": self._llm.model_name},
        )

        return ChatReply(
            conversation_id=conversation.id,
            message=assistant_message,
            model=self._llm.model_name,
        )

    async def chat_stream(
        self,
        user_message: str,
        *,
        conversation_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        content = (user_message or "").strip()
        if not content:
            raise ValidationError("El mensaje del usuario no puede estar vacío")

        if self._guardrails is not None:
            self._guardrails.check_input(content)

        conversation = await self._resolve_conversation(conversation_id)
        conversation.add_message(Message(role=Role.USER, content=content))

        retrieved = await self._retrieve(content)
        sources = self._sources_payload(retrieved)
        yield StreamMeta(
            conversation_id=conversation.id,
            model=self._llm.model_name,
            sources=sources,
        )

        if self._guardrails is not None and not self._guardrails.is_in_scope(
            [item.score for item in retrieved]
        ):
            refusal = self._guardrails.out_of_scope_message
            yield StreamToken(content=refusal)
            conversation.add_message(Message(role=Role.ASSISTANT, content=refusal))
            await self._repository.save(conversation)
            yield StreamDone(conversation_id=conversation.id)
            return

        system_prompt = self._build_system_prompt(retrieved)
        logger.info(
            "Generando respuesta (stream)",
            extra={
                "conversation_id": conversation.id,
                "model": self._llm.model_name,
                "history_size": len(conversation.messages),
                "rag_chunks": len(retrieved),
            },
        )

        parts: list[str] = []
        async for delta in self._llm.generate_stream(
            conversation.history(),
            system_prompt=system_prompt,
        ):
            if not delta:
                continue
            parts.append(delta)
            yield StreamToken(content=delta)

        full_text = "".join(parts).strip()
        if not full_text:
            raise ValidationError("El modelo devolvió una respuesta vacía")

        if self._guardrails is not None:
            self._guardrails.check_output(full_text)

        conversation.add_message(Message(role=Role.ASSISTANT, content=full_text))
        await self._repository.save(conversation)
        yield StreamDone(conversation_id=conversation.id)

    async def get_conversation(self, conversation_id: str) -> Conversation:
        conversation = await self._repository.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    async def _resolve_conversation(self, conversation_id: str | None) -> Conversation:
        if conversation_id is None:
            return Conversation()

        conversation = await self._repository.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    async def _retrieve(self, query: str) -> list[RetrievedChunk]:
        if self._embeddings is None or self._vector_store is None:
            return []
        vectors = await self._embeddings.embed([query])
        if not vectors:
            return []
        return await self._vector_store.search(vectors[0], top_k=self._rag_top_k)

    def _build_system_prompt(self, retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return self._system_prompt

        parts = [
            self._system_prompt,
            "",
            "Usa el siguiente contexto de documentos para responder. "
            "Si el contexto no es suficiente, dilo con claridad.",
            _CITATION_INSTRUCTIONS,
            "",
        ]
        for index, item in enumerate(retrieved, start=1):
            filename = str(item.chunk.metadata.get("filename", "documento"))
            doc_id = item.chunk.document_id
            citation = (
                f"<index = {index}, source=rag, title={filename}, id={doc_id}>"
            )
            parts.append(
                f"[Fragmento {index} | score={item.score:.3f} | cita={citation}]"
            )
            parts.append(item.chunk.content)
            parts.append("")
        return "\n".join(parts).strip()

    @staticmethod
    def _sources_payload(
        retrieved: list[RetrievedChunk],
    ) -> tuple[dict[str, str | int], ...]:
        sources: list[dict[str, str | int]] = []
        for index, item in enumerate(retrieved, start=1):
            sources.append(
                {
                    "index": index,
                    "source": "rag",
                    "title": str(item.chunk.metadata.get("filename", "documento")),
                    "id": item.chunk.document_id,
                }
            )
        return tuple(sources)
