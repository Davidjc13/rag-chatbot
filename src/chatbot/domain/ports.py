"""Puertos (interfaces) del dominio — arquitectura hexagonal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from chatbot.domain.documents import (
    DocumentChunk,
    DocumentSummary,
    ParsedDocument,
    RetrievedChunk,
)
from chatbot.domain.entities import Conversation, Message
from chatbot.domain.llm_stream import LLMDelta


@dataclass(frozen=True, slots=True)
class ChatGenerationTrace:
    """Datos de una petición RAG completada para observabilidad."""

    user_query: str
    chunk_ids: tuple[str, ...]
    chunk_scores: tuple[float, ...]
    model_response: str
    model: str
    conversation_id: str
    duration_ms: int
    retrieval_duration_ms: int
    mode: Literal["stream", "sync"] = "stream"
    retrieval_backend: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class TracingPort(ABC):
    """Puerto de salida para registrar trazas de generación."""

    @abstractmethod
    def record_chat_generation(self, trace: ChatGenerationTrace) -> None:
        """Registra una generación completada (p. ej. tras cerrar el SSE)."""


class LLMPort(ABC):
    """Puerto de salida hacia un proveedor de modelos de lenguaje."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> Message:
        """Genera una respuesta del asistente a partir del historial."""

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        """Genera la respuesta del asistente en streaming (deltas tipados)."""
        yield LLMDelta("")

    @abstractmethod
    async def health_check(self) -> bool:
        """Comprueba si el proveedor está disponible."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nombre del modelo en uso."""


class GuardrailPort(ABC):
    """Puerto de guardarraíles de entrada/salida y alcance RAG."""

    @abstractmethod
    def check_input(self, text: str) -> None:
        """Valida el mensaje del usuario. Lanza GuardrailBlockedError si falla."""

    @abstractmethod
    def check_output(self, text: str) -> None:
        """Valida la respuesta del asistente. Lanza GuardrailBlockedError si falla."""

    @abstractmethod
    def is_in_scope(self, scores: list[float]) -> bool:
        """Indica si el retrieval tiene suficiente relevancia para responder."""

    @property
    @abstractmethod
    def out_of_scope_message(self) -> str:
        """Mensaje fijo cuando la consulta queda fuera del alcance documental."""


class ConversationRepositoryPort(Protocol):
    """Puerto de persistencia de conversaciones."""

    async def get(self, conversation_id: str) -> Conversation | None: ...

    async def save(self, conversation: Conversation) -> None: ...

    async def delete(self, conversation_id: str) -> None: ...


class PromptRepositoryPort(Protocol):
    """Puerto de persistencia de prompts markdown (system / user_message)."""

    async def get(self, key: str) -> str | None: ...

    async def upsert(self, key: str, content: str) -> None: ...


class DocumentParserPort(ABC):
    """Puerto de salida para parsear documentos binarios."""

    @abstractmethod
    def supports(self, filename: str) -> bool:
        """Indica si el parser acepta el fichero."""

    @abstractmethod
    def parse(self, *, filename: str, data: bytes) -> ParsedDocument:
        """Extrae bloques de texto y tablas del documento."""


class EmbeddingPort(ABC):
    """Puerto de salida para embeddings."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Devuelve un vector por cada texto."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nombre del modelo de embeddings."""


class VectorStorePort(ABC):
    """Puerto de salida hacia un almacén vectorial."""

    @abstractmethod
    async def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Inserta o actualiza chunks con embedding."""

    @abstractmethod
    async def search(self, query_embedding: list[float], *, top_k: int) -> list[RetrievedChunk]:
        """Busca los chunks más similares al embedding de consulta."""

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> int:
        """Elimina todos los chunks de un documento. Devuelve cuántos borró."""

    @abstractmethod
    async def list_documents(self) -> list[DocumentSummary]:
        """Lista documentos indexados."""

    @abstractmethod
    async def get_document(self, document_id: str) -> DocumentSummary | None:
        """Obtiene el resumen de un documento o None."""
