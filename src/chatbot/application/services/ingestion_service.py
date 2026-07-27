"""Casos de uso de ingestión de documentos."""

from __future__ import annotations

import logging
from typing import Protocol

from chatbot.application.services.table_aware_chunker import TableAwareChunker
from chatbot.domain.documents import DocumentSummary, IngestionResult
from chatbot.domain.exceptions import DocumentNotFoundError, ValidationError
from chatbot.domain.ports import DocumentParserPort, EmbeddingPort, VectorStorePort

logger = logging.getLogger(__name__)


class DocumentParserResolver(Protocol):
    def get_parser(self, filename: str) -> DocumentParserPort: ...


class IngestionService:
    """Orquesta parse → chunk (tablas protegidas) → embed → upsert."""

    def __init__(
        self,
        *,
        parser_factory: DocumentParserResolver,
        chunker: TableAwareChunker,
        embeddings: EmbeddingPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._parser_factory = parser_factory
        self._chunker = chunker
        self._embeddings = embeddings
        self._vector_store = vector_store

    async def ingest(self, *, filename: str, data: bytes) -> IngestionResult:
        name = (filename or "").strip()
        if not name:
            raise ValidationError("El nombre del fichero es obligatorio")
        if not data:
            raise ValidationError("El fichero está vacío")

        parser = self._parser_factory.get_parser(name)
        parsed = parser.parse(filename=name, data=data)
        chunks = self._chunker.chunk(parsed)
        if not chunks:
            raise ValidationError("No se generaron chunks a partir del documento")

        vectors = await self._embeddings.embed([c.content for c in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector

        await self._vector_store.upsert(chunks)
        logger.info(
            "Documento ingerido",
            extra={
                "document_id": parsed.id,
                "document_filename": name,
                "chunk_count": len(chunks),
                "embedding_model": self._embeddings.model_name,
            },
        )
        return IngestionResult(
            document_id=parsed.id,
            filename=name,
            format=parsed.format,
            chunk_count=len(chunks),
        )

    async def list_documents(self) -> list[DocumentSummary]:
        return await self._vector_store.list_documents()

    async def delete_document(self, document_id: str) -> None:
        existing = await self._vector_store.get_document(document_id)
        if existing is None:
            raise DocumentNotFoundError(document_id)
        deleted = await self._vector_store.delete_by_document(document_id)
        logger.info(
            "Documento eliminado",
            extra={"document_id": document_id, "deleted_chunks": deleted},
        )
