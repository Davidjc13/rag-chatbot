"""Casos de uso de ingestión de documentos."""

from __future__ import annotations

import logging
from typing import Protocol

from chatbot.application.services.table_aware_chunker import TableAwareChunker
from chatbot.domain.documents import DocumentPurpose, DocumentSummary, IngestionResult
from chatbot.domain.exceptions import DocumentNotFoundError, ValidationError
from chatbot.domain.ports import DocumentParserPort, EmbeddingPort, VectorStorePort

logger = logging.getLogger(__name__)


class DocumentParserResolver(Protocol):
    def get_parser(self, filename: str) -> DocumentParserPort: ...


class StudyProfileGenerator(Protocol):
    async def generate_profile(self, document_id: str) -> object: ...


class IngestionService:
    """Orquesta parse → chunk (tablas protegidas) → embed → upsert."""

    def __init__(
        self,
        *,
        parser_factory: DocumentParserResolver,
        chunker: TableAwareChunker,
        embeddings: EmbeddingPort,
        vector_store: VectorStorePort,
        study_service: StudyProfileGenerator | None = None,
    ) -> None:
        self._parser_factory = parser_factory
        self._chunker = chunker
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._study_service = study_service

    async def ingest(  # pylint: disable=too-many-locals
        self,
        *,
        filename: str,
        data: bytes,
        purpose: DocumentPurpose = DocumentPurpose.GENERAL,
        document_id: str | None = None,
    ) -> IngestionResult:
        name = (filename or "").strip()
        if not name:
            raise ValidationError("El nombre del fichero es obligatorio")
        if not data:
            raise ValidationError("El fichero está vacío")

        replace_id = (document_id or "").strip() or None
        existing_purpose = purpose
        if replace_id:
            existing = await self._vector_store.get_document(replace_id)
            if existing is None:
                raise DocumentNotFoundError(replace_id)
            existing_purpose = existing.purpose
            await self._vector_store.delete_by_document(replace_id)

        parser = self._parser_factory.get_parser(name)
        parsed = parser.parse(filename=name, data=data)
        if replace_id:
            parsed.id = replace_id
        chunks = self._chunker.chunk(parsed)
        if not chunks:
            raise ValidationError("No se generaron chunks a partir del documento")

        resolved_purpose = existing_purpose if replace_id else purpose
        for chunk in chunks:
            chunk.metadata["purpose"] = resolved_purpose.value

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
                "purpose": resolved_purpose.value,
                "embedding_model": self._embeddings.model_name,
            },
        )

        if resolved_purpose == DocumentPurpose.NOTES and self._study_service is not None:
            await self._study_service.generate_profile(parsed.id)

        return IngestionResult(
            document_id=parsed.id,
            filename=name,
            format=parsed.format,
            chunk_count=len(chunks),
            purpose=resolved_purpose,
        )

    async def list_documents(
        self,
        *,
        purpose: DocumentPurpose | None = None,
    ) -> list[DocumentSummary]:
        return await self._vector_store.list_documents(purpose=purpose)

    async def list_document_chunks(self, document_id: str):
        existing = await self._vector_store.get_document(document_id)
        if existing is None:
            raise DocumentNotFoundError(document_id)
        return await self._vector_store.get_chunks_by_document(document_id)

    async def delete_document(self, document_id: str) -> None:
        existing = await self._vector_store.get_document(document_id)
        if existing is None:
            raise DocumentNotFoundError(document_id)
        deleted = await self._vector_store.delete_by_document(document_id)
        logger.info(
            "Documento eliminado",
            extra={"document_id": document_id, "deleted_chunks": deleted},
        )
