"""Vector store que escribe en varios backends y enruta el retrieval."""

from __future__ import annotations

from chatbot.domain.documents import DocumentChunk, DocumentSummary, RetrievedChunk
from chatbot.domain.exceptions import ConfigurationError
from chatbot.domain.ports import VectorStorePort
from chatbot.domain.retrieval import (
    RETRIEVAL_BACKEND_NEO4J,
    RETRIEVAL_BACKEND_POSTGRES,
)


class RoutedVectorStore(VectorStorePort):
    def __init__(
        self,
        *,
        primary: VectorStorePort,
        neo4j: VectorStorePort | None = None,
        default_backend: str = RETRIEVAL_BACKEND_POSTGRES,
    ) -> None:
        self._primary = primary
        self._neo4j = neo4j
        self._default_backend = default_backend

    async def upsert(self, chunks: list[DocumentChunk]) -> None:
        await self._primary.upsert(chunks)
        if self._neo4j is not None:
            await self._neo4j.upsert(chunks)

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        return await self.search_backend(
            self._default_backend,
            query_embedding,
            top_k=top_k,
        )

    async def search_backend(
        self,
        backend: str,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        normalized = (backend or self._default_backend).lower()
        if normalized == RETRIEVAL_BACKEND_POSTGRES:
            return await self._primary.search(query_embedding, top_k=top_k)
        if normalized == RETRIEVAL_BACKEND_NEO4J:
            if self._neo4j is None:
                raise ConfigurationError("El flujo Neo4j no está disponible")
            return await self._neo4j.search(query_embedding, top_k=top_k)
        raise ConfigurationError(f"Backend de retrieval no soportado: {backend}")

    async def delete_by_document(self, document_id: str) -> int:
        deleted = await self._primary.delete_by_document(document_id)
        if self._neo4j is not None:
            await self._neo4j.delete_by_document(document_id)
        return deleted

    async def list_documents(self) -> list[DocumentSummary]:
        return await self._primary.list_documents()

    async def get_document(self, document_id: str) -> DocumentSummary | None:
        return await self._primary.get_document(document_id)
