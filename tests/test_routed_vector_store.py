"""Tests del vector store enrutado Postgres/Neo4j."""

from __future__ import annotations

import pytest

from chatbot.domain.documents import DocumentChunk
from chatbot.domain.exceptions import ConfigurationError
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore
from chatbot.infrastructure.adapters.persistence.routed_vector_store import RoutedVectorStore


class TrackingStore(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.deleted_ids: list[str] = []

    async def delete_by_document(self, document_id: str) -> int:
        self.deleted_ids.append(document_id)
        return await super().delete_by_document(document_id)


@pytest.mark.asyncio
async def test_routed_vector_store_writes_to_both_backends() -> None:
    primary = TrackingStore()
    secondary = TrackingStore()
    store = RoutedVectorStore(primary=primary, neo4j=secondary)
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        content="contenido postgres y neo4j",
        metadata={"filename": "policy.docx", "format": "docx"},
        embedding=[1.0, 0.0],
    )

    await store.upsert([chunk])

    assert await primary.get_document("doc-1") is not None
    assert await secondary.get_document("doc-1") is not None


@pytest.mark.asyncio
async def test_routed_vector_store_selects_backend_for_search() -> None:
    primary = TrackingStore()
    secondary = TrackingStore()
    store = RoutedVectorStore(primary=primary, neo4j=secondary)

    await primary.upsert(
        [
            DocumentChunk(
                id="pg-1",
                document_id="doc-pg",
                content="respuesta desde postgres",
                metadata={"filename": "pg.docx", "format": "docx"},
                embedding=[1.0, 0.0],
            )
        ]
    )
    await secondary.upsert(
        [
            DocumentChunk(
                id="neo-1",
                document_id="doc-neo",
                content="respuesta desde neo4j",
                metadata={"filename": "neo.docx", "format": "docx"},
                embedding=[1.0, 0.0],
            )
        ]
    )

    postgres_hits = await store.search_backend("postgres", [1.0, 0.0], top_k=1)
    neo4j_hits = await store.search_backend("neo4j", [1.0, 0.0], top_k=1)

    assert postgres_hits[0].chunk.document_id == "doc-pg"
    assert neo4j_hits[0].chunk.document_id == "doc-neo"


@pytest.mark.asyncio
async def test_routed_vector_store_delete_mirrors_to_secondary() -> None:
    primary = TrackingStore()
    secondary = TrackingStore()
    store = RoutedVectorStore(primary=primary, neo4j=secondary)
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        content="contenido",
        metadata={"filename": "policy.docx", "format": "docx"},
        embedding=[1.0, 0.0],
    )
    await store.upsert([chunk])

    deleted = await store.delete_by_document("doc-1")

    assert deleted == 1
    assert primary.deleted_ids == ["doc-1"]
    assert secondary.deleted_ids == ["doc-1"]


@pytest.mark.asyncio
async def test_routed_vector_store_raises_if_neo4j_missing() -> None:
    store = RoutedVectorStore(primary=TrackingStore(), neo4j=None)

    with pytest.raises(ConfigurationError):
        await store.search_backend("neo4j", [1.0, 0.0], top_k=1)
