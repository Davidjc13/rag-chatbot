"""Tests de filtrado por document_id en vector stores."""

from __future__ import annotations

import pytest

from chatbot.domain.documents import DocumentChunk, DocumentPurpose
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_search_filters_by_document_id() -> None:
    store = InMemoryVectorStore()
    embeddings = MockEmbeddingAdapter()
    v1 = (await embeddings.embed(["doc uno"]))[0]
    v2 = (await embeddings.embed(["doc dos"]))[0]

    await store.upsert(
        [
            DocumentChunk(
                document_id="doc-a",
                content="Contenido del documento A",
                metadata={"filename": "a.pdf", "purpose": DocumentPurpose.NOTES.value},
                embedding=v1,
            ),
            DocumentChunk(
                document_id="doc-b",
                content="Contenido del documento B",
                metadata={"filename": "b.pdf", "purpose": DocumentPurpose.GENERAL.value},
                embedding=v2,
            ),
        ]
    )

    hits = await store.search(v1, top_k=5, document_id="doc-a")
    assert len(hits) == 1
    assert hits[0].chunk.document_id == "doc-a"


@pytest.mark.asyncio
async def test_get_chunks_by_document() -> None:
    store = InMemoryVectorStore()
    embeddings = MockEmbeddingAdapter()
    vector = (await embeddings.embed(["chunk"]))[0]
    await store.upsert(
        [
            DocumentChunk(
                document_id="doc-x",
                content="Segundo",
                metadata={"chunk_index": 1},
                embedding=vector,
            ),
            DocumentChunk(
                document_id="doc-x",
                content="Primero",
                metadata={"chunk_index": 0},
                embedding=vector,
            ),
        ]
    )
    chunks = await store.get_chunks_by_document("doc-x")
    assert [c.content for c in chunks] == ["Primero", "Segundo"]


@pytest.mark.asyncio
async def test_list_documents_by_purpose() -> None:
    store = InMemoryVectorStore()
    embeddings = MockEmbeddingAdapter()
    vector = (await embeddings.embed(["x"]))[0]
    await store.upsert(
        [
            DocumentChunk(
                document_id="n1",
                content="notas",
                metadata={"purpose": DocumentPurpose.NOTES.value},
                embedding=vector,
            ),
            DocumentChunk(
                document_id="g1",
                content="general",
                metadata={"purpose": DocumentPurpose.GENERAL.value},
                embedding=vector,
            ),
        ]
    )
    notes = await store.list_documents(purpose=DocumentPurpose.NOTES)
    assert len(notes) == 1
    assert notes[0].purpose == DocumentPurpose.NOTES
