"""Tests unitarios del adapter Neo4j vector store."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from chatbot.domain.documents import DocumentChunk
from chatbot.infrastructure.adapters.persistence.neo4j_vector_store import Neo4jVectorStore


class FakeResult:
    def __init__(
        self,
        *,
        data_rows: list[dict] | None = None,
        single_row: dict | None = None,
    ) -> None:
        self._data_rows = data_rows or []
        self._single_row = single_row

    async def data(self) -> list[dict]:
        return self._data_rows

    async def single(self) -> dict | None:
        return self._single_row


class FakeSession:
    def __init__(self) -> None:
        self.run = AsyncMock(return_value=FakeResult())

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakeDriver:
    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self.close = AsyncMock()

    def session(self, *, database: str) -> FakeSession:
        return self._session


@pytest.mark.asyncio
async def test_initialize_creates_constraints_and_index() -> None:
    session = FakeSession()
    store = Neo4jVectorStore(
        FakeDriver(session),  # type: ignore[arg-type]
        database="neo4j",
        vector_index_name="chunk_embeddings",
        embedding_dimension=768,
    )

    await store.initialize()

    calls = [call.args[0] for call in session.run.await_args_list]
    assert any("CREATE CONSTRAINT document_id_unique" in query for query in calls)
    assert any("CREATE CONSTRAINT chunk_id_unique" in query for query in calls)
    assert any("CREATE VECTOR INDEX `chunk_embeddings`" in query for query in calls)


@pytest.mark.asyncio
async def test_search_maps_rows_to_retrieved_chunks() -> None:
    session = FakeSession()
    session.run = AsyncMock(
        return_value=FakeResult(
            data_rows=[
                {
                    "id": "chunk-1",
                    "document_id": "doc-1",
                    "content": "contenido",
                    "metadata_json": '{"filename":"policy.docx","format":"docx"}',
                    "score": 0.98,
                }
            ]
        )
    )
    store = Neo4jVectorStore(
        FakeDriver(session),  # type: ignore[arg-type]
        database="neo4j",
        vector_index_name="chunk_embeddings",
        embedding_dimension=768,
    )

    hits = await store.search([1.0, 0.0], top_k=1)

    assert hits[0].chunk.document_id == "doc-1"
    assert hits[0].chunk.metadata["filename"] == "policy.docx"
    assert hits[0].score == pytest.approx(0.98)


@pytest.mark.asyncio
async def test_upsert_requires_embedding() -> None:
    session = FakeSession()
    store = Neo4jVectorStore(
        FakeDriver(session),  # type: ignore[arg-type]
        database="neo4j",
        vector_index_name="chunk_embeddings",
        embedding_dimension=768,
    )

    with pytest.raises(ValueError, match="sin embedding"):
        await store.upsert([DocumentChunk(document_id="doc-1", content="hola")])
