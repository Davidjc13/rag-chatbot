"""Vector store en Neo4j con índice vectorial para chunks."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from neo4j import AsyncDriver

from chatbot.domain.documents import (
    DocumentChunk,
    DocumentFormat,
    DocumentSummary,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


class Neo4jVectorStore:
    def __init__(
        self,
        driver: AsyncDriver,
        *,
        database: str,
        vector_index_name: str,
        embedding_dimension: int,
    ) -> None:
        self._driver = driver
        self._database = database
        self._vector_index_name = vector_index_name
        self._embedding_dimension = embedding_dimension

    async def initialize(self) -> None:
        async with self._driver.session(database=self._database) as session:
            await session.run(
                "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.id IS UNIQUE"
            )
            await session.run(
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            )
            index_name = self._quote_identifier(self._vector_index_name)
            await session.run(
                f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {self._embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
            await session.run(
                "CREATE INDEX document_created_at IF NOT EXISTS "
                "FOR (d:Document) ON (d.created_at)"
            )
            await session.run(
                "CREATE INDEX chunk_document_id IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.document_id)"
            )
        logger.info(
            "Esquema Neo4j inicializado",
            extra={"vector_index": self._vector_index_name, "database": self._database},
        )

    async def close(self) -> None:
        await self._driver.close()

    async def upsert(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        async with self._driver.session(database=self._database) as session:
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.id} sin embedding")

                filename = str(chunk.metadata.get("filename", "unknown"))
                fmt_raw = str(chunk.metadata.get("format", DocumentFormat.PDF.value))
                try:
                    fmt = DocumentFormat(fmt_raw)
                except ValueError:
                    fmt = DocumentFormat.PDF

                metadata_json = json.dumps(chunk.metadata, ensure_ascii=False)
                await session.run(
                    """
                    MERGE (d:Document {id: $document_id})
                    ON CREATE SET
                      d.filename = $filename,
                      d.format = $format,
                      d.created_at = $created_at,
                      d.chunk_count = 0
                    SET
                      d.filename = $filename,
                      d.format = $format
                    MERGE (c:Chunk {id: $chunk_id})
                    SET
                      c.document_id = $document_id,
                      c.content = $content,
                      c.metadata_json = $metadata_json,
                      c.filename = $filename,
                      c.format = $format,
                      c.embedding = $embedding
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    document_id=chunk.document_id,
                    filename=filename,
                    format=fmt.value,
                    created_at=chunk.metadata.get("document_created_at")
                    or datetime.now(UTC).isoformat(),
                    chunk_id=chunk.id,
                    content=chunk.content,
                    metadata_json=metadata_json,
                    embedding=chunk.embedding,
                )

            for document_id in {chunk.document_id for chunk in chunks}:
                await self._refresh_document_count(session, document_id)

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                YIELD node, score
                RETURN
                  node.id AS id,
                  node.document_id AS document_id,
                  node.content AS content,
                  node.metadata_json AS metadata_json,
                  score
                ORDER BY score DESC
                """,
                index_name=self._vector_index_name,
                top_k=top_k,
                query_embedding=query_embedding,
            )
            rows = await result.data()

        retrieved: list[RetrievedChunk] = []
        for row in rows:
            metadata = self._load_metadata(row.get("metadata_json"))
            retrieved.append(
                RetrievedChunk(
                    chunk=DocumentChunk(
                        id=str(row["id"]),
                        document_id=str(row["document_id"]),
                        content=str(row["content"]),
                        metadata=metadata,
                    ),
                    score=float(row["score"]),
                )
            )
        return retrieved

    async def delete_by_document(self, document_id: str) -> int:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
                WITH d, collect(c) AS chunks, count(c) AS chunk_count
                FOREACH (chunk IN chunks | DETACH DELETE chunk)
                DETACH DELETE d
                RETURN chunk_count
                """,
                document_id=document_id,
            )
            row = await result.single()
            return 0 if row is None else int(row["chunk_count"])

    async def list_documents(self) -> list[DocumentSummary]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (d:Document)
                RETURN d.id AS id, d.filename AS filename, d.format AS format,
                       d.chunk_count AS chunk_count, d.created_at AS created_at
                ORDER BY d.created_at DESC
                """
            )
            rows = await result.data()
        return [self._document_summary_from_row(row) for row in rows]

    async def get_document(self, document_id: str) -> DocumentSummary | None:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (d:Document {id: $document_id})
                RETURN d.id AS id, d.filename AS filename, d.format AS format,
                       d.chunk_count AS chunk_count, d.created_at AS created_at
                """,
                document_id=document_id,
            )
            row = await result.single()
        if row is None:
            return None
        return self._document_summary_from_row(dict(row))

    async def _refresh_document_count(self, session: Any, document_id: str) -> None:
        await session.run(
            """
            MATCH (d:Document {id: $document_id})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            WITH d, count(c) AS chunk_count
            SET d.chunk_count = chunk_count
            """,
            document_id=document_id,
        )

    @staticmethod
    def _quote_identifier(name: str) -> str:
        return f"`{name.replace('`', '``')}`"

    @staticmethod
    def _load_metadata(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _document_summary_from_row(row: dict[str, Any]) -> DocumentSummary:
        fmt_raw = str(row.get("format", DocumentFormat.PDF.value))
        try:
            fmt = DocumentFormat(fmt_raw)
        except ValueError:
            fmt = DocumentFormat.PDF

        created_at_raw = row.get("created_at")
        if isinstance(created_at_raw, datetime):
            created_at = created_at_raw
        else:
            created_at = datetime.now(UTC)

        return DocumentSummary(
            id=str(row["id"]),
            filename=str(row["filename"]),
            format=fmt,
            chunk_count=int(row.get("chunk_count") or 0),
            created_at=created_at,
        )
