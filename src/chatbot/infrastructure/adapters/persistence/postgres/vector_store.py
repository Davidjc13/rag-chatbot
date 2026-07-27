"""Vector store PostgreSQL + pgvector."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatbot.domain.documents import (
    DocumentChunk,
    DocumentFormat,
    DocumentSummary,
    RetrievedChunk,
)
from chatbot.infrastructure.adapters.persistence.postgres.models import (
    ChunkModel,
    DocumentModel,
)


class PostgresVectorStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def upsert(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        async with self._session_factory() as session:
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.id} sin embedding")

                filename = str(chunk.metadata.get("filename", "unknown"))
                fmt_raw = str(chunk.metadata.get("format", DocumentFormat.PDF.value))
                try:
                    fmt = DocumentFormat(fmt_raw)
                except ValueError:
                    fmt = DocumentFormat.PDF

                doc = await session.get(DocumentModel, chunk.document_id)
                if doc is None:
                    session.add(
                        DocumentModel(
                            id=chunk.document_id,
                            filename=filename,
                            format=fmt.value,
                            chunk_count=0,
                            created_at=datetime.now(UTC),
                        )
                    )
                else:
                    doc.filename = filename or doc.filename
                    doc.format = fmt.value

                existing = await session.get(ChunkModel, chunk.id)
                if existing is None:
                    session.add(
                        ChunkModel(
                            id=chunk.id,
                            document_id=chunk.document_id,
                            content=chunk.content,
                            chunk_metadata=dict(chunk.metadata),
                            embedding=chunk.embedding,
                        )
                    )
                else:
                    existing.document_id = chunk.document_id
                    existing.content = chunk.content
                    existing.chunk_metadata = dict(chunk.metadata)
                    existing.embedding = chunk.embedding

            await session.flush()
            # Actualizar conteos por documento tocado
            doc_ids = {c.document_id for c in chunks}
            for doc_id in doc_ids:
                count = await session.scalar(
                    select(func.count())
                    .select_from(ChunkModel)
                    .where(ChunkModel.document_id == doc_id)
                )
                doc = await session.get(DocumentModel, doc_id)
                if doc is not None:
                    doc.chunk_count = int(count or 0)

            await session.commit()

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        async with self._session_factory() as session:
            distance = ChunkModel.embedding.cosine_distance(query_embedding)
            result = await session.execute(
                select(ChunkModel, distance.label("distance"))
                .order_by(distance)
                .limit(top_k)
            )
            rows = result.all()
            retrieved: list[RetrievedChunk] = []
            for chunk_row, dist in rows:
                score = 1.0 - float(dist)
                retrieved.append(
                    RetrievedChunk(
                        chunk=DocumentChunk(
                            id=chunk_row.id,
                            document_id=chunk_row.document_id,
                            content=chunk_row.content,
                            metadata=dict(chunk_row.chunk_metadata or {}),
                            embedding=list(chunk_row.embedding)
                            if chunk_row.embedding is not None
                            else None,
                        ),
                        score=score,
                    )
                )
            return retrieved

    async def delete_by_document(self, document_id: str) -> int:
        async with self._session_factory() as session:
            result = await session.execute(
                delete(ChunkModel).where(ChunkModel.document_id == document_id)
            )
            await session.execute(
                delete(DocumentModel).where(DocumentModel.id == document_id)
            )
            await session.commit()
            return int(result.rowcount or 0)

    async def list_documents(self) -> list[DocumentSummary]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentModel).order_by(DocumentModel.created_at.desc())
            )
            docs = result.scalars().all()
            return [
                DocumentSummary(
                    id=d.id,
                    filename=d.filename,
                    format=DocumentFormat(d.format),
                    chunk_count=d.chunk_count,
                    created_at=d.created_at,
                )
                for d in docs
            ]

    async def get_document(self, document_id: str) -> DocumentSummary | None:
        async with self._session_factory() as session:
            d = await session.get(DocumentModel, document_id)
            if d is None:
                return None
            return DocumentSummary(
                id=d.id,
                filename=d.filename,
                format=DocumentFormat(d.format),
                chunk_count=d.chunk_count,
                created_at=d.created_at,
            )
