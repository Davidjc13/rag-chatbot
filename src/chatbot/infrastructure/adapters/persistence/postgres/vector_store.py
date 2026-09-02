"""Vector store PostgreSQL + pgvector."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatbot.domain.documents import (
    DocumentChunk,
    DocumentFormat,
    DocumentPurpose,
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
                purpose_raw = str(
                    chunk.metadata.get("purpose", DocumentPurpose.GENERAL.value)
                )
                try:
                    fmt = DocumentFormat(fmt_raw)
                except ValueError:
                    fmt = DocumentFormat.PDF
                try:
                    purpose = DocumentPurpose(purpose_raw)
                except ValueError:
                    purpose = DocumentPurpose.GENERAL

                doc = await session.get(DocumentModel, chunk.document_id)
                if doc is None:
                    session.add(
                        DocumentModel(
                            id=chunk.document_id,
                            filename=filename,
                            format=fmt.value,
                            purpose=purpose.value,
                            chunk_count=0,
                            created_at=datetime.now(UTC),
                        )
                    )
                else:
                    doc.filename = filename or doc.filename
                    doc.format = fmt.value
                    doc.purpose = purpose.value

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
                    select(func.count(ChunkModel.id)).where(  # pylint: disable=not-callable
                        ChunkModel.document_id == doc_id
                    )
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
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        async with self._session_factory() as session:
            distance = ChunkModel.embedding.cosine_distance(query_embedding)
            stmt = select(ChunkModel, distance.label("distance")).order_by(distance).limit(top_k)
            if document_id:
                stmt = stmt.where(ChunkModel.document_id == document_id)
            result = await session.execute(stmt)
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

    async def keyword_search(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        cleaned = (query or "").strip()
        if top_k <= 0 or not cleaned:
            return []
        params: dict[str, object] = {"q": cleaned, "top_k": top_k}
        sql = """
            SELECT id, document_id, content, metadata,
                   ts_rank_cd(content_tsv, plainto_tsquery('simple', :q)) AS rank
            FROM chunks
            WHERE content_tsv @@ plainto_tsquery('simple', :q)
        """
        if document_id:
            sql += " AND document_id = :document_id"
            params["document_id"] = document_id
        sql += " ORDER BY rank DESC LIMIT :top_k"
        async with self._session_factory() as session:
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            retrieved: list[RetrievedChunk] = []
            for row in rows:
                retrieved.append(
                    RetrievedChunk(
                        chunk=DocumentChunk(
                            id=str(row["id"]),
                            document_id=str(row["document_id"]),
                            content=str(row["content"]),
                            metadata=dict(row["metadata"] or {}),
                            embedding=None,
                        ),
                        score=float(row["rank"] or 0.0),
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

    async def get_chunks_by_document(self, document_id: str) -> list[DocumentChunk]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ChunkModel).where(ChunkModel.document_id == document_id)
            )
            rows = result.scalars().all()
            chunks = [
                DocumentChunk(
                    id=row.id,
                    document_id=row.document_id,
                    content=row.content,
                    metadata=dict(row.chunk_metadata or {}),
                    embedding=list(row.embedding) if row.embedding is not None else None,
                )
                for row in rows
            ]
            chunks.sort(key=lambda c: int(c.metadata.get("chunk_index", 0)))
            return chunks

    async def list_documents(
        self,
        *,
        purpose: DocumentPurpose | None = None,
    ) -> list[DocumentSummary]:
        async with self._session_factory() as session:
            stmt = select(DocumentModel).order_by(DocumentModel.created_at.desc())
            if purpose is not None:
                stmt = stmt.where(DocumentModel.purpose == purpose.value)
            result = await session.execute(stmt)
            docs = result.scalars().all()
            return [
                DocumentSummary(
                    id=d.id,
                    filename=d.filename,
                    format=DocumentFormat(d.format),
                    chunk_count=d.chunk_count,
                    created_at=d.created_at,
                    purpose=DocumentPurpose(d.purpose)
                    if d.purpose in DocumentPurpose._value2member_map_
                    else DocumentPurpose.GENERAL,
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
                purpose=DocumentPurpose(d.purpose)
                if d.purpose in DocumentPurpose._value2member_map_
                else DocumentPurpose.GENERAL,
            )
