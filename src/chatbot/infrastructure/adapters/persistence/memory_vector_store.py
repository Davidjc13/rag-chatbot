"""Vector store en memoria con similitud coseno."""

from __future__ import annotations

import math
import threading
from datetime import UTC, datetime

from chatbot.domain.documents import (
    DocumentChunk,
    DocumentFormat,
    DocumentSummary,
    RetrievedChunk,
)
from chatbot.domain.ports import VectorStorePort


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._documents: dict[str, DocumentSummary] = {}
        self._lock = threading.RLock()

    async def upsert(self, chunks: list[DocumentChunk]) -> None:
        with self._lock:
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.id} sin embedding")
                self._chunks[chunk.id] = chunk

                filename = str(chunk.metadata.get("filename", "unknown"))
                fmt_raw = str(chunk.metadata.get("format", DocumentFormat.PDF.value))
                try:
                    fmt = DocumentFormat(fmt_raw)
                except ValueError:
                    fmt = DocumentFormat.PDF

                existing = self._documents.get(chunk.document_id)
                if existing is None:
                    self._documents[chunk.document_id] = DocumentSummary(
                        id=chunk.document_id,
                        filename=filename,
                        format=fmt,
                        chunk_count=1,
                        created_at=datetime.now(UTC),
                    )
                else:
                    count = sum(
                        1 for c in self._chunks.values() if c.document_id == chunk.document_id
                    )
                    self._documents[chunk.document_id] = DocumentSummary(
                        id=existing.id,
                        filename=filename or existing.filename,
                        format=fmt,
                        chunk_count=count,
                        created_at=existing.created_at,
                    )

    async def search(self, query_embedding: list[float], *, top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        with self._lock:
            scored: list[RetrievedChunk] = []
            for chunk in self._chunks.values():
                if chunk.embedding is None:
                    continue
                score = _cosine(query_embedding, chunk.embedding)
                scored.append(RetrievedChunk(chunk=chunk, score=score))
            scored.sort(key=lambda item: item.score, reverse=True)
            return scored[:top_k]

    async def delete_by_document(self, document_id: str) -> int:
        with self._lock:
            to_delete = [cid for cid, c in self._chunks.items() if c.document_id == document_id]
            for cid in to_delete:
                del self._chunks[cid]
            self._documents.pop(document_id, None)
            return len(to_delete)

    async def list_documents(self) -> list[DocumentSummary]:
        with self._lock:
            return sorted(self._documents.values(), key=lambda d: d.created_at, reverse=True)

    async def get_document(self, document_id: str) -> DocumentSummary | None:
        with self._lock:
            return self._documents.get(document_id)
