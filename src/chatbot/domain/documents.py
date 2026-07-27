"""Entidades de dominio para documentos e ingestión RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class DocumentFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"


class ContentKind(StrEnum):
    TEXT = "text"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class ContentBlock:
    kind: ContentKind
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind == ContentKind.TEXT and not self.text.strip():
            raise ValueError("Un bloque de texto no puede estar vacío")


@dataclass(slots=True)
class ParsedDocument:
    filename: str
    format: DocumentFormat
    blocks: list[ContentBlock]
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class DocumentChunk:
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("El contenido del chunk no puede estar vacío")


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    id: str
    filename: str
    format: DocumentFormat
    chunk_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: str
    filename: str
    format: DocumentFormat
    chunk_count: int


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
