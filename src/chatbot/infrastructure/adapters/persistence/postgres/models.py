"""Modelos ORM SQLAlchemy para PostgreSQL + pgvector."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Dimensión por defecto (nomic-embed-text). Se puede alinear con EMBEDDING_DIMENSION.
DEFAULT_EMBEDDING_DIMENSION = 768


class Base(DeclarativeBase):
    pass


class ConversationModel(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    messages: Mapped[list[MessageModel]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageModel.created_at",
    )


class MessageModel(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    conversation: Mapped[ConversationModel] = relationship(back_populates="messages")


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    format: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


class ChunkModel(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
    )
    embedding = mapped_column(Vector(DEFAULT_EMBEDDING_DIMENSION), nullable=False)


class PromptModel(Base):
    __tablename__ = "prompts"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class EvalDatasetModel(Base):
    __tablename__ = "eval_datasets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    hf_source: Mapped[str] = mapped_column(String(512), nullable=False)
    passage_count: Mapped[int] = mapped_column(Integer, default=0)
    qa_count: Mapped[int] = mapped_column(Integer, default=0)
    import_stats: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvalPassageModel(Base):
    __tablename__ = "eval_passages"

    dataset_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("eval_datasets.id", ondelete="CASCADE"),
        primary_key=True,
    )
    passage_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)


class EvalQASampleModel(Base):
    __tablename__ = "eval_qa_samples"

    dataset_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("eval_datasets.id", ondelete="CASCADE"),
        primary_key=True,
    )
    sample_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str] = mapped_column(Text, nullable=False)
    relevant_passage_ids: Mapped[list[int]] = mapped_column(JSONB, nullable=False)


class EvalSuiteModel(Base):
    __tablename__ = "eval_suites"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("eval_datasets.id", ondelete="CASCADE"),
        index=True,
    )
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    sample_ids: Mapped[list[int]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


class EvalRunModel(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    suite_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("eval_suites.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="retrieval")
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    retrieval_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    ragas_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvalRunResultModel(Base):
    __tablename__ = "eval_run_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        index=True,
    )
    sample_id: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    contexts: Mapped[list[str]] = mapped_column(JSONB, default=list)
    retrieved_passage_ids: Mapped[list[int]] = mapped_column(JSONB, default=list)
    scores: Mapped[list[float]] = mapped_column(JSONB, default=list)
