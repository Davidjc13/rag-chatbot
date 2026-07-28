"""Esquemas Pydantic de la API (adaptador primario)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16_000)
    conversation_id: str | None = None
    retrieval_backend: Literal["postgres", "neo4j"] = "postgres"


class MessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    conversation_id: str
    reply: MessageResponse
    model: str


class ConversationResponse(BaseModel):
    id: str
    messages: list[MessageResponse]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    llm_healthy: bool


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: str | None = None


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    format: str
    chunk_count: int


class DocumentSummaryResponse(BaseModel):
    id: str
    filename: str
    format: str
    chunk_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummaryResponse]
