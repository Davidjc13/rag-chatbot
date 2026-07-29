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


class EvalSuiteConfigRequest(BaseModel):
    limit: int = Field(default=20, ge=1, le=500)
    distractors: int = Field(default=50, ge=0, le=5000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    seed: int = Field(default=42)
    generate: bool = False
    ragas: bool = False
    ragas_timeout: int = Field(default=600, ge=60, le=3600)


class EvalDatasetStatusResponse(BaseModel):
    dataset_id: str
    name: str
    hf_source: str
    passage_count: int
    qa_count: int
    imported_at: datetime | None
    import_stats: dict[str, object] = Field(default_factory=dict)


class EvalImportRequest(BaseModel):
    force: bool = False


class EvalSuiteCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: str | None = None
    config: EvalSuiteConfigRequest = Field(default_factory=EvalSuiteConfigRequest)
    sample_ids: list[int] | None = None


class EvalSuiteUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = None
    config: EvalSuiteConfigRequest | None = None
    sample_ids: list[int] | None = None


class EvalSuiteResponse(BaseModel):
    id: str
    name: str
    dataset_id: str
    description: str | None
    config: EvalSuiteConfigRequest
    sample_ids: list[int]
    created_at: datetime


class EvalSuiteListResponse(BaseModel):
    suites: list[EvalSuiteResponse]


class EvalRunStartRequest(BaseModel):
    suite_id: str | None = None
    name: str | None = None
    config: EvalSuiteConfigRequest | None = None
    use_db: bool = True


class EvalRunResponse(BaseModel):
    id: str
    suite_id: str | None
    dataset_id: str
    name: str | None
    status: str
    mode: str
    config: dict[str, object]
    retrieval_metrics: dict[str, object] | None
    ragas_metrics: dict[str, object] | None
    error: str | None
    started_at: datetime
    finished_at: datetime | None


class EvalRunListResponse(BaseModel):
    runs: list[EvalRunResponse]


class EvalRunSampleResponse(BaseModel):
    sample_id: int
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    retrieved_passage_ids: list[int]
    scores: list[float]


class EvalRunSamplesResponse(BaseModel):
    samples: list[EvalRunSampleResponse]
    total: int
    offset: int
    limit: int
