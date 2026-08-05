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


class TranscriptionResponse(BaseModel):
    text: str


class EvalSuiteConfigRequest(BaseModel):
    limit: int = Field(default=20, ge=1, le=500)
    distractors: int = Field(default=50, ge=0, le=5000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    seed: int = Field(default=42)
    generate: bool = False
    ragas: bool = False
    ragas_timeout: int = Field(default=600, ge=60, le=3600)
    deepeval: bool = False
    deepeval_timeout: int = Field(default=600, ge=60, le=3600)
    deepeval_metrics: list[str] = Field(
        default_factory=lambda: ["answer_relevancy", "faithfulness", "contextual_relevancy"]
    )
    llm_model: str | None = None
    llm_provider: Literal["litellm", "ollama", "mock"] | None = None


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


class EvalJsonImportRequest(BaseModel):
    dataset_id: str | None = Field(default=None, max_length=63)
    force: bool = False


class EvalDatasetListResponse(BaseModel):
    datasets: list[EvalDatasetStatusResponse]


class EvalSuiteCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: str | None = None
    dataset_id: str = Field(default="bioasq", min_length=1, max_length=64)
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
    deepeval_metrics: dict[str, object] | None
    experiment_id: str | None = None
    variant_label: str | None = None
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


class EvalABVariantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    config: EvalSuiteConfigRequest | None = None


class EvalABTestRequest(BaseModel):
    suite_id: str
    name: str | None = None
    variant_a: EvalABVariantRequest
    variant_b: EvalABVariantRequest


class EvalExperimentResponse(BaseModel):
    id: str
    name: str
    suite_id: str
    dataset_id: str
    run_a_id: str
    run_b_id: str
    created_at: datetime


class EvalExperimentListResponse(BaseModel):
    experiments: list[EvalExperimentResponse]


class EvalClearResponse(BaseModel):
    runs_deleted: int
    experiments_deleted: int


class EvalComparisonSampleResponse(BaseModel):
    sample_id: int
    question: str
    ground_truth: str
    answer_a: str
    answer_b: str
    retrieved_a: list[int]
    retrieved_b: list[int]
    hit_a: bool
    hit_b: bool


class EvalComparisonResponse(BaseModel):
    run_a_id: str
    run_b_id: str
    run_a_name: str | None
    run_b_name: str | None
    retrieval_delta: dict[str, float | None]
    ragas_delta: dict[str, float | None]
    deepeval_delta: dict[str, float | None]
    win_rates: dict[str, float | None]
    samples: list[EvalComparisonSampleResponse]
