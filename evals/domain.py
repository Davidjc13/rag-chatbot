"""Entidades de dominio para evaluaciones RAGAS / BioASQ."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

BIOASQ_DATASET_ID = "bioasq"
EvalRunStatus = Literal["pending", "running", "completed", "failed"]
EvalRunMode = Literal["retrieval", "generate", "ragas"]


@dataclass(frozen=True, slots=True)
class EvalDatasetStatus:
    dataset_id: str
    name: str
    hf_source: str
    passage_count: int
    qa_count: int
    imported_at: datetime | None
    import_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalSuiteConfig:
    limit: int = 20
    distractors: int = 50
    top_k: int | None = None
    seed: int = 42
    generate: bool = False
    ragas: bool = False
    ragas_timeout: int = 600


@dataclass(frozen=True, slots=True)
class EvalSuite:
    id: str
    name: str
    dataset_id: str
    description: str | None
    config: EvalSuiteConfig
    sample_ids: tuple[int, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class EvalRunSummary:
    id: str
    suite_id: str | None
    dataset_id: str
    name: str | None
    status: EvalRunStatus
    mode: EvalRunMode
    config: dict[str, Any]
    retrieval_metrics: dict[str, Any] | None
    ragas_metrics: dict[str, Any] | None
    error: str | None
    started_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class EvalRunSample:
    sample_id: int
    question: str
    ground_truth: str
    answer: str
    contexts: tuple[str, ...]
    retrieved_passage_ids: tuple[int, ...]
    scores: tuple[float, ...]
