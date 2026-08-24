"""Entidades de dominio para el modo apuntes / estudio."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class StudyProfileStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class StudySessionMode(StrEnum):
    CHAT = "chat"
    QUIZ = "quiz"
    EXAM = "exam"


class StudySessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class StudyProfile:
    document_id: str
    status: StudyProfileStatus
    summary: str = ""
    key_concepts: tuple[str, ...] = ()
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class StudyQuestion:
    id: str
    session_id: str
    order: int
    question: str
    reference_answer: str
    user_answer: str = ""
    score: float | None = None
    feedback: str = ""
    evaluated_at: datetime | None = None


@dataclass(slots=True)
class StudySession:
    id: str = field(default_factory=lambda: str(uuid4()))
    document_id: str = ""
    mode: StudySessionMode = StudySessionMode.QUIZ
    status: StudySessionStatus = StudySessionStatus.ACTIVE
    score: float | None = None
    question_count: int = 0
    questions: list[StudyQuestion] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AnswerEvaluation:
    score: float
    max_score: float
    feedback: str
    missing_points: tuple[str, ...]
    is_correct: bool


@dataclass(frozen=True, slots=True)
class GeneratedQuestion:
    question: str
    reference_answer: str
    rubric: str = ""


@dataclass(frozen=True, slots=True)
class StudySummaryResult:
    summary: str
    key_concepts: tuple[str, ...]


def parse_study_json(raw: str) -> Any:
    """Extrae JSON de una respuesta LLM que puede incluir markdown."""
    import json
    import re

    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)
