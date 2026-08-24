"""Repositorio PostgreSQL para perfiles y sesiones de estudio."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from chatbot.domain.study import (
    StudyProfile,
    StudyProfileStatus,
    StudyQuestion,
    StudySession,
    StudySessionMode,
    StudySessionStatus,
)
from chatbot.infrastructure.adapters.persistence.postgres.models import (
    DocumentStudyProfileModel,
    StudySessionModel,
    StudySessionQuestionModel,
)


def _profile_from_model(model: DocumentStudyProfileModel) -> StudyProfile:
    return StudyProfile(
        document_id=model.document_id,
        status=StudyProfileStatus(model.status),
        summary=model.summary or "",
        key_concepts=tuple(model.key_concepts or []),
        error=model.error,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _question_from_model(model: StudySessionQuestionModel) -> StudyQuestion:
    return StudyQuestion(
        id=model.id,
        session_id=model.session_id,
        order=model.order,
        question=model.question,
        reference_answer=model.reference_answer,
        user_answer=model.user_answer or "",
        score=model.score,
        feedback=model.feedback or "",
        evaluated_at=model.evaluated_at,
    )


def _session_from_model(model: StudySessionModel) -> StudySession:
    return StudySession(
        id=model.id,
        document_id=model.document_id,
        mode=StudySessionMode(model.mode),
        status=StudySessionStatus(model.status),
        score=model.score,
        question_count=model.question_count,
        questions=[_question_from_model(q) for q in model.questions],
        created_at=model.created_at,
        finished_at=model.finished_at,
    )


class PostgresStudyRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def upsert_profile(self, profile: StudyProfile) -> None:
        async with self._session_factory() as session:
            existing = await session.get(DocumentStudyProfileModel, profile.document_id)
            if existing is None:
                session.add(
                    DocumentStudyProfileModel(
                        document_id=profile.document_id,
                        summary=profile.summary,
                        key_concepts=list(profile.key_concepts),
                        status=profile.status.value,
                        error=profile.error,
                        created_at=profile.created_at,
                        updated_at=profile.updated_at,
                    )
                )
            else:
                existing.summary = profile.summary
                existing.key_concepts = list(profile.key_concepts)
                existing.status = profile.status.value
                existing.error = profile.error
                existing.updated_at = profile.updated_at or datetime.now(UTC)
            await session.commit()

    async def get_profile(self, document_id: str) -> StudyProfile | None:
        async with self._session_factory() as session:
            model = await session.get(DocumentStudyProfileModel, document_id)
            if model is None:
                return None
            return _profile_from_model(model)

    async def create_session(self, study_session: StudySession) -> None:
        async with self._session_factory() as session:
            session.add(
                StudySessionModel(
                    id=study_session.id,
                    document_id=study_session.document_id,
                    mode=study_session.mode.value,
                    status=study_session.status.value,
                    score=study_session.score,
                    question_count=study_session.question_count,
                    created_at=study_session.created_at,
                    finished_at=study_session.finished_at,
                )
            )
            for question in study_session.questions:
                session.add(
                    StudySessionQuestionModel(
                        id=question.id,
                        session_id=study_session.id,
                        order=question.order,
                        question=question.question,
                        reference_answer=question.reference_answer,
                        user_answer=question.user_answer,
                        score=question.score,
                        feedback=question.feedback,
                        evaluated_at=question.evaluated_at,
                    )
                )
            await session.commit()

    async def get_session(self, session_id: str) -> StudySession | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(StudySessionModel)
                .where(StudySessionModel.id == session_id)
                .options(selectinload(StudySessionModel.questions))
            )
            model = result.scalar_one_or_none()
            if model is None:
                return None
            return _session_from_model(model)

    async def save_session(self, study_session: StudySession) -> None:
        async with self._session_factory() as session:
            model = await session.get(StudySessionModel, study_session.id)
            if model is None:
                raise ValueError(f"Sesión no encontrada: {study_session.id}")
            model.status = study_session.status.value
            model.score = study_session.score
            model.question_count = study_session.question_count
            model.finished_at = study_session.finished_at
            await session.commit()

    async def update_question(self, question: StudyQuestion) -> None:
        async with self._session_factory() as session:
            model = await session.get(StudySessionQuestionModel, question.id)
            if model is None:
                raise ValueError(f"Pregunta no encontrada: {question.id}")
            model.user_answer = question.user_answer
            model.score = question.score
            model.feedback = question.feedback
            model.evaluated_at = question.evaluated_at
            await session.commit()
