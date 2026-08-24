"""Repositorio en memoria para tests de estudio."""

from __future__ import annotations

import threading

from chatbot.domain.study import StudyProfile, StudyQuestion, StudySession


class InMemoryStudyRepository:
    def __init__(self) -> None:
        self._profiles: dict[str, StudyProfile] = {}
        self._sessions: dict[str, StudySession] = {}
        self._lock = threading.RLock()

    async def upsert_profile(self, profile: StudyProfile) -> None:
        with self._lock:
            self._profiles[profile.document_id] = profile

    async def get_profile(self, document_id: str) -> StudyProfile | None:
        with self._lock:
            return self._profiles.get(document_id)

    async def create_session(self, session: StudySession) -> None:
        with self._lock:
            self._sessions[session.id] = session

    async def get_session(self, session_id: str) -> StudySession | None:
        with self._lock:
            stored = self._sessions.get(session_id)
            if stored is None:
                return None
            return StudySession(
                id=stored.id,
                document_id=stored.document_id,
                mode=stored.mode,
                status=stored.status,
                score=stored.score,
                question_count=stored.question_count,
                questions=list(stored.questions),
                created_at=stored.created_at,
                finished_at=stored.finished_at,
            )

    async def save_session(self, session: StudySession) -> None:
        with self._lock:
            if session.id not in self._sessions:
                raise ValueError(f"Sesión no encontrada: {session.id}")
            self._sessions[session.id] = session

    async def update_question(self, question: StudyQuestion) -> None:
        with self._lock:
            session = self._sessions.get(question.session_id)
            if session is None:
                raise ValueError(f"Sesión no encontrada: {question.session_id}")
            updated = [
                question if q.id == question.id else q for q in session.questions
            ]
            session.questions = updated
