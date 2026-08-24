"""Casos de uso del modo apuntes: resúmenes, tests y evaluación."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from chatbot.domain.documents import DocumentPurpose
from chatbot.domain.entities import Message, Role
from chatbot.domain.exceptions import DocumentNotFoundError, ValidationError
from chatbot.domain.ports import LLMPort, PromptRepositoryPort, StudyRepositoryPort, VectorStorePort
from chatbot.domain.prompts import (
    PROMPT_STUDY_EVALUATE,
    PROMPT_STUDY_QUIZ_GENERATE,
    PROMPT_STUDY_SUMMARY,
)
from chatbot.domain.study import (
    AnswerEvaluation,
    GeneratedQuestion,
    StudyProfile,
    StudyProfileStatus,
    StudyQuestion,
    StudySession,
    StudySessionMode,
    StudySessionStatus,
    StudySummaryResult,
    parse_study_json,
)

logger = logging.getLogger(__name__)

_SUMMARY_CHAR_LIMIT = 12_000
_BATCH_CHAR_LIMIT = 6_000
_QUIZ_DEFAULT_COUNT = 6
_EXAM_DEFAULT_COUNT = 12


class StudyService:
    def __init__(
        self,
        *,
        llm: LLMPort,
        prompts: PromptRepositoryPort,
        study_repo: StudyRepositoryPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._study_repo = study_repo
        self._vector_store = vector_store

    async def generate_profile(self, document_id: str) -> StudyProfile:
        doc = await self._vector_store.get_document(document_id)
        if doc is None:
            raise DocumentNotFoundError(document_id)
        if doc.purpose != DocumentPurpose.NOTES:
            raise ValidationError("Solo los documentos de tipo apuntes tienen perfil de estudio")

        pending = StudyProfile(
            document_id=document_id,
            status=StudyProfileStatus.PENDING,
        )
        await self._study_repo.upsert_profile(pending)

        try:
            chunks = await self._vector_store.get_chunks_by_document(document_id)
            if not chunks:
                raise ValidationError("El documento no tiene chunks indexados")

            full_text = "\n\n".join(c.content for c in chunks)
            summary_result = await self._summarize_text(full_text)
            ready = StudyProfile(
                document_id=document_id,
                status=StudyProfileStatus.READY,
                summary=summary_result.summary,
                key_concepts=summary_result.key_concepts,
                updated_at=datetime.now(UTC),
            )
            await self._study_repo.upsert_profile(ready)
            logger.info("Perfil de estudio generado", extra={"document_id": document_id})
            return ready
        except Exception as exc:  # pylint: disable=broad-except
            failed = StudyProfile(
                document_id=document_id,
                status=StudyProfileStatus.FAILED,
                error=str(exc),
                updated_at=datetime.now(UTC),
            )
            await self._study_repo.upsert_profile(failed)
            logger.exception("Error generando perfil de estudio", extra={"document_id": document_id})
            return failed

    async def get_profile(self, document_id: str) -> StudyProfile | None:
        return await self._study_repo.get_profile(document_id)

    async def regenerate_summary(self, document_id: str) -> StudyProfile:
        return await self.generate_profile(document_id)

    async def create_session(
        self,
        *,
        document_id: str,
        mode: StudySessionMode,
        question_count: int | None = None,
    ) -> StudySession:
        doc = await self._vector_store.get_document(document_id)
        if doc is None:
            raise DocumentNotFoundError(document_id)
        if doc.purpose != DocumentPurpose.NOTES:
            raise ValidationError("Solo los documentos apuntes admiten sesiones de estudio")

        profile = await self._study_repo.get_profile(document_id)
        if profile is None or profile.status != StudyProfileStatus.READY:
            raise ValidationError(
                "El documento aún no tiene resumen listo. Espera a que termine el procesamiento."
            )

        if mode == StudySessionMode.CHAT:
            session = StudySession(
                document_id=document_id,
                mode=mode,
                question_count=0,
            )
            await self._study_repo.create_session(session)
            return session

        count = question_count or (
            _QUIZ_DEFAULT_COUNT if mode == StudySessionMode.QUIZ else _EXAM_DEFAULT_COUNT
        )
        if count < 1 or count > 20:
            raise ValidationError("question_count debe estar entre 1 y 20")

        chunks = await self._vector_store.get_chunks_by_document(document_id)
        content = "\n\n".join(c.content for c in chunks[:12])
        questions = await self._generate_questions(
            summary=profile.summary,
            content=content,
            question_count=count,
            mode=mode.value,
        )

        session = StudySession(
            document_id=document_id,
            mode=mode,
            question_count=len(questions),
        )
        study_questions = [
            StudyQuestion(
                id=str(uuid4()),
                session_id=session.id,
                order=index,
                question=q.question,
                reference_answer=q.reference_answer,
            )
            for index, q in enumerate(questions, start=1)
        ]
        session.questions = study_questions
        session.question_count = len(study_questions)
        await self._study_repo.create_session(session)
        return session

    async def get_session(self, session_id: str) -> StudySession:
        session = await self._study_repo.get_session(session_id)
        if session is None:
            raise ValidationError(f"Sesión no encontrada: {session_id}")
        return session

    async def submit_answer(
        self,
        *,
        session_id: str,
        question_id: str,
        answer: str,
    ) -> AnswerEvaluation:
        session = await self.get_session(session_id)
        if session.status != StudySessionStatus.ACTIVE:
            raise ValidationError("La sesión ya está finalizada")

        question = next((q for q in session.questions if q.id == question_id), None)
        if question is None:
            raise ValidationError(f"Pregunta no encontrada: {question_id}")
        if question.evaluated_at is not None:
            raise ValidationError("Esta pregunta ya fue evaluada")

        user_answer = (answer or "").strip()
        if not user_answer:
            raise ValidationError("La respuesta no puede estar vacía")

        chunks = await self._vector_store.get_chunks_by_document(session.document_id)
        content = "\n\n".join(c.content for c in chunks[:8])
        evaluation = await self._evaluate_answer(
            question=question.question,
            reference_answer=question.reference_answer,
            rubric="Evalúa con precisión terminológica del material.",
            user_answer=user_answer,
            content=content,
            strict=session.mode == StudySessionMode.EXAM,
        )

        question.user_answer = user_answer
        question.score = evaluation.score
        question.feedback = evaluation.feedback
        question.evaluated_at = datetime.now(UTC)
        await self._study_repo.update_question(question)
        return evaluation

    async def finish_session(self, session_id: str) -> StudySession:
        session = await self.get_session(session_id)
        if session.mode == StudySessionMode.CHAT:
            raise ValidationError("Las sesiones de chat no se finalizan con nota")

        evaluated = [q for q in session.questions if q.score is not None]
        if not evaluated:
            raise ValidationError("No hay respuestas evaluadas")

        total = sum(q.score or 0.0 for q in evaluated)
        max_total = len(evaluated) * 10.0
        session.score = round((total / max_total) * 10.0, 2) if max_total else 0.0
        session.status = StudySessionStatus.COMPLETED
        session.finished_at = datetime.now(UTC)
        await self._study_repo.save_session(session)
        return await self.get_session(session_id)

    async def _summarize_text(self, text: str) -> StudySummaryResult:
        if len(text) <= _SUMMARY_CHAR_LIMIT:
            return await self._summarize_batch(text)

        partials: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + _BATCH_CHAR_LIMIT, len(text))
            batch = text[start:end]
            partial = await self._summarize_batch(batch)
            partials.append(partial.summary)
            start = end

        combined = "\n\n".join(partials)
        if len(combined) <= _SUMMARY_CHAR_LIMIT:
            return await self._summarize_batch(combined)

        return StudySummaryResult(
            summary=combined[: _SUMMARY_CHAR_LIMIT],
            key_concepts=(),
        )

    async def _summarize_batch(self, content: str) -> StudySummaryResult:
        template = await self._prompts.get(PROMPT_STUDY_SUMMARY)
        if not template:
            raise ValidationError("Prompt study_summary no configurado")
        prompt = template.replace("{content}", content)
        reply = await self._llm.generate(
            [Message(role=Role.USER, content=prompt)],
            system_prompt="Responde únicamente con JSON válido.",
        )
        data = parse_study_json(reply.content)
        summary = str(data.get("summary", "")).strip()
        concepts_raw = data.get("key_concepts", [])
        concepts = tuple(str(c).strip() for c in concepts_raw if str(c).strip())
        if not summary:
            raise ValidationError("El modelo no generó un resumen válido")
        return StudySummaryResult(summary=summary, key_concepts=concepts)

    async def _generate_questions(
        self,
        *,
        summary: str,
        content: str,
        question_count: int,
        mode: str,
    ) -> list[GeneratedQuestion]:
        template = await self._prompts.get(PROMPT_STUDY_QUIZ_GENERATE)
        if not template:
            raise ValidationError("Prompt study_quiz_generate no configurado")
        prompt = (
            template.replace("{question_count}", str(question_count))
            .replace("{mode}", mode)
            .replace("{summary}", summary)
            .replace("{content}", content[:8000])
        )
        reply = await self._llm.generate(
            [Message(role=Role.USER, content=prompt)],
            system_prompt="Responde únicamente con JSON válido (array).",
        )
        data = parse_study_json(reply.content)
        if not isinstance(data, list):
            raise ValidationError("Formato de preguntas inválido")
        questions: list[GeneratedQuestion] = []
        for item in data[:question_count]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            reference = str(item.get("reference_answer", "")).strip()
            rubric = str(item.get("rubric", "")).strip()
            if question and reference:
                questions.append(
                    GeneratedQuestion(
                        question=question,
                        reference_answer=reference,
                        rubric=rubric,
                    )
                )
        if not questions:
            raise ValidationError("No se generaron preguntas válidas")
        return questions

    async def _evaluate_answer(
        self,
        *,
        question: str,
        reference_answer: str,
        rubric: str,
        user_answer: str,
        content: str,
        strict: bool,
    ) -> AnswerEvaluation:
        template = await self._prompts.get(PROMPT_STUDY_EVALUATE)
        if not template:
            raise ValidationError("Prompt study_evaluate no configurado")
        prompt = (
            template.replace("{question}", question)
            .replace("{reference_answer}", reference_answer)
            .replace("{rubric}", rubric)
            .replace("{user_answer}", user_answer)
            .replace("{content}", content[:6000])
        )
        system = (
            "Evalúa con criterio estricto y objetivo. Responde solo con JSON válido."
            if strict
            else "Evalúa con criterio objetivo. Responde solo con JSON válido."
        )
        reply = await self._llm.generate(
            [Message(role=Role.USER, content=prompt)],
            system_prompt=system,
        )
        data = parse_study_json(reply.content)
        score = float(data.get("score", 0))
        max_score = float(data.get("max_score", 10))
        feedback = str(data.get("feedback", "")).strip()
        missing_raw = data.get("missing_points", [])
        missing = tuple(str(m).strip() for m in missing_raw if str(m).strip())
        is_correct = bool(data.get("is_correct", score >= 7))
        return AnswerEvaluation(
            score=min(max(score, 0.0), max_score),
            max_score=max_score,
            feedback=feedback or "Sin feedback.",
            missing_points=missing,
            is_correct=is_correct,
        )
