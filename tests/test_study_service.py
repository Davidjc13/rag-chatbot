"""Tests del servicio de estudio (modo apuntes)."""

from __future__ import annotations

import json

import pytest

from chatbot.application.services.study_service import StudyService
from chatbot.domain.documents import DocumentChunk, DocumentFormat, DocumentPurpose
from chatbot.domain.entities import Message, Role
from chatbot.domain.ports import LLMPort
from chatbot.domain.study import StudyProfileStatus, StudySessionMode
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.persistence.memory_study_repository import (
    InMemoryStudyRepository,
)
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore
from tests.prompt_fixtures import default_prompt_repo


class StudyJsonLLM(LLMPort):
    @property
    def model_name(self) -> str:
        return "study-json"

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> Message:
        content = messages[-1].content if messages else ""
        if "key_concepts" in content or "Material:" in content:
            payload = {"summary": "Resumen de prueba", "key_concepts": ["Concepto A"]}
        elif "reference_answer" in content and "Respuesta del estudiante" not in content:
            payload = [
                {
                    "question": "¿Qué es X?",
                    "reference_answer": "X es la respuesta",
                    "rubric": "Precisión",
                }
            ]
        else:
            payload = {
                "score": 7.5,
                "max_score": 10,
                "feedback": "Correcto en lo esencial",
                "missing_points": ["Detalle menor"],
                "is_correct": True,
            }
        return Message(role=Role.ASSISTANT, content=json.dumps(payload))

    async def generate_stream(self, messages, *, system_prompt=None, model=None):
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return [self.model_name]

    async def health_check(self) -> bool:
        return True


@pytest.fixture
async def study_setup() -> tuple[StudyService, InMemoryVectorStore, str]:
    store = InMemoryVectorStore()
    embeddings = MockEmbeddingAdapter()
    vectors = await embeddings.embed(["Contenido de apuntes sobre biología celular."])
    doc_id = "doc-notes-1"
    await store.upsert(
        [
            DocumentChunk(
                document_id=doc_id,
                content="Contenido de apuntes sobre biología celular.",
                metadata={
                    "filename": "bio.docx",
                    "format": DocumentFormat.DOCX.value,
                    "purpose": DocumentPurpose.NOTES.value,
                    "chunk_index": 0,
                },
                embedding=vectors[0],
            )
        ]
    )
    service = StudyService(
        llm=StudyJsonLLM(),
        prompts=default_prompt_repo(),
        study_repo=InMemoryStudyRepository(),
        vector_store=store,
    )
    return service, store, doc_id


@pytest.mark.asyncio
async def test_generate_profile(study_setup) -> None:
    service, _, doc_id = study_setup
    profile = await service.generate_profile(doc_id)
    assert profile.status == StudyProfileStatus.READY
    assert "Resumen" in profile.summary
    assert profile.key_concepts == ("Concepto A",)


@pytest.mark.asyncio
async def test_create_quiz_session(study_setup) -> None:
    service, _, doc_id = study_setup
    await service.generate_profile(doc_id)
    session = await service.create_session(
        document_id=doc_id,
        mode=StudySessionMode.QUIZ,
        question_count=1,
    )
    assert session.question_count == 1
    assert session.questions[0].question == "¿Qué es X?"


@pytest.mark.asyncio
async def test_submit_and_finish_session(study_setup) -> None:
    service, _, doc_id = study_setup
    await service.generate_profile(doc_id)
    session = await service.create_session(
        document_id=doc_id,
        mode=StudySessionMode.QUIZ,
        question_count=1,
    )
    question = session.questions[0]
    evaluation = await service.submit_answer(
        session_id=session.id,
        question_id=question.id,
        answer="X es la respuesta del alumno",
    )
    assert evaluation.score == 7.5
    finished = await service.finish_session(session.id)
    assert finished.score is not None
    assert finished.status.value == "completed"
