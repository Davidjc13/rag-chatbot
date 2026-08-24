"""Tests de ingesta con purpose apuntes."""

from __future__ import annotations

import io

import pytest
from docx import Document

from chatbot.application.services.ingestion_service import IngestionService
from chatbot.application.services.study_service import StudyService
from chatbot.application.services.table_aware_chunker import TableAwareChunker
from chatbot.domain.documents import DocumentPurpose
from chatbot.infrastructure.adapters.ingestion.parser_factory import DocumentParserFactory
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.persistence.memory_study_repository import (
    InMemoryStudyRepository,
)
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore
from tests.prompt_fixtures import default_prompt_repo
from tests.test_study_service import StudyJsonLLM


def _docx_bytes() -> bytes:
    document = Document()
    document.add_paragraph("Apuntes de historia del arte.")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_ingest_notes_sets_purpose_and_generates_profile() -> None:
    store = InMemoryVectorStore()
    study_repo = InMemoryStudyRepository()
    study_service = StudyService(
        llm=StudyJsonLLM(),
        prompts=default_prompt_repo(),
        study_repo=study_repo,
        vector_store=store,
    )
    service = IngestionService(
        parser_factory=DocumentParserFactory(),
        chunker=TableAwareChunker(chunk_size=400, chunk_overlap=40),
        embeddings=MockEmbeddingAdapter(),
        vector_store=store,
        study_service=study_service,
    )

    result = await service.ingest(
        filename="apuntes.docx",
        data=_docx_bytes(),
        purpose=DocumentPurpose.NOTES,
    )
    assert result.purpose == DocumentPurpose.NOTES

    doc = await store.get_document(result.document_id)
    assert doc is not None
    assert doc.purpose == DocumentPurpose.NOTES

    profile = await study_repo.get_profile(result.document_id)
    assert profile is not None
    assert profile.status.value == "ready"


@pytest.mark.asyncio
async def test_ingest_general_without_study_profile() -> None:
    store = InMemoryVectorStore()
    study_repo = InMemoryStudyRepository()
    study_service = StudyService(
        llm=StudyJsonLLM(),
        prompts=default_prompt_repo(),
        study_repo=study_repo,
        vector_store=store,
    )
    service = IngestionService(
        parser_factory=DocumentParserFactory(),
        chunker=TableAwareChunker(chunk_size=400, chunk_overlap=40),
        embeddings=MockEmbeddingAdapter(),
        vector_store=store,
        study_service=study_service,
    )

    result = await service.ingest(
        filename="doc.docx",
        data=_docx_bytes(),
        purpose=DocumentPurpose.GENERAL,
    )
    assert result.purpose == DocumentPurpose.GENERAL
    profile = await study_repo.get_profile(result.document_id)
    assert profile is None
