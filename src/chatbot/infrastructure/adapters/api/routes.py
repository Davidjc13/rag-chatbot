"""Rutas HTTP del chatbot."""

from __future__ import annotations

# pylint: disable=too-many-lines

import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from chatbot.application.services.chat_service import (
    ChatService,
    StreamCancelled,
    StreamDone,
    StreamMeta,
    StreamThinking,
    StreamToken,
    StreamTool,
)
from chatbot.application.services.eval_service import EvalService
from chatbot.application.services.ingestion_service import IngestionService
from chatbot.application.services.study_service import StudyService
from chatbot.application.services.transcription_service import TranscriptionService
from chatbot.core.env import Env
from chatbot.domain.documents import DocumentPurpose
from chatbot.domain.exceptions import ChatbotError
from chatbot.domain.ports import LLMPort
from chatbot.domain.study import StudySessionMode
from chatbot.infrastructure.adapters.api.mime_validation import validate_upload
from chatbot.infrastructure.adapters.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
    ConversationResponse,
    ConversationSummaryResponse,
    ConversationUpdateRequest,
    DocumentChunkResponse,
    DocumentChunksResponse,
    DocumentListResponse,
    DocumentSummaryResponse,
    EvalABTestRequest,
    EvalComparisonResponse,
    EvalComparisonSampleResponse,
    EvalClearResponse,
    EvalDatasetListResponse,
    EvalDatasetStatusResponse,
    EvalExperimentListResponse,
    EvalExperimentResponse,
    EvalImportRequest,
    EvalRunListResponse,
    EvalRunResponse,
    EvalRunSampleResponse,
    EvalRunSamplesResponse,
    EvalRunStartRequest,
    EvalSuiteConfigRequest,
    EvalSuiteCreateRequest,
    EvalSuiteListResponse,
    EvalSuiteResponse,
    EvalSuiteUpdateRequest,
    HealthResponse,
    IngestionResponse,
    MessageResponse,
    ModelsResponse,
    StudyAnswerRequest,
    StudyAnswerResponse,
    StudyFinishResponse,
    StudyProfileResponse,
    StudyQuestionResponse,
    StudySessionCreateRequest,
    StudySessionResponse,
    TranscriptionResponse,
)
from evals.domain import (
    EvalComparisonResult,
    EvalExperiment,
    EvalRunSummary,
    EvalSuite,
    EvalSuiteConfig,
)
from evals.json_dataset import dataset_template_path

_EVAL_STATIC_DIR = Path(__file__).resolve().parent / "static"

router = APIRouter()


def _chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def _ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def _study_service(request: Request) -> StudyService:
    return request.app.state.study_service


def _eval_service(request: Request) -> EvalService:
    return request.app.state.eval_service


def _transcription_service(request: Request) -> TranscriptionService:
    return request.app.state.transcription_service


def _llm(request: Request) -> LLMPort:
    return request.app.state.llm


def _env(request: Request) -> Env:
    return request.app.state.settings


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health(request: Request) -> HealthResponse:
    env = _env(request)
    llm = _llm(request)
    healthy = await llm.health_check()
    return HealthResponse(
        status="ok" if healthy else "degraded",
        llm_provider=env.llm_provider,
        llm_model=llm.model_name,
        llm_healthy=healthy,
    )


@router.get("/models", response_model=ModelsResponse, tags=["chat"])
async def list_models(request: Request) -> ModelsResponse:
    llm = _llm(request)
    models = await llm.list_models()
    active = llm.model_name.removeprefix("ollama/")
    if not models:
        models = [active]
    if active not in models:
        models = [active, *models]
    return ModelsResponse(models=models, active=active)


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service = _chat_service(request)
    reply = await service.chat(
        payload.message,
        conversation_id=payload.conversation_id,
        retrieval_backend=payload.retrieval_backend,
        model=payload.model,
        mode=payload.mode,
        document_id=payload.document_id,
    )
    return ChatResponse(
        conversation_id=reply.conversation_id,
        reply=MessageResponse(
            role=reply.message.role.value,
            content=reply.message.content,
            created_at=reply.message.created_at,
        ),
        model=reply.model,
    )


@router.post("/chat/stream", tags=["chat"])
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    service = _chat_service(request)

    async def is_cancelled() -> bool:
        return await request.is_disconnected()

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in service.chat_stream(
                payload.message,
                conversation_id=payload.conversation_id,
                retrieval_backend=payload.retrieval_backend,
                model=payload.model,
                mode=payload.mode,
                document_id=payload.document_id,
                is_cancelled=is_cancelled,
            ):
                if isinstance(event, StreamMeta):
                    yield _sse(
                        "meta",
                        {
                            "conversation_id": event.conversation_id,
                            "model": event.model,
                            "sources": list(event.sources),
                        },
                    )
                elif isinstance(event, StreamThinking):
                    yield _sse("thinking", {"content": event.content})
                elif isinstance(event, StreamTool):
                    yield _sse(
                        "tool",
                        {
                            "name": event.name,
                            "status": event.status,
                            "query": event.query,
                            "output_preview": event.output_preview,
                        },
                    )
                elif isinstance(event, StreamToken):
                    yield _sse("token", {"content": event.content})
                elif isinstance(event, StreamDone):
                    yield _sse("done", {"conversation_id": event.conversation_id})
                elif isinstance(event, StreamCancelled):
                    yield _sse("cancelled", {"conversation_id": event.conversation_id})
        except ChatbotError as exc:
            yield _sse("error", {"code": exc.code, "error": exc.message})
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            yield _sse(
                "error",
                {"code": "internal_error", "error": "Error interno del servidor"},
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/transcribe", response_model=TranscriptionResponse, tags=["chat"])
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
) -> TranscriptionResponse:
    service = _transcription_service(request)
    data = await file.read()
    text = await service.transcribe(data=data, content_type=file.content_type)
    return TranscriptionResponse(text=text)


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    tags=["chat"],
)
async def list_conversations(request: Request) -> ConversationListResponse:
    service = _chat_service(request)
    items = await service.list_conversations()
    return ConversationListResponse(
        conversations=[
            ConversationSummaryResponse(
                id=item.id,
                title=item.title,
                updated_at=item.updated_at,
                created_at=item.created_at,
                preview=item.preview,
            )
            for item in items
        ]
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    tags=["chat"],
)
async def get_conversation(conversation_id: str, request: Request) -> ConversationResponse:
    service = _chat_service(request)
    conversation = await service.get_conversation(conversation_id)
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        messages=[
            MessageResponse(
                role=m.role.value,
                content=m.content,
                created_at=m.created_at,
            )
            for m in conversation.messages
        ],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    tags=["chat"],
)
async def rename_conversation(
    conversation_id: str,
    payload: ConversationUpdateRequest,
    request: Request,
) -> ConversationResponse:
    service = _chat_service(request)
    conversation = await service.rename_conversation(conversation_id, payload.title)
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        messages=[
            MessageResponse(
                role=m.role.value,
                content=m.content,
                created_at=m.created_at,
            )
            for m in conversation.messages
        ],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.delete("/conversations/{conversation_id}", status_code=204, tags=["chat"])
async def delete_conversation(conversation_id: str, request: Request) -> None:
    service = _chat_service(request)
    await service.delete_conversation(conversation_id)


@router.post("/documents", response_model=IngestionResponse, tags=["documents"])
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    purpose: str = Form(default="general"),
) -> IngestionResponse:
    service = _ingestion_service(request)
    data = await file.read()
    filename = file.filename or "upload.bin"
    validate_upload(filename=filename, content_type=file.content_type, data=data)
    try:
        doc_purpose = DocumentPurpose(purpose)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="purpose debe ser general o notes") from exc
    result = await service.ingest(filename=filename, data=data, purpose=doc_purpose)
    return IngestionResponse(
        document_id=result.document_id,
        filename=result.filename,
        format=result.format.value,
        chunk_count=result.chunk_count,
        purpose=result.purpose.value,
    )


@router.put("/documents/{document_id}", response_model=IngestionResponse, tags=["documents"])
async def replace_document(
    document_id: str,
    request: Request,
    file: UploadFile = File(...),
    purpose: str | None = Form(default=None),
) -> IngestionResponse:
    service = _ingestion_service(request)
    data = await file.read()
    filename = file.filename or "upload.bin"
    validate_upload(filename=filename, content_type=file.content_type, data=data)
    doc_purpose = DocumentPurpose.GENERAL
    if purpose:
        try:
            doc_purpose = DocumentPurpose(purpose)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="purpose debe ser general o notes") from exc
    result = await service.ingest(
        filename=filename,
        data=data,
        purpose=doc_purpose,
        document_id=document_id,
    )
    return IngestionResponse(
        document_id=result.document_id,
        filename=result.filename,
        format=result.format.value,
        chunk_count=result.chunk_count,
        purpose=result.purpose.value,
    )


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
async def list_documents(
    request: Request,
    purpose: str | None = Query(default=None),
) -> DocumentListResponse:
    service = _ingestion_service(request)
    doc_purpose: DocumentPurpose | None = None
    if purpose:
        try:
            doc_purpose = DocumentPurpose(purpose)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="purpose inválido") from exc
    documents = await service.list_documents(purpose=doc_purpose)
    return DocumentListResponse(
        documents=[
            DocumentSummaryResponse(
                id=doc.id,
                filename=doc.filename,
                format=doc.format.value,
                chunk_count=doc.chunk_count,
                created_at=doc.created_at,
                purpose=doc.purpose.value,
            )
            for doc in documents
        ]
    )


@router.get(
    "/documents/{document_id}/chunks",
    response_model=DocumentChunksResponse,
    tags=["documents"],
)
async def list_document_chunks(document_id: str, request: Request) -> DocumentChunksResponse:
    service = _ingestion_service(request)
    chunks = await service.list_document_chunks(document_id)
    return DocumentChunksResponse(
        document_id=document_id,
        chunks=[
            DocumentChunkResponse(
                id=chunk.id,
                index=int(chunk.metadata.get("chunk_index", idx)),
                content=chunk.content,
                metadata={
                    key: value
                    for key, value in chunk.metadata.items()
                    if key != "embedding"
                },
            )
            for idx, chunk in enumerate(chunks)
        ],
    )


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"])
async def delete_document(document_id: str, request: Request) -> None:
    service = _ingestion_service(request)
    await service.delete_document(document_id)


def _study_profile_response(profile) -> StudyProfileResponse:
    return StudyProfileResponse(
        document_id=profile.document_id,
        status=profile.status.value,
        summary=profile.summary,
        key_concepts=list(profile.key_concepts),
        error=profile.error,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _study_session_response(session, *, reveal_answers: bool = False) -> StudySessionResponse:
    questions: list[StudyQuestionResponse] = []
    for question in session.questions:
        show_reference = reveal_answers or session.mode.value == "quiz"
        if session.status.value == "completed":
            show_reference = True
        questions.append(
            StudyQuestionResponse(
                id=question.id,
                order=question.order,
                question=question.question,
                reference_answer=question.reference_answer if show_reference else None,
                user_answer=question.user_answer,
                score=question.score,
                feedback=question.feedback
                if reveal_answers or session.mode.value == "quiz"
                else "",
                evaluated_at=question.evaluated_at,
            )
        )
    return StudySessionResponse(
        id=session.id,
        document_id=session.document_id,
        mode=session.mode.value,
        status=session.status.value,
        score=session.score,
        question_count=session.question_count,
        questions=questions,
        created_at=session.created_at,
        finished_at=session.finished_at,
    )


@router.get(
    "/documents/{document_id}/study",
    response_model=StudyProfileResponse,
    tags=["study"],
)
async def get_document_study_profile(
    document_id: str,
    request: Request,
) -> StudyProfileResponse:
    service = _study_service(request)
    profile = await service.get_profile(document_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Perfil de estudio no encontrado")
    return _study_profile_response(profile)


@router.post(
    "/documents/{document_id}/study/summary",
    response_model=StudyProfileResponse,
    tags=["study"],
)
async def regenerate_study_summary(
    document_id: str,
    request: Request,
) -> StudyProfileResponse:
    service = _study_service(request)
    profile = await service.regenerate_summary(document_id)
    return _study_profile_response(profile)


@router.post("/study/sessions", response_model=StudySessionResponse, tags=["study"])
async def create_study_session(
    payload: StudySessionCreateRequest,
    request: Request,
) -> StudySessionResponse:
    service = _study_service(request)
    session = await service.create_session(
        document_id=payload.document_id,
        mode=StudySessionMode(payload.mode),
        question_count=payload.question_count,
    )
    return _study_session_response(session)


@router.get(
    "/study/sessions/{session_id}",
    response_model=StudySessionResponse,
    tags=["study"],
)
async def get_study_session(session_id: str, request: Request) -> StudySessionResponse:
    service = _study_service(request)
    session = await service.get_session(session_id)
    reveal = session.status.value == "completed"
    return _study_session_response(session, reveal_answers=reveal)


@router.post(
    "/study/sessions/{session_id}/answers",
    response_model=StudyAnswerResponse,
    tags=["study"],
)
async def submit_study_answer(
    session_id: str,
    payload: StudyAnswerRequest,
    request: Request,
) -> StudyAnswerResponse:
    service = _study_service(request)
    evaluation = await service.submit_answer(
        session_id=session_id,
        question_id=payload.question_id,
        answer=payload.answer,
    )
    return StudyAnswerResponse(
        score=evaluation.score,
        max_score=evaluation.max_score,
        feedback=evaluation.feedback,
        missing_points=list(evaluation.missing_points),
        is_correct=evaluation.is_correct,
    )


@router.post(
    "/study/sessions/{session_id}/finish",
    response_model=StudyFinishResponse,
    tags=["study"],
)
async def finish_study_session(
    session_id: str,
    request: Request,
) -> StudyFinishResponse:
    service = _study_service(request)
    session = await service.finish_session(session_id)
    evaluated = [q for q in session.questions if q.score is not None]
    lines = [
        f"Nota final: {session.score}/10",
        f"Preguntas respondidas: {len(evaluated)}/{session.question_count}",
        "",
    ]
    for question in session.questions:
        if question.score is None:
            continue
        lines.append(f"P{question.order} ({question.score}/10): {question.question}")
        lines.append(f"  Feedback: {question.feedback}")
        lines.append("")
    report = "\n".join(lines).strip()
    return StudyFinishResponse(
        session=_study_session_response(session, reveal_answers=True),
        report=report,
    )


def _suite_config_from_request(payload: EvalSuiteConfigRequest) -> EvalSuiteConfig:
    return EvalSuiteConfig(
        limit=payload.limit,
        distractors=payload.distractors,
        top_k=payload.top_k,
        seed=payload.seed,
        generate=payload.generate,
        ragas=payload.ragas,
        ragas_timeout=payload.ragas_timeout,
        deepeval=payload.deepeval,
        deepeval_timeout=payload.deepeval_timeout,
        deepeval_metrics=tuple(payload.deepeval_metrics),
        llm_model=payload.llm_model,
        llm_provider=payload.llm_provider,
    )


def _suite_config_to_response(config: EvalSuiteConfig) -> EvalSuiteConfigRequest:
    return EvalSuiteConfigRequest(
        limit=config.limit,
        distractors=config.distractors,
        top_k=config.top_k,
        seed=config.seed,
        generate=config.generate,
        ragas=config.ragas,
        ragas_timeout=config.ragas_timeout,
        deepeval=config.deepeval,
        deepeval_timeout=config.deepeval_timeout,
        deepeval_metrics=list(config.deepeval_metrics),
        llm_model=config.llm_model,
        llm_provider=config.llm_provider,  # type: ignore[arg-type]
    )


def _suite_to_response(suite: EvalSuite) -> EvalSuiteResponse:
    return EvalSuiteResponse(
        id=suite.id,
        name=suite.name,
        dataset_id=suite.dataset_id,
        description=suite.description,
        config=_suite_config_to_response(suite.config),
        sample_ids=list(suite.sample_ids),
        created_at=suite.created_at,
    )


def _run_to_response(run: EvalRunSummary) -> EvalRunResponse:
    return EvalRunResponse(
        id=run.id,
        suite_id=run.suite_id,
        dataset_id=run.dataset_id,
        name=run.name,
        status=run.status,
        mode=run.mode,
        config=run.config,
        retrieval_metrics=run.retrieval_metrics,
        ragas_metrics=run.ragas_metrics,
        deepeval_metrics=run.deepeval_metrics,
        experiment_id=run.experiment_id,
        variant_label=run.variant_label,
        error=run.error,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


def _experiment_to_response(item: EvalExperiment) -> EvalExperimentResponse:
    return EvalExperimentResponse(
        id=item.id,
        name=item.name,
        suite_id=item.suite_id,
        dataset_id=item.dataset_id,
        run_a_id=item.run_a_id,
        run_b_id=item.run_b_id,
        created_at=item.created_at,
    )


def _comparison_to_response(item: EvalComparisonResult) -> EvalComparisonResponse:
    return EvalComparisonResponse(
        run_a_id=item.run_a_id,
        run_b_id=item.run_b_id,
        run_a_name=item.run_a_name,
        run_b_name=item.run_b_name,
        retrieval_delta=item.retrieval_delta,
        ragas_delta=item.ragas_delta,
        deepeval_delta=item.deepeval_delta,
        win_rates=item.win_rates,
        samples=[
            EvalComparisonSampleResponse(
                sample_id=sample.sample_id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                answer_a=sample.answer_a,
                answer_b=sample.answer_b,
                retrieved_a=list(sample.retrieved_a),
                retrieved_b=list(sample.retrieved_b),
                hit_a=sample.hit_a,
                hit_b=sample.hit_b,
            )
            for sample in item.samples
        ],
    )


@router.get(
    "/evals/datasets",
    response_model=EvalDatasetListResponse,
    tags=["evals"],
)
async def list_eval_datasets(request: Request) -> EvalDatasetListResponse:
    service = _eval_service(request)
    datasets = await service.list_datasets()
    return EvalDatasetListResponse(
        datasets=[
            EvalDatasetStatusResponse(
                dataset_id=item.dataset_id,
                name=item.name,
                hf_source=item.hf_source,
                passage_count=item.passage_count,
                qa_count=item.qa_count,
                imported_at=item.imported_at,
                import_stats=item.import_stats,
            )
            for item in datasets
        ]
    )


@router.get("/evals/datasets/template", tags=["evals"])
async def download_dataset_template() -> FileResponse:
    static_path = _EVAL_STATIC_DIR / "dataset.template.json"
    path = static_path if static_path.is_file() else dataset_template_path()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Plantilla no encontrada")
    return FileResponse(path, filename="dataset.template.json", media_type="application/json")


@router.post(
    "/evals/datasets/json/import",
    response_model=EvalDatasetStatusResponse,
    tags=["evals"],
)
async def import_json_dataset(
    request: Request,
    file: UploadFile = File(...),
    dataset_id: str | None = None,
    force: bool = False,
) -> EvalDatasetStatusResponse:
    service = _eval_service(request)
    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="JSON inválido") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="El dataset debe ser un objeto JSON")
    try:
        status = await service.import_json_dataset(
            payload,
            dataset_id=dataset_id,
            force=force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EvalDatasetStatusResponse(
        dataset_id=status.dataset_id,
        name=status.name,
        hf_source=status.hf_source,
        passage_count=status.passage_count,
        qa_count=status.qa_count,
        imported_at=status.imported_at,
        import_stats=status.import_stats,
    )


@router.get(
    "/evals/datasets/bioasq",
    response_model=EvalDatasetStatusResponse | None,
    tags=["evals"],
)
async def get_bioasq_dataset_status(request: Request) -> EvalDatasetStatusResponse | None:
    service = _eval_service(request)
    status = await service.get_bioasq_status()
    if status is None:
        return None
    return EvalDatasetStatusResponse(
        dataset_id=status.dataset_id,
        name=status.name,
        hf_source=status.hf_source,
        passage_count=status.passage_count,
        qa_count=status.qa_count,
        imported_at=status.imported_at,
        import_stats=status.import_stats,
    )


@router.post(
    "/evals/datasets/bioasq/import",
    response_model=EvalDatasetStatusResponse,
    tags=["evals"],
)
async def import_bioasq_dataset(
    request: Request,
    payload: EvalImportRequest | None = None,
) -> EvalDatasetStatusResponse:
    service = _eval_service(request)
    status = await service.import_bioasq(force=bool(payload and payload.force))
    return EvalDatasetStatusResponse(
        dataset_id=status.dataset_id,
        name=status.name,
        hf_source=status.hf_source,
        passage_count=status.passage_count,
        qa_count=status.qa_count,
        imported_at=status.imported_at,
        import_stats=status.import_stats,
    )


@router.get("/evals/suites", response_model=EvalSuiteListResponse, tags=["evals"])
async def list_eval_suites(request: Request) -> EvalSuiteListResponse:
    service = _eval_service(request)
    suites = await service.list_suites()
    return EvalSuiteListResponse(suites=[_suite_to_response(s) for s in suites])


@router.post("/evals/suites", response_model=EvalSuiteResponse, tags=["evals"])
async def create_eval_suite(
    payload: EvalSuiteCreateRequest,
    request: Request,
) -> EvalSuiteResponse:
    service = _eval_service(request)
    suite = await service.create_suite(
        name=payload.name,
        description=payload.description,
        config=_suite_config_from_request(payload.config),
        sample_ids=payload.sample_ids,
        dataset_id=payload.dataset_id,
    )
    return _suite_to_response(suite)


@router.get("/evals/suites/{suite_id}", response_model=EvalSuiteResponse, tags=["evals"])
async def get_eval_suite(suite_id: str, request: Request) -> EvalSuiteResponse:
    service = _eval_service(request)
    suite = await service.get_suite(suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite no encontrada")
    return _suite_to_response(suite)


@router.patch("/evals/suites/{suite_id}", response_model=EvalSuiteResponse, tags=["evals"])
async def update_eval_suite(
    suite_id: str,
    payload: EvalSuiteUpdateRequest,
    request: Request,
) -> EvalSuiteResponse:
    service = _eval_service(request)
    suite = await service.update_suite(
        suite_id,
        name=payload.name,
        description=payload.description,
        config=_suite_config_from_request(payload.config) if payload.config else None,
        sample_ids=payload.sample_ids,
    )
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite no encontrada")
    return _suite_to_response(suite)


@router.delete("/evals/suites/{suite_id}", status_code=204, tags=["evals"])
async def delete_eval_suite(suite_id: str, request: Request) -> None:
    service = _eval_service(request)
    deleted = await service.delete_suite(suite_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Suite no encontrada")


@router.get("/evals/runs", response_model=EvalRunListResponse, tags=["evals"])
async def list_eval_runs(request: Request) -> EvalRunListResponse:
    service = _eval_service(request)
    runs = await service.list_runs()
    return EvalRunListResponse(runs=[_run_to_response(r) for r in runs])


@router.post("/evals/runs", response_model=EvalRunResponse, tags=["evals"])
async def start_eval_run(
    payload: EvalRunStartRequest,
    request: Request,
) -> EvalRunResponse:
    service = _eval_service(request)
    config = _suite_config_from_request(payload.config) if payload.config else None
    try:
        run = await service.start_run(
            suite_id=payload.suite_id,
            name=payload.name,
            config=config,
            use_db=payload.use_db,
        )
    except ValueError as exc:
        raise ChatbotError(str(exc), code="invalid_request") from exc
    return _run_to_response(run)


@router.get("/evals/runs/{run_id}", response_model=EvalRunResponse, tags=["evals"])
async def get_eval_run(run_id: str, request: Request) -> EvalRunResponse:
    service = _eval_service(request)
    run = await service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run no encontrado")
    return _run_to_response(run)


@router.delete("/evals/runs/{run_id}", status_code=204, tags=["evals"])
async def delete_eval_run(run_id: str, request: Request) -> None:
    service = _eval_service(request)
    deleted = await service.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run no encontrado")


@router.post("/evals/runs/{run_id}/cancel", response_model=EvalRunResponse, tags=["evals"])
async def cancel_eval_run(run_id: str, request: Request) -> EvalRunResponse:
    service = _eval_service(request)
    cancelled = await service.cancel_run(run_id)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail="Solo se pueden cancelar runs en cola",
        )
    run = await service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run no encontrado")
    return _run_to_response(run)


@router.delete("/evals/runs", response_model=EvalClearResponse, tags=["evals"])
async def clear_eval_runs(request: Request) -> EvalClearResponse:
    service = _eval_service(request)
    runs_deleted, experiments_deleted = await service.clear_runs_and_experiments()
    return EvalClearResponse(
        runs_deleted=runs_deleted,
        experiments_deleted=experiments_deleted,
    )


@router.get(
    "/evals/runs/{run_id}/samples",
    response_model=EvalRunSamplesResponse,
    tags=["evals"],
)
async def get_eval_run_samples(
    run_id: str,
    request: Request,
    offset: int = 0,
    limit: int = 50,
) -> EvalRunSamplesResponse:
    service = _eval_service(request)
    run = await service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run no encontrado")
    samples, total = await service.get_run_samples(run_id, offset=offset, limit=limit)
    return EvalRunSamplesResponse(
        samples=[
            EvalRunSampleResponse(
                sample_id=s.sample_id,
                question=s.question,
                ground_truth=s.ground_truth,
                answer=s.answer,
                contexts=list(s.contexts),
                retrieved_passage_ids=list(s.retrieved_passage_ids),
                scores=list(s.scores),
            )
            for s in samples
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post("/evals/ab-test", response_model=EvalExperimentResponse, tags=["evals"])
async def start_ab_test(payload: EvalABTestRequest, request: Request) -> EvalExperimentResponse:
    service = _eval_service(request)
    try:
        experiment = await service.start_ab_test(
            suite_id=payload.suite_id,
            name=payload.name,
            variant_a_name=payload.variant_a.name,
            variant_b_name=payload.variant_b.name,
            variant_a_config=(
                _suite_config_from_request(payload.variant_a.config)
                if payload.variant_a.config
                else None
            ),
            variant_b_config=(
                _suite_config_from_request(payload.variant_b.config)
                if payload.variant_b.config
                else None
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _experiment_to_response(experiment)


@router.get("/evals/experiments", response_model=EvalExperimentListResponse, tags=["evals"])
async def list_eval_experiments(request: Request) -> EvalExperimentListResponse:
    service = _eval_service(request)
    experiments = await service.list_experiments()
    return EvalExperimentListResponse(
        experiments=[_experiment_to_response(item) for item in experiments]
    )


@router.get("/evals/compare", response_model=EvalComparisonResponse, tags=["evals"])
async def compare_eval_runs(
    request: Request,
    run_a: str,
    run_b: str,
) -> EvalComparisonResponse:
    service = _eval_service(request)
    try:
        result = await service.compare_runs(run_a, run_b)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _comparison_to_response(result)
