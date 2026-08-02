"""Rutas HTTP del chatbot."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from chatbot.application.services.chat_service import (
    ChatService,
    StreamDone,
    StreamMeta,
    StreamThinking,
    StreamToken,
)
from chatbot.application.services.eval_service import EvalService
from chatbot.application.services.ingestion_service import IngestionService
from chatbot.core.env import Env
from chatbot.domain.exceptions import ChatbotError
from chatbot.domain.ports import LLMPort
from chatbot.infrastructure.adapters.api.mime_validation import validate_upload
from chatbot.infrastructure.adapters.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
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
)
from evals.domain import EvalComparisonResult, EvalExperiment, EvalRunSummary, EvalSuite, EvalSuiteConfig
from evals.json_dataset import dataset_template_path

_EVAL_STATIC_DIR = Path(__file__).resolve().parent / "static"

router = APIRouter()


def _chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def _ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def _eval_service(request: Request) -> EvalService:
    return request.app.state.eval_service


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


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service = _chat_service(request)
    reply = await service.chat(
        payload.message,
        conversation_id=payload.conversation_id,
        retrieval_backend=payload.retrieval_backend,
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

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in service.chat_stream(
                payload.message,
                conversation_id=payload.conversation_id,
                retrieval_backend=payload.retrieval_backend,
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
                elif isinstance(event, StreamToken):
                    yield _sse("token", {"content": event.content})
                elif isinstance(event, StreamDone):
                    yield _sse("done", {"conversation_id": event.conversation_id})
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
        messages=[
            MessageResponse(
                role=m.role.value,
                content=m.content,
                created_at=m.created_at,
            )
            for m in conversation.messages
        ],
        created_at=conversation.created_at,
    )


@router.post("/documents", response_model=IngestionResponse, tags=["documents"])
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
) -> IngestionResponse:
    service = _ingestion_service(request)
    data = await file.read()
    filename = file.filename or "upload.bin"
    validate_upload(filename=filename, content_type=file.content_type, data=data)
    result = await service.ingest(filename=filename, data=data)
    return IngestionResponse(
        document_id=result.document_id,
        filename=result.filename,
        format=result.format.value,
        chunk_count=result.chunk_count,
    )


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
async def list_documents(request: Request) -> DocumentListResponse:
    service = _ingestion_service(request)
    documents = await service.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentSummaryResponse(
                id=doc.id,
                filename=doc.filename,
                format=doc.format.value,
                chunk_count=doc.chunk_count,
                created_at=doc.created_at,
            )
            for doc in documents
        ]
    )


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"])
async def delete_document(document_id: str, request: Request) -> None:
    service = _ingestion_service(request)
    await service.delete_document(document_id)


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
