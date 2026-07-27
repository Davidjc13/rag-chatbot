"""Rutas HTTP del chatbot."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import StreamingResponse

from chatbot.application.services.chat_service import (
    ChatService,
    StreamDone,
    StreamMeta,
    StreamToken,
)
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
    HealthResponse,
    IngestionResponse,
    MessageResponse,
)

router = APIRouter()


def _chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def _ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


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
