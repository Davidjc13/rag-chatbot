"""Manejadores globales de excepciones."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from chatbot.domain.exceptions import (
    ChatbotError,
    ConversationNotFoundError,
    DocumentNotFoundError,
    GuardrailBlockedError,
    InvalidMimeTypeError,
    LLMUnavailableError,
    UnsupportedDocumentError,
    ValidationError,
)

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ConversationNotFoundError)
    async def conversation_not_found_handler(
        _request: Request,
        exc: ConversationNotFoundError,
    ) -> JSONResponse:
        logger.warning("Conversación no encontrada: %s", exc.conversation_id)
        return JSONResponse(
            status_code=404,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(
        _request: Request,
        exc: DocumentNotFoundError,
    ) -> JSONResponse:
        logger.warning("Documento no encontrado: %s", exc.document_id)
        return JSONResponse(
            status_code=404,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(UnsupportedDocumentError)
    async def unsupported_document_handler(
        _request: Request,
        exc: UnsupportedDocumentError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=415,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(InvalidMimeTypeError)
    async def invalid_mime_handler(
        _request: Request,
        exc: InvalidMimeTypeError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=415,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(GuardrailBlockedError)
    async def guardrail_blocked_handler(
        _request: Request,
        exc: GuardrailBlockedError,
    ) -> JSONResponse:
        logger.warning("Guardrail bloqueó mensaje: %s", exc.reason)
        return JSONResponse(
            status_code=403,
            content={"error": exc.message, "code": exc.code, "detail": exc.reason},
        )

    @app.exception_handler(ValidationError)
    async def validation_handler(_request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable_handler(
        _request: Request,
        exc: LLMUnavailableError,
    ) -> JSONResponse:
        logger.error("LLM no disponible: %s", exc.message)
        return JSONResponse(
            status_code=503,
            content={"error": exc.message, "code": exc.code, "detail": f"provider={exc.provider}"},
        )

    @app.exception_handler(ChatbotError)
    async def chatbot_error_handler(_request: Request, exc: ChatbotError) -> JSONResponse:
        logger.error("Error de aplicación: %s [%s]", exc.message, exc.code)
        return JSONResponse(
            status_code=400,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Datos de entrada inválidos",
                "code": "request_validation_error",
                "detail": str(exc.errors()),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
        logger.exception("Error no controlado en %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error interno del servidor",
                "code": "internal_error",
            },
        )
