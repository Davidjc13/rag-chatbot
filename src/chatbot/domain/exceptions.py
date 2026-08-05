"""Excepciones de dominio e infraestructura."""

from __future__ import annotations


class ChatbotError(Exception):
    """Error base de la aplicación."""

    def __init__(self, message: str, *, code: str = "chatbot_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ValidationError(ChatbotError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="validation_error")


class ConversationNotFoundError(ChatbotError):
    def __init__(self, conversation_id: str) -> None:
        super().__init__(
            f"Conversación no encontrada: {conversation_id}",
            code="conversation_not_found",
        )
        self.conversation_id = conversation_id


class LLMProviderError(ChatbotError):
    def __init__(self, message: str, *, provider: str = "unknown") -> None:
        super().__init__(message, code="llm_provider_error")
        self.provider = provider


class LLMUnavailableError(LLMProviderError):
    def __init__(self, message: str, *, provider: str = "unknown") -> None:
        super().__init__(message, provider=provider)
        self.code = "llm_unavailable"


class ConfigurationError(ChatbotError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="configuration_error")


class UnsupportedDocumentError(ChatbotError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            f"Tipo de documento no soportado: {filename}",
            code="unsupported_document",
        )
        self.filename = filename


class DocumentParseError(ChatbotError):
    def __init__(self, message: str, *, filename: str | None = None) -> None:
        super().__init__(message, code="document_parse_error")
        self.filename = filename


class DocumentNotFoundError(ChatbotError):
    def __init__(self, document_id: str) -> None:
        super().__init__(
            f"Documento no encontrado: {document_id}",
            code="document_not_found",
        )
        self.document_id = document_id


class GuardrailBlockedError(ChatbotError):
    def __init__(self, message: str, *, reason: str = "blocked") -> None:
        super().__init__(message, code="guardrail_blocked")
        self.reason = reason


class InvalidMimeTypeError(ChatbotError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_mime_type")


class TranscriptionUnavailableError(ChatbotError):
    def __init__(self, message: str, *, provider: str = "unknown") -> None:
        super().__init__(message, code="transcription_unavailable")
        self.provider = provider
