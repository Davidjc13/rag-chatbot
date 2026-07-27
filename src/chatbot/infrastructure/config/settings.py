"""Configuración de la aplicación (Singleton)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Ajustes cargados desde variables de entorno / .env.

    Patrón Singleton: usar `get_settings()` para obtener la única instancia.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "RAG Chatbot"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Proveedor LLM configurable: litellm | ollama | mock
    llm_provider: Literal["litellm", "ollama", "mock"] = "litellm"

    # LiteLLM (OpenAI, Ollama, Anthropic, Azure, etc.)
    litellm_model: str = "ollama/qwen2.5:3b"
    litellm_api_base: str | None = "http://localhost:11434"
    litellm_api_key: str | None = None
    litellm_timeout_seconds: float = 120.0
    litellm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Ollama directo (legacy / alternativa)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout_seconds: float = 120.0
    ollama_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    system_prompt: str = (
        "Eres un asistente de documentos. Responde solo con información "
        "presente en el contexto RAG proporcionado. Si la pregunta no se "
        "puede responder con ese contexto, indícalo con claridad y no inventes. "
        "Cita las fuentes usadas con el formato de cita indicado en el contexto. "
        "No respondas temas ajenos a los documentos ni uses lenguaje ofensivo. "
        "Responde en el mismo idioma en que te escriben, de forma clara y concisa."
    )

    # RAG / ingestión
    chunk_size: int = Field(default=800, ge=64)
    chunk_overlap: int = Field(default=100, ge=0)
    rag_top_k: int = Field(default=4, ge=1, le=20)
    rag_min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    litellm_embedding_model: str = "ollama/nomic-embed-text"
    litellm_embedding_timeout_seconds: float = 60.0

    @property
    def active_model(self) -> str:
        if self.llm_provider == "litellm":
            return self.litellm_model
        if self.llm_provider == "ollama":
            return self.ollama_model
        return self.litellm_model or self.ollama_model

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("litellm_api_base", "litellm_api_key", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la instancia única de Settings (Singleton)."""
    return Settings()


def reset_settings_cache() -> None:
    """Útil en tests para forzar recarga de configuración."""
    get_settings.cache_clear()
