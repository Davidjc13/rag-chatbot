"""Singleton lazy de variables de entorno."""

from __future__ import annotations

import os
import threading
from typing import Literal

from dotenv import load_dotenv


class Env:
    """
    Lee variables de entorno bajo demanda y las cachea tras el primer acceso.

    No materializa todo el entorno al construir la instancia.
    """

    _instance: Env | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._cache: dict[str, str | None] = {}
        self._dotenv_loaded = False
        self._dotenv_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> Env:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Resetea el singleton (tests)."""
        with cls._lock:
            cls._instance = None

    def _ensure_dotenv(self) -> None:
        if self._dotenv_loaded:
            return
        with self._dotenv_lock:
            if self._dotenv_loaded:
                return
            load_dotenv()
            self._dotenv_loaded = True

    def get(self, key: str, default: str | None = None) -> str | None:
        if key in self._cache:
            return self._cache[key]
        self._ensure_dotenv()
        value = os.environ.get(key)
        if value is None or value == "":
            resolved = default
        else:
            resolved = value
        self._cache[key] = resolved
        return resolved

    def require(self, key: str) -> str:
        value = self.get(key)
        if value is None:
            raise KeyError(f"Variable de entorno requerida no definida: {key}")
        return value

    def get_int(self, key: str, default: int) -> int:
        raw = self.get(key)
        if raw is None:
            return default
        return int(raw)

    def get_float(self, key: str, default: float) -> float:
        raw = self.get(key)
        if raw is None:
            return default
        return float(raw)

    def get_bool(self, key: str, default: bool = False) -> bool:
        raw = self.get(key)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    # --- Accesores tipados (lazy) ---

    @property
    def app_name(self) -> str:
        return self.get("APP_NAME", "RAG Chatbot") or "RAG Chatbot"

    @property
    def app_env(self) -> Literal["development", "staging", "production"]:
        value = (self.get("APP_ENV", "development") or "development").lower()
        if value not in {"development", "staging", "production"}:
            return "development"
        return value  # type: ignore[return-value]

    @property
    def log_level(self) -> str:
        return (self.get("LOG_LEVEL", "INFO") or "INFO").upper()

    @property
    def log_json(self) -> bool:
        return self.get_bool("LOG_JSON", False)

    @property
    def host(self) -> str:
        return self.get("HOST", "0.0.0.0") or "0.0.0.0"

    @property
    def port(self) -> int:
        return self.get_int("PORT", 8000)

    @property
    def llm_provider(self) -> Literal["litellm", "ollama", "mock"]:
        value = (self.get("LLM_PROVIDER", "litellm") or "litellm").lower()
        if value not in {"litellm", "ollama", "mock"}:
            return "litellm"
        return value  # type: ignore[return-value]

    @property
    def litellm_model(self) -> str:
        return self.get("LITELLM_MODEL", "ollama/qwen2.5:3b") or "ollama/qwen2.5:3b"

    @property
    def litellm_api_base(self) -> str | None:
        return self.get("LITELLM_API_BASE", "http://localhost:11434")

    @property
    def litellm_api_key(self) -> str | None:
        return self.get("LITELLM_API_KEY")

    @property
    def litellm_timeout_seconds(self) -> float:
        return self.get_float("LITELLM_TIMEOUT_SECONDS", 120.0)

    @property
    def litellm_temperature(self) -> float:
        return self.get_float("LITELLM_TEMPERATURE", 0.7)

    @property
    def ollama_base_url(self) -> str:
        return self.get("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"

    @property
    def ollama_model(self) -> str:
        return self.get("OLLAMA_MODEL", "qwen2.5:3b") or "qwen2.5:3b"

    @property
    def ollama_timeout_seconds(self) -> float:
        return self.get_float("OLLAMA_TIMEOUT_SECONDS", 120.0)

    @property
    def ollama_temperature(self) -> float:
        return self.get_float("OLLAMA_TEMPERATURE", 0.7)

    @property
    def chunk_size(self) -> int:
        return self.get_int("CHUNK_SIZE", 800)

    @property
    def chunk_overlap(self) -> int:
        return self.get_int("CHUNK_OVERLAP", 100)

    @property
    def rag_top_k(self) -> int:
        return self.get_int("RAG_TOP_K", 4)

    @property
    def rag_min_score(self) -> float:
        return self.get_float("RAG_MIN_SCORE", 0.25)

    @property
    def litellm_embedding_model(self) -> str:
        return (
            self.get("LITELLM_EMBEDDING_MODEL", "ollama/nomic-embed-text")
            or "ollama/nomic-embed-text"
        )

    @property
    def litellm_embedding_timeout_seconds(self) -> float:
        return self.get_float("LITELLM_EMBEDDING_TIMEOUT_SECONDS", 60.0)

    @property
    def database_url(self) -> str:
        return (
            self.get(
                "DATABASE_URL",
                "postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot",
            )
            or "postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot"
        )

    @property
    def embedding_dimension(self) -> int:
        return self.get_int("EMBEDDING_DIMENSION", 768)

    @property
    def http_timeout_seconds(self) -> float:
        return self.get_float("HTTP_TIMEOUT_SECONDS", 120.0)

    @property
    def active_model(self) -> str:
        if self.llm_provider == "litellm":
            return self.litellm_model
        if self.llm_provider == "ollama":
            return self.ollama_model
        return self.litellm_model or self.ollama_model
