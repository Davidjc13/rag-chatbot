"""Contenedor de dependencias (Singleton + Composition Root)."""

from __future__ import annotations

import logging
import threading

from chatbot.application.services.chat_service import ChatService
from chatbot.application.services.guardrails import RuleBasedGuardrail
from chatbot.application.services.ingestion_service import IngestionService
from chatbot.application.services.table_aware_chunker import TableAwareChunker
from chatbot.domain.ports import (
    ConversationRepositoryPort,
    EmbeddingPort,
    GuardrailPort,
    LLMPort,
    VectorStorePort,
)
from chatbot.infrastructure.adapters.ingestion.parser_factory import DocumentParserFactory
from chatbot.infrastructure.adapters.llm.embedding_adapter import LiteLLMEmbeddingAdapter
from chatbot.infrastructure.adapters.llm.llm_factory import LLMFactory
from chatbot.infrastructure.adapters.persistence.memory_repository import (
    InMemoryConversationRepository,
)
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore
from chatbot.infrastructure.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class AppContainer:  # pylint: disable=too-many-instance-attributes
    """
    Contenedor de inyección de dependencias con patrón Singleton.

    Ensamblla puertos y adaptadores en un único punto (composition root).
    """

    _instance: AppContainer | None = None
    _lock = threading.Lock()

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm: LLMPort = LLMFactory.create(self.settings)
        self.repository: ConversationRepositoryPort = InMemoryConversationRepository()
        self.embeddings: EmbeddingPort = LiteLLMEmbeddingAdapter(
            model=self.settings.litellm_embedding_model,
            api_base=self.settings.litellm_api_base,
            api_key=self.settings.litellm_api_key,
            timeout_seconds=self.settings.litellm_embedding_timeout_seconds,
        )
        self.vector_store: VectorStorePort = InMemoryVectorStore()
        self.parser_factory = DocumentParserFactory()
        self.chunker = TableAwareChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.ingestion_service = IngestionService(
            parser_factory=self.parser_factory,
            chunker=self.chunker,
            embeddings=self.embeddings,
            vector_store=self.vector_store,
        )
        self.guardrails: GuardrailPort = RuleBasedGuardrail(
            min_score=self.settings.rag_min_score,
        )
        self.chat_service = ChatService(
            llm=self.llm,
            repository=self.repository,
            system_prompt=self.settings.system_prompt,
            embeddings=self.embeddings,
            vector_store=self.vector_store,
            guardrails=self.guardrails,
            rag_top_k=self.settings.rag_top_k,
        )
        logger.info(
            "AppContainer inicializado",
            extra={
                "provider": self.settings.llm_provider,
                "model": self.settings.active_model,
                "embedding_model": self.settings.litellm_embedding_model,
            },
        )

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> AppContainer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(settings=settings)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resetea el singleton (tests / recargas)."""
        with cls._lock:
            cls._instance = None
