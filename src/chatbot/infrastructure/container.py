"""Contenedor de dependencias (Singleton + Composition Root)."""

from __future__ import annotations

import logging
import threading

from neo4j import AsyncDriver, AsyncGraphDatabase
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from chatbot.application.services.chat_service import ChatService
from chatbot.application.services.eval_service import EvalService
from chatbot.application.services.guardrails import RuleBasedGuardrail
from chatbot.application.services.ingestion_service import IngestionService
from chatbot.application.services.table_aware_chunker import TableAwareChunker
from chatbot.core.env import Env
from chatbot.core.http import AsyncHttpClient
from chatbot.domain.ports import (
    ConversationRepositoryPort,
    EmbeddingPort,
    GuardrailPort,
    LLMPort,
    PromptRepositoryPort,
    VectorStorePort,
)
from chatbot.domain.retrieval import RETRIEVAL_BACKEND_POSTGRES
from chatbot.infrastructure.adapters.ingestion.parser_factory import DocumentParserFactory
from chatbot.infrastructure.adapters.llm.embedding_adapter import LiteLLMEmbeddingAdapter
from chatbot.infrastructure.adapters.llm.llm_factory import LLMFactory
from chatbot.infrastructure.adapters.persistence.neo4j_vector_store import (
    Neo4jVectorStore,
)
from chatbot.infrastructure.adapters.persistence.postgres.conversation_repository import (
    PostgresConversationRepository,
)
from chatbot.infrastructure.adapters.persistence.postgres.engine import (
    create_engine,
    create_session_factory,
)
from chatbot.infrastructure.adapters.persistence.postgres.eval_repository import (
    PostgresEvalRepository,
)
from chatbot.infrastructure.adapters.persistence.postgres.prompt_repository import (
    PostgresPromptRepository,
)
from chatbot.infrastructure.adapters.persistence.postgres.schema import (
    init_schema,
    seed_prompts,
)
from chatbot.infrastructure.adapters.persistence.postgres.vector_store import (
    PostgresVectorStore,
)
from chatbot.infrastructure.adapters.persistence.routed_vector_store import (
    RoutedVectorStore,
)

logger = logging.getLogger(__name__)


class AppContainer:  # pylint: disable=too-many-instance-attributes
    """
    Contenedor de inyección de dependencias con patrón Singleton.

    Ensamblla puertos y adaptadores en un único punto (composition root).
    """

    _instance: AppContainer | None = None
    _lock = threading.Lock()

    def __init__(self, env: Env | None = None) -> None:
        self.env = env or Env.get_instance()
        self.http = AsyncHttpClient.get_instance(
            timeout_seconds=self.env.http_timeout_seconds,
        )
        self.engine: AsyncEngine = create_engine(self.env.database_url)
        self.session_factory: async_sessionmaker[AsyncSession] = create_session_factory(
            self.engine
        )
        self.llm: LLMPort = LLMFactory.create(self.env)
        self.repository: ConversationRepositoryPort = PostgresConversationRepository(
            self.session_factory
        )
        self.prompts: PromptRepositoryPort = PostgresPromptRepository(self.session_factory)
        self.eval_repository = PostgresEvalRepository(self.session_factory)
        self.eval_service = EvalService(repository=self.eval_repository, env=self.env)
        embedding_api_base = self._resolve_embedding_api_base(self.env)
        self.embeddings: EmbeddingPort = LiteLLMEmbeddingAdapter(
            model=self.env.litellm_embedding_model,
            api_base=embedding_api_base,
            api_key=self.env.litellm_api_key,
            timeout_seconds=self.env.litellm_embedding_timeout_seconds,
        )
        self.postgres_vector_store: VectorStorePort = PostgresVectorStore(self.session_factory)
        self.neo4j_driver: AsyncDriver | None = None
        self.neo4j_vector_store: Neo4jVectorStore | None = None
        if self.env.neo4j_enabled:
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.env.neo4j_uri,
                auth=(self.env.neo4j_username, self.env.neo4j_password),
            )
            self.neo4j_vector_store = Neo4jVectorStore(
                self.neo4j_driver,
                database=self.env.neo4j_database,
                vector_index_name=self.env.neo4j_vector_index,
                embedding_dimension=self.env.neo4j_embedding_dimension,
            )
        self.vector_store: VectorStorePort = RoutedVectorStore(
            primary=self.postgres_vector_store,
            neo4j=self.neo4j_vector_store,
            default_backend=self.env.vector_backend or RETRIEVAL_BACKEND_POSTGRES,
        )
        self.parser_factory = DocumentParserFactory()
        self.chunker = TableAwareChunker(
            chunk_size=self.env.chunk_size,
            chunk_overlap=self.env.chunk_overlap,
        )
        self.ingestion_service = IngestionService(
            parser_factory=self.parser_factory,
            chunker=self.chunker,
            embeddings=self.embeddings,
            vector_store=self.vector_store,
        )
        self.guardrails: GuardrailPort = RuleBasedGuardrail(
            min_score=self.env.rag_min_score,
        )
        self.chat_service = ChatService(
            llm=self.llm,
            repository=self.repository,
            prompts=self.prompts,
            embeddings=self.embeddings,
            vector_store=self.vector_store,
            guardrails=self.guardrails,
            rag_top_k=self.env.rag_top_k,
        )
        # Compat: rutas health leen settings.llm_provider
        self.settings = self.env
        logger.info(
            "AppContainer inicializado",
            extra={
                "provider": self.env.llm_provider,
                "model": self.env.active_model,
                "embedding_model": self.env.litellm_embedding_model,
                "embedding_api_base": embedding_api_base,
                "vector_backend": self.env.vector_backend,
                "neo4j_enabled": self.env.neo4j_enabled,
            },
        )

    async def startup(self) -> None:
        await init_schema(
            self.engine,
            embedding_dimension=self.env.embedding_dimension,
        )
        await seed_prompts(self.session_factory)
        if self.neo4j_vector_store is not None:
            await self.neo4j_vector_store.initialize()

    async def shutdown(self) -> None:
        await self.http.aclose()
        if self.neo4j_vector_store is not None:
            await self.neo4j_vector_store.close()
        await self.engine.dispose()

    @staticmethod
    def _resolve_embedding_api_base(env: Env) -> str | None:
        """
        Alinea el endpoint de embeddings con el de chat.

        Con LLM_PROVIDER=ollama usamos OLLAMA_BASE_URL (p.ej. localhost en host,
        http://ollama:11434 en Compose). Evita que un LITELLM_API_BASE de Docker
        apunte a un hostname irresoluble fuera de la red de Compose.
        """
        if env.llm_provider == "ollama":
            return env.ollama_base_url
        return env.litellm_api_base

    @classmethod
    def get_instance(cls, env: Env | None = None) -> AppContainer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(env=env)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resetea el singleton (tests / recargas)."""
        with cls._lock:
            cls._instance = None
