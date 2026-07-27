"""Inicialización de esquema y seed de prompts."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from chatbot.domain.prompts import PROMPT_SYSTEM, PROMPT_USER_MESSAGE
from chatbot.infrastructure.adapters.persistence.postgres.models import (
    DEFAULT_EMBEDDING_DIMENSION,
    Base,
    PromptModel,
)

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT_MD = """\
Eres un asistente de documentos. Responde solo con información presente en el \
contexto RAG proporcionado. Si la pregunta no se puede responder con ese \
contexto, indícalo con claridad y no inventes. Cita las fuentes usadas con el \
formato de cita indicado en el contexto (un documento, una sola cita). No \
respondas temas ajenos a los documentos ni uses lenguaje ofensivo. Responde en \
el mismo idioma en que te escriben, de forma clara y concisa. Si razonas antes \
de responder, mantén ese razonamiento breve (pocas frases).

Usa el siguiente contexto de documentos para responder. Si el contexto no es \
suficiente, dilo con claridad.

{context}
"""

DEFAULT_USER_MESSAGE_MD = """\
{question}
"""


async def init_schema(
    engine: AsyncEngine,
    *,
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
) -> None:
    if embedding_dimension != DEFAULT_EMBEDDING_DIMENSION:
        logger.warning(
            "EMBEDDING_DIMENSION=%s difiere del Vector(%s) del modelo ORM; "
            "asegúrate de que coincidan.",
            embedding_dimension,
            DEFAULT_EMBEDDING_DIMENSION,
        )

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                ON chunks
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        )
    logger.info("Esquema PostgreSQL inicializado")


async def seed_prompts(session_factory: async_sessionmaker[AsyncSession]) -> None:
    defaults = {
        PROMPT_SYSTEM: DEFAULT_SYSTEM_PROMPT_MD,
        PROMPT_USER_MESSAGE: DEFAULT_USER_MESSAGE_MD,
    }
    async with session_factory() as session:
        for key, content in defaults.items():
            existing = await session.get(PromptModel, key)
            if existing is None:
                session.add(PromptModel(key=key, content=content))
                logger.info("Prompt seed: %s", key)
        await session.commit()
