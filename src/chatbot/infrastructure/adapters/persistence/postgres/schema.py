"""Inicialización de esquema y seed de prompts."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from chatbot.domain.prompts import (
    PROMPT_AGENT_SYSTEM,
    PROMPT_STUDY_EVALUATE,
    PROMPT_STUDY_QUIZ_GENERATE,
    PROMPT_STUDY_SUMMARY,
    PROMPT_STUDY_TUTOR_SYSTEM,
    PROMPT_SYSTEM,
    PROMPT_USER_MESSAGE,
)
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

DEFAULT_AGENT_SYSTEM_MD = """\
Eres un asistente de documentos con acceso a la herramienta search_documents.

Reglas:
- Antes de afirmar hechos sobre los documentos, llama a search_documents.
- Si la primera búsqueda no basta o indica baja relevancia, reformula la \
consulta y busca otra vez.
- Cita las fuentes con exactamente el formato indicado en los resultados \
(un documento, una sola cita). No inventes índices, títulos ni ids.
- Si tras buscar no hay contexto suficiente, dilo con claridad y no inventes.
- Responde en el mismo idioma en que te escriben, de forma clara y concisa.
- Si razonas internamente, hazlo de forma breve.
"""

DEFAULT_STUDY_SUMMARY_MD = """\
Eres un asistente académico. Resume el siguiente material de apuntes de forma \
estructurada en español.

Incluye:
1. Título o tema principal
2. Puntos clave (lista)
3. Definiciones importantes
4. Relaciones o procesos relevantes

Responde SOLO con JSON válido (sin markdown) con esta forma:
{"summary": "texto markdown del resumen", "key_concepts": ["concepto1", "concepto2"]}

Material:
{content}
"""

DEFAULT_STUDY_TUTOR_SYSTEM_MD = """\
Eres un tutor académico estricto pero constructivo. Ayudas al estudiante a \
repasar apuntes usando SOLO el contexto proporcionado.

Reglas:
- Explica conceptos con claridad y precisión terminológica del material.
- Si el estudiante pide un test oral, formula UNA pregunta a la vez y evalúa \
su respuesta de forma objetiva (correcto/incorrecto, qué falta, puntuación 0-10).
- No inventes información fuera del contexto.
- Responde en el mismo idioma del estudiante.

Contexto de los apuntes:
{context}
"""

DEFAULT_STUDY_QUIZ_GENERATE_MD = """\
Genera {question_count} preguntas de estudio basadas en el material. \
Modo: {mode}.

El material incluye resumen y fragmentos originales. Las preguntas deben ser \
respondibles con el material (no triviales ni ambiguas).

Responde SOLO con JSON válido (array):
[
  {
    "question": "pregunta",
    "reference_answer": "respuesta esperada completa",
    "rubric": "criterios de evaluación estrictos"
  }
]

Resumen:
{summary}

Fragmentos:
{content}
"""

DEFAULT_STUDY_EVALUATE_MD = """\
Evalúa la respuesta del estudiante de forma OBJETIVA y ESTRICTA.

Pregunta: {question}
Respuesta esperada: {reference_answer}
Rúbrica: {rubric}
Respuesta del estudiante: {user_answer}

Material de referencia:
{content}

Criterios:
- Puntuación de 0 a 10 (max_score: 10)
- Penaliza respuestas vagas, incompletas o incorrectas
- Exige precisión terminológica del material
- is_correct es true solo si score >= 7

Responde SOLO con JSON válido:
{"score": 0, "max_score": 10, "feedback": "...", "missing_points": ["..."], "is_correct": false}
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
                ALTER TABLE eval_runs
                ADD COLUMN IF NOT EXISTS deepeval_metrics JSONB
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE eval_runs
                ADD COLUMN IF NOT EXISTS experiment_id VARCHAR(36)
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE eval_runs
                ADD COLUMN IF NOT EXISTS variant_label VARCHAR(8)
                """
            )
        )
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS eval_runs_experiment_id_idx
                ON eval_runs (experiment_id)
                """
            )
        )
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                ON chunks
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        )
        await conn.execute(
            text(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS purpose VARCHAR(32) NOT NULL DEFAULT 'general'
                """
            )
        )
    logger.info("Esquema PostgreSQL inicializado")


async def seed_prompts(session_factory: async_sessionmaker[AsyncSession]) -> None:
    defaults = {
        PROMPT_SYSTEM: DEFAULT_SYSTEM_PROMPT_MD,
        PROMPT_USER_MESSAGE: DEFAULT_USER_MESSAGE_MD,
        PROMPT_AGENT_SYSTEM: DEFAULT_AGENT_SYSTEM_MD,
        PROMPT_STUDY_SUMMARY: DEFAULT_STUDY_SUMMARY_MD,
        PROMPT_STUDY_TUTOR_SYSTEM: DEFAULT_STUDY_TUTOR_SYSTEM_MD,
        PROMPT_STUDY_QUIZ_GENERATE: DEFAULT_STUDY_QUIZ_GENERATE_MD,
        PROMPT_STUDY_EVALUATE: DEFAULT_STUDY_EVALUATE_MD,
    }
    async with session_factory() as session:
        for key, content in defaults.items():
            existing = await session.get(PromptModel, key)
            if existing is None:
                session.add(PromptModel(key=key, content=content))
                logger.info("Prompt seed: %s", key)
        await session.commit()
