"""Helper de prompts en memoria para tests."""

from __future__ import annotations

from chatbot.domain.prompts import (
    PROMPT_AGENT_SYSTEM,
    PROMPT_STUDY_EVALUATE,
    PROMPT_STUDY_QUIZ_GENERATE,
    PROMPT_STUDY_SUMMARY,
    PROMPT_STUDY_TUTOR_SYSTEM,
    PROMPT_SYSTEM,
    PROMPT_USER_MESSAGE,
)
from chatbot.infrastructure.adapters.persistence.memory_prompt_repository import (
    InMemoryPromptRepository,
)


def default_prompt_repo(
    *,
    system: str = "Eres un bot.\n\n{context}",
    user_message: str = "{question}",
    study_summary: str = 'Responde JSON: {"summary": "Resumen test", "key_concepts": ["A", "B"]}\n{content}',
    study_quiz_generate: str = (
        'Genera preguntas.\nResponde JSON: [{"question": "Q1", "reference_answer": "A1", "rubric": "R1"}]'
    ),
    study_evaluate: str = (
        'Evalúa.\nResponde JSON: {"score": 8, "max_score": 10, "feedback": "Bien", '
        '"missing_points": [], "is_correct": true}'
    ),
    study_tutor_system: str = "Tutor.\n{context}",
    agent_system: str = "Usa search_documents antes de responder.",
) -> InMemoryPromptRepository:
    return InMemoryPromptRepository(
        {
            PROMPT_SYSTEM: system,
            PROMPT_USER_MESSAGE: user_message,
            PROMPT_AGENT_SYSTEM: agent_system,
            PROMPT_STUDY_SUMMARY: study_summary,
            PROMPT_STUDY_QUIZ_GENERATE: study_quiz_generate,
            PROMPT_STUDY_EVALUATE: study_evaluate,
            PROMPT_STUDY_TUTOR_SYSTEM: study_tutor_system,
        }
    )
