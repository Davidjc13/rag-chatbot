"""Helper de prompts en memoria para tests."""

from __future__ import annotations

from chatbot.domain.prompts import PROMPT_SYSTEM, PROMPT_USER_MESSAGE
from chatbot.infrastructure.adapters.persistence.memory_prompt_repository import (
    InMemoryPromptRepository,
)


def default_prompt_repo(
    *,
    system: str = "Eres un bot.\n\n{context}",
    user_message: str = "{question}",
) -> InMemoryPromptRepository:
    return InMemoryPromptRepository(
        {
            PROMPT_SYSTEM: system,
            PROMPT_USER_MESSAGE: user_message,
        }
    )
