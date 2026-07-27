"""Repositorio en memoria de conversaciones."""

from __future__ import annotations

import asyncio

from chatbot.domain.entities import Conversation


class InMemoryConversationRepository:
    """Almacenamiento efímero thread-safe para demos y entornos locales."""

    def __init__(self) -> None:
        self._store: dict[str, Conversation] = {}
        self._lock = asyncio.Lock()

    async def get(self, conversation_id: str) -> Conversation | None:
        async with self._lock:
            return self._store.get(conversation_id)

    async def save(self, conversation: Conversation) -> None:
        async with self._lock:
            self._store[conversation.id] = conversation

    async def delete(self, conversation_id: str) -> None:
        async with self._lock:
            self._store.pop(conversation_id, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
