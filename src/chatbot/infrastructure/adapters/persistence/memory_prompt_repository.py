"""Repositorio en memoria de prompts (tests / fallback)."""

from __future__ import annotations

import asyncio


class InMemoryPromptRepository:
    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._store: dict[str, str] = dict(initial or {})
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            return self._store.get(key)

    async def upsert(self, key: str, content: str) -> None:
        async with self._lock:
            self._store[key] = content
