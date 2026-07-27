"""Repositorio PostgreSQL de prompts markdown."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatbot.infrastructure.adapters.persistence.postgres.models import PromptModel


class PostgresPromptRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get(self, key: str) -> str | None:
        async with self._session_factory() as session:
            row = await session.get(PromptModel, key)
            return None if row is None else row.content

    async def upsert(self, key: str, content: str) -> None:
        async with self._session_factory() as session:
            row = await session.get(PromptModel, key)
            if row is None:
                session.add(PromptModel(key=key, content=content))
            else:
                row.content = content
                row.updated_at = datetime.now(UTC)
            await session.commit()
