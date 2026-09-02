"""Repositorio PostgreSQL de conversaciones."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from chatbot.domain.entities import Conversation, ConversationSummary, Message, Role
from chatbot.infrastructure.adapters.persistence.postgres.models import (
    ConversationModel,
    MessageModel,
)


class PostgresConversationRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get(self, conversation_id: str) -> Conversation | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ConversationModel)
                .where(ConversationModel.id == conversation_id)
                .options(selectinload(ConversationModel.messages))
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return Conversation(
                id=row.id,
                created_at=row.created_at,
                updated_at=row.updated_at or row.created_at,
                title=row.title or "Nueva conversación",
                messages=[
                    Message(
                        role=Role(m.role),
                        content=m.content,
                        created_at=m.created_at,
                    )
                    for m in row.messages
                ],
            )

    async def save(self, conversation: Conversation) -> None:
        async with self._session_factory() as session:
            row = await session.get(ConversationModel, conversation.id)
            if row is None:
                session.add(
                    ConversationModel(
                        id=conversation.id,
                        title=conversation.title,
                        created_at=conversation.created_at,
                        updated_at=conversation.updated_at,
                    )
                )
            else:
                row.title = conversation.title
                row.updated_at = conversation.updated_at
                await session.execute(
                    delete(MessageModel).where(
                        MessageModel.conversation_id == conversation.id
                    )
                )
                await session.flush()

            for message in conversation.messages:
                session.add(
                    MessageModel(
                        conversation_id=conversation.id,
                        role=message.role.value,
                        content=message.content,
                        created_at=message.created_at,
                    )
                )
            await session.commit()

    async def delete(self, conversation_id: str) -> None:
        async with self._session_factory() as session:
            row = await session.get(ConversationModel, conversation_id)
            if row is not None:
                await session.delete(row)
                await session.commit()

    async def list(self, *, limit: int = 50) -> list[ConversationSummary]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ConversationModel)
                .options(selectinload(ConversationModel.messages))
                .order_by(ConversationModel.updated_at.desc())
                .limit(max(limit, 0))
            )
            rows = result.scalars().all()
            summaries: list[ConversationSummary] = []
            for row in rows:
                preview = ""
                for message in row.messages:
                    if message.role == Role.USER.value:
                        preview = message.content[:160]
                        break
                summaries.append(
                    ConversationSummary(
                        id=row.id,
                        title=row.title or "Nueva conversación",
                        updated_at=row.updated_at or row.created_at,
                        created_at=row.created_at,
                        preview=preview,
                    )
                )
            return summaries
