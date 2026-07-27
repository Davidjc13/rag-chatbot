"""Entidades de dominio del chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("El contenido del mensaje no puede estar vacío")


@dataclass(slots=True)
class Conversation:
    id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def history(self) -> list[Message]:
        return list(self.messages)


@dataclass(frozen=True, slots=True)
class ChatReply:
    conversation_id: str
    message: Message
    model: str
