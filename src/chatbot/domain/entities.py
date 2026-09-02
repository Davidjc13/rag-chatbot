"""Entidades de dominio del chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

_DEFAULT_TITLE = "Nueva conversación"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


def conversation_title_from_query(text: str, max_len: int = 64) -> str:
    value = (text or "").strip().replace("\n", " ")
    while "  " in value:
        value = value.replace("  ", " ")
    if not value:
        return _DEFAULT_TITLE
    if len(value) <= max_len:
        return value
    return f"{value[: max_len - 1]}…"


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
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    title: str = _DEFAULT_TITLE

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(UTC)
        if message.role == Role.USER and (
            not self.title or self.title == _DEFAULT_TITLE
        ):
            self.title = conversation_title_from_query(message.content)

    def history(self) -> list[Message]:
        return list(self.messages)

    def history_window(self, max_messages: int) -> list[Message]:
        if max_messages <= 0 or len(self.messages) <= max_messages:
            return self.history()
        return list(self.messages[-max_messages:])


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    id: str
    title: str
    updated_at: datetime
    created_at: datetime
    preview: str = ""


@dataclass(frozen=True, slots=True)
class ChatReply:
    conversation_id: str
    message: Message
    model: str
