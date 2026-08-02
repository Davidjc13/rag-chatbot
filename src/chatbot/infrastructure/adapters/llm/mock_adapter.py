"""Adaptador mock para desarrollo y tests sin Ollama."""

from __future__ import annotations

from collections.abc import AsyncIterator

from chatbot.domain.entities import Message, Role
from chatbot.domain.llm_stream import LLMDelta
from chatbot.domain.ports import LLMPort


class MockLLMAdapter(LLMPort):
    """Responde de forma determinista sin llamar a un modelo real."""

    def __init__(self, *, model: str = "mock-model") -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def _build_reply(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None,
    ) -> str:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == Role.USER),
            "",
        )
        reply = f"[mock:{self._model}] Recibí: {last_user}"
        if system_prompt:
            reply = f"{reply} (system ok)"
        return reply

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> Message:
        reply = self._build_reply(messages, system_prompt=system_prompt)
        return Message(role=Role.ASSISTANT, content=reply)

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        reply = self._build_reply(messages, system_prompt=system_prompt)
        words = reply.split(" ")
        for index, word in enumerate(words):
            suffix = " " if index < len(words) - 1 else ""
            yield LLMDelta(text=f"{word}{suffix}", kind="content")
        yield LLMDelta(
            text="",
            kind="content",
            input_tokens=42,
            output_tokens=len(reply.split()),
            duration_ms=10,
        )

    async def health_check(self) -> bool:
        return True
