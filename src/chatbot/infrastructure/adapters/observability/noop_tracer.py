"""Tracer nulo cuando la observabilidad está desactivada."""

from __future__ import annotations

from chatbot.domain.ports import ChatGenerationTrace, TracingPort


class NoOpTracer(TracingPort):
    """No envía trazas; cumple el puerto sin efectos secundarios."""

    def record_chat_generation(self, trace: ChatGenerationTrace) -> None:
        return None
