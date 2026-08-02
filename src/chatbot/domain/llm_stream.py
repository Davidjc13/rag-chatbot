"""Deltas de streaming del LLM (contenido vs thinking)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LLMDeltaKind = Literal["content", "thinking"]


@dataclass(frozen=True, slots=True)
class LLMDelta:
    text: str
    kind: LLMDeltaKind = "content"
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int | None = None
