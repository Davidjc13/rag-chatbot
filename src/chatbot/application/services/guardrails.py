"""Guardarraíles deterministas: toxicidad y alcance RAG."""

from __future__ import annotations

import re

from chatbot.domain.exceptions import GuardrailBlockedError
from chatbot.domain.ports import GuardrailPort

_DEFAULT_OUT_OF_SCOPE = (
    "Solo puedo responder con la información de los documentos indexados. "
    "No he encontrado contexto suficiente para esta pregunta."
)

_DEFAULT_BLOCKED = (
    "No puedo procesar ese mensaje porque incumple las normas de uso "
    "(lenguaje ofensivo o contenido no permitido)."
)

# Patrones básicos ES/EN (lista no exhaustiva; defensa en profundidad junto al system prompt).
_TOXIC_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(kill\s+yourself|kys)\b",
        r"\b(nigger|nigga|faggot|retard)\b",
        r"\b(hijo\s+de\s+puta|hijaputa|hijoputa)\b",
        r"\b(puta|puto|cabron|cabrón|gilipollas|capullo)\b",
        r"\b(maric[oó]n|marica|retrasad[oa])\b",
        r"\b(go\s+die|rape\s+you)\b",
        r"\b(suic[ií]date|m[aá]tate)\b",
    )
)


class RuleBasedGuardrail(GuardrailPort):
    """Filtro por patrones + umbral mínimo de score de retrieval."""

    def __init__(
        self,
        *,
        min_score: float = 0.25,
        out_of_scope_message: str = _DEFAULT_OUT_OF_SCOPE,
        blocked_message: str = _DEFAULT_BLOCKED,
        toxic_patterns: tuple[re.Pattern[str], ...] | None = None,
    ) -> None:
        self._min_score = min_score
        self._out_of_scope_message = out_of_scope_message
        self._blocked_message = blocked_message
        self._toxic_patterns = toxic_patterns if toxic_patterns is not None else _TOXIC_PATTERNS

    @property
    def out_of_scope_message(self) -> str:
        return self._out_of_scope_message

    def check_input(self, text: str) -> None:
        self._assert_not_toxic(text, reason="toxic_input")

    def check_output(self, text: str) -> None:
        self._assert_not_toxic(text, reason="toxic_output")

    def is_in_scope(self, scores: list[float]) -> bool:
        if not scores:
            return False
        return max(scores) >= self._min_score

    def _assert_not_toxic(self, text: str, *, reason: str) -> None:
        normalized = (text or "").strip()
        if not normalized:
            return
        for pattern in self._toxic_patterns:
            if pattern.search(normalized):
                raise GuardrailBlockedError(self._blocked_message, reason=reason)
