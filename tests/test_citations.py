"""Tests del parseo conceptual de citas (lógica espejo del front)."""

from __future__ import annotations

import re

_CITATION_RE = re.compile(
    r"<index\s*=\s*(\d+)\s*,\s*source\s*=\s*([^,>]+?)\s*,\s*title\s*=\s*([^,>]+?)\s*,\s*id\s*=\s*([^>]+?)>",
    re.IGNORECASE,
)


def test_citation_regex_extracts_fields() -> None:
    text = (
        "Según la política "
        "<index = 1, source=rag, title=policy.pdf, id=abc-123> "
        "el plazo es de 30 días."
    )
    match = _CITATION_RE.search(text)
    assert match is not None
    assert match.group(1) == "1"
    assert match.group(2).strip() == "rag"
    assert match.group(3).strip() == "policy.pdf"
    assert match.group(4).strip() == "abc-123"


def test_citation_display_replaces_with_index() -> None:
    text = "Dato <index = 2, source=rag, title=a.docx, id=x> citado."
    display = _CITATION_RE.sub(lambda m: m.group(1), text)
    assert display == "Dato 2 citado."
    assert "<index" not in display
