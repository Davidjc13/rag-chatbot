"""Parser de texto plano, Markdown y CSV."""

from __future__ import annotations

from pathlib import Path

from chatbot.domain.documents import ContentBlock, ContentKind, DocumentFormat, ParsedDocument
from chatbot.domain.exceptions import DocumentParseError
from chatbot.domain.ports import DocumentParserPort

_EXTENSIONS = {".txt": DocumentFormat.TXT, ".md": DocumentFormat.MD, ".csv": DocumentFormat.CSV}


class TextParserAdapter(DocumentParserPort):
    def supports(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in _EXTENSIONS

    def parse(self, *, filename: str, data: bytes) -> ParsedDocument:
        suffix = Path(filename).suffix.lower()
        fmt = _EXTENSIONS.get(suffix, DocumentFormat.TXT)
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentParseError(
                "El fichero de texto no está en UTF-8",
                filename=filename,
            ) from exc
        cleaned = text.replace("\x00", "").strip()
        if not cleaned:
            raise DocumentParseError("El fichero de texto está vacío", filename=filename)

        if fmt == DocumentFormat.CSV:
            block = ContentBlock(
                kind=ContentKind.TABLE,
                text=_csv_to_markdown(cleaned),
                metadata={"filename": filename},
            )
        else:
            block = ContentBlock(
                kind=ContentKind.TEXT,
                text=cleaned,
                metadata={"filename": filename},
            )
        return ParsedDocument(filename=filename, format=fmt, blocks=[block])


def _csv_to_markdown(raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return raw
    rows = [line.split(",") for line in lines]
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:] or [[""] * width]
    header_line = "| " + " | ".join(cell.strip() for cell in header) + " |"
    sep = "| " + " | ".join("---" for _ in header) + " |"
    body_lines = ["| " + " | ".join(cell.strip() for cell in row) + " |" for row in body]
    return "\n".join([header_line, sep, *body_lines])
