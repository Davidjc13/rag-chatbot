"""Parser PDF con extracción explícita de tablas."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pdfplumber

from chatbot.application.services.table_markdown import table_to_markdown
from chatbot.domain.documents import ContentBlock, ContentKind, DocumentFormat, ParsedDocument
from chatbot.domain.exceptions import DocumentParseError
from chatbot.domain.ports import DocumentParserPort

logger = logging.getLogger(__name__)


class PdfParserAdapter(DocumentParserPort):
    def supports(self, filename: str) -> bool:
        return Path(filename).suffix.lower() == ".pdf"

    def parse(self, *, filename: str, data: bytes) -> ParsedDocument:
        blocks: list[ContentBlock] = []
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables() or []
                    text = (page.extract_text() or "").strip()
                    if text:
                        blocks.append(
                            ContentBlock(
                                kind=ContentKind.TEXT,
                                text=text,
                                metadata={"page": page_number},
                            )
                        )
                    for table in tables:
                        markdown = self._table_block(table, page_number)
                        if markdown:
                            blocks.append(
                                ContentBlock(
                                    kind=ContentKind.TABLE,
                                    text=markdown,
                                    metadata={"page": page_number},
                                )
                            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error parseando PDF %s", filename)
            raise DocumentParseError(
                f"No se pudo parsear el PDF: {exc}",
                filename=filename,
            ) from exc

        if not blocks:
            raise DocumentParseError("El PDF no contiene texto ni tablas", filename=filename)

        return ParsedDocument(filename=filename, format=DocumentFormat.PDF, blocks=blocks)

    @staticmethod
    def _table_block(table: list[list[str | None]], page_number: int) -> str | None:
        if not table:
            return None
        headers = [cell if cell is not None else "" for cell in table[0]]
        rows = table[1:] if len(table) > 1 else []
        if not any(str(h).strip() for h in headers) and not rows:
            return None
        return table_to_markdown(
            [str(h) if h is not None else "" for h in headers],
            rows,
            title=f"Tabla (página {page_number})",
        )
