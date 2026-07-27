"""Parser DOCX con párrafos y tablas."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from chatbot.application.services.table_markdown import table_to_markdown
from chatbot.domain.documents import ContentBlock, ContentKind, DocumentFormat, ParsedDocument
from chatbot.domain.exceptions import DocumentParseError
from chatbot.domain.ports import DocumentParserPort

logger = logging.getLogger(__name__)


class DocxParserAdapter(DocumentParserPort):
    def supports(self, filename: str) -> bool:
        return Path(filename).suffix.lower() == ".docx"

    def parse(self, *, filename: str, data: bytes) -> ParsedDocument:
        blocks: list[ContentBlock] = []
        try:
            document = Document(io.BytesIO(data))
            table_counter = 0
            for element in document.element.body:
                if element.tag == qn("w:p"):
                    paragraph = Paragraph(element, document)
                    text = paragraph.text.strip()
                    if text:
                        blocks.append(
                            ContentBlock(kind=ContentKind.TEXT, text=text),
                        )
                elif element.tag == qn("w:tbl"):
                    table = Table(element, document)
                    table_counter += 1
                    markdown = self._table_to_block(table, table_counter)
                    if markdown:
                        blocks.append(
                            ContentBlock(
                                kind=ContentKind.TABLE,
                                text=markdown,
                                metadata={"table_index": table_counter},
                            )
                        )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error parseando DOCX %s", filename)
            raise DocumentParseError(
                f"No se pudo parsear el DOCX: {exc}",
                filename=filename,
            ) from exc

        if not blocks:
            raise DocumentParseError("El DOCX no contiene texto ni tablas", filename=filename)

        return ParsedDocument(filename=filename, format=DocumentFormat.DOCX, blocks=blocks)

    @staticmethod
    def _table_to_block(table: Table, index: int) -> str | None:
        matrix: list[list[str]] = []
        for row in table.rows:
            matrix.append([cell.text.strip() for cell in row.cells])
        if not matrix:
            return None
        headers = matrix[0]
        rows = matrix[1:]
        return table_to_markdown(headers, rows, title=f"Tabla {index}")
