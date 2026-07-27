"""Factory de parsers de documentos."""

from __future__ import annotations

from chatbot.domain.exceptions import UnsupportedDocumentError
from chatbot.domain.ports import DocumentParserPort
from chatbot.infrastructure.adapters.ingestion.docx_parser import DocxParserAdapter
from chatbot.infrastructure.adapters.ingestion.pdf_parser import PdfParserAdapter
from chatbot.infrastructure.adapters.ingestion.xlsx_parser import XlsxParserAdapter


class DocumentParserFactory:
    """Resuelve el parser adecuado según la extensión del fichero."""

    def __init__(self, parsers: list[DocumentParserPort] | None = None) -> None:
        self._parsers = parsers or [
            PdfParserAdapter(),
            DocxParserAdapter(),
            XlsxParserAdapter(),
        ]

    def get_parser(self, filename: str) -> DocumentParserPort:
        for parser in self._parsers:
            if parser.supports(filename):
                return parser
        raise UnsupportedDocumentError(filename)

    def supported_extensions(self) -> list[str]:
        return [".pdf", ".docx", ".xlsx", ".xlsm"]
