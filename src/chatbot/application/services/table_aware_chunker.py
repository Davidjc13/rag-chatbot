"""Chunking consciente de tablas mediante placeholders t{i}."""

from __future__ import annotations

import re

from chatbot.application.services.table_markdown import split_markdown_table_by_rows
from chatbot.domain.documents import ContentBlock, ContentKind, DocumentChunk, ParsedDocument

_TOKEN_PATTERN = re.compile(r"\bt\d+\b")
_TOKEN_SPLIT = re.compile(r"(t\d+)")


class TableAwareChunker:
    """
    Protege tablas en el chunking:

    1. Sustituye cada tabla por un token ``t{i}``.
    2. Hace split del texto (tamaño / overlap).
    3. Restaura cada ``t{i}`` por la tabla Markdown completa.
    """

    def __init__(self, *, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        if chunk_size < 64:
            raise ValueError("chunk_size debe ser >= 64")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe estar en [0, chunk_size)")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        protected_text, tables = self._protect_tables(document.blocks)
        raw_chunks = self._split_preserving_tokens(protected_text)

        restored: list[str] = []
        for raw in raw_chunks:
            content = self._restore_tables(raw, tables).strip()
            if not content:
                continue
            if len(content) <= self._chunk_size * 2:
                restored.append(content)
            else:
                restored.extend(self._expand_oversized(content))

        chunks: list[DocumentChunk] = []
        for index, content in enumerate(restored):
            if not content.strip():
                continue
            chunks.append(
                DocumentChunk(
                    document_id=document.id,
                    content=content.strip(),
                    metadata={
                        "filename": document.filename,
                        "format": document.format.value,
                        "chunk_index": index,
                        "has_table": self._looks_like_table(content),
                    },
                )
            )
        return chunks

    @staticmethod
    def _looks_like_table(content: str) -> bool:
        return "|" in content and "---" in content

    def _protect_tables(
        self,
        blocks: list[ContentBlock],
    ) -> tuple[str, dict[str, str]]:
        parts: list[str] = []
        tables: dict[str, str] = {}
        table_index = 0

        for block in blocks:
            if block.kind == ContentKind.TEXT:
                text = block.text.strip()
                if text:
                    parts.append(text)
                continue

            token = f"t{table_index}"
            tables[token] = block.text.strip()
            parts.append(token)
            table_index += 1

        return "\n\n".join(parts), tables

    def _split_preserving_tokens(self, text: str) -> list[str]:
        if not text.strip():
            return []

        parts = [part for part in _TOKEN_SPLIT.split(text) if part != ""]
        chunks: list[str] = []
        current = ""

        def flush(*, keep_overlap: bool) -> None:
            nonlocal current
            if current.strip():
                chunks.append(current.strip())
            if keep_overlap and self._chunk_overlap > 0 and current:
                overlap_source = _TOKEN_PATTERN.sub("", current).strip()
                current = overlap_source[-self._chunk_overlap :] if overlap_source else ""
            else:
                current = ""

        for part in parts:
            if _TOKEN_PATTERN.fullmatch(part):
                # Token atómico: nunca se parte.
                if current and len(current) + len(part) > self._chunk_size:
                    flush(keep_overlap=False)
                current = f"{current}{part}" if current else part
                if len(current) >= self._chunk_size:
                    flush(keep_overlap=False)
                continue

            remaining = part
            while remaining:
                space_left = self._chunk_size - len(current)
                if space_left <= 0:
                    flush(keep_overlap=True)
                    space_left = self._chunk_size - len(current)
                    if space_left <= 0:
                        flush(keep_overlap=False)
                        space_left = self._chunk_size

                if len(remaining) <= space_left:
                    current = f"{current}{remaining}" if current else remaining
                    remaining = ""
                    break

                piece = remaining[:space_left]
                cut = piece.rfind(" ")
                if cut > max(20, space_left // 4):
                    piece = piece[:cut]
                if not piece:
                    piece = remaining[:space_left]

                current = f"{current}{piece}" if current else piece
                remaining = remaining[len(piece) :].lstrip()
                flush(keep_overlap=True)

        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _restore_tables(self, chunk: str, tables: dict[str, str]) -> str:
        def replacer(match: re.Match[str]) -> str:
            token = match.group(0)
            return tables.get(token, token)

        return _TOKEN_PATTERN.sub(replacer, chunk)

    def _expand_oversized(self, content: str) -> list[str]:
        if self._looks_like_table(content):
            return split_markdown_table_by_rows(content, max_chars=self._chunk_size)

        pieces: list[str] = []
        start = 0
        while start < len(content):
            end = min(start + self._chunk_size, len(content))
            piece = content[start:end]
            if end < len(content):
                cut = piece.rfind(" ")
                if cut > self._chunk_size // 4:
                    piece = piece[:cut]
                    end = start + cut
            stripped = piece.strip()
            if stripped:
                pieces.append(stripped)
            if end >= len(content):
                break
            start = max(end - self._chunk_overlap, start + 1)
        return pieces
