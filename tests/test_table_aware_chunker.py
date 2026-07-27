"""Tests del chunker consciente de tablas."""

from __future__ import annotations

from chatbot.application.services.table_aware_chunker import TableAwareChunker
from chatbot.application.services.table_markdown import table_to_markdown
from chatbot.domain.documents import ContentBlock, ContentKind, DocumentFormat, ParsedDocument


def _doc(*blocks: ContentBlock) -> ParsedDocument:
    return ParsedDocument(filename="sample.docx", format=DocumentFormat.DOCX, blocks=list(blocks))


def test_table_to_markdown_includes_headers() -> None:
    md = table_to_markdown(
        ["Nombre", "Edad"],
        [["Ana", 30], ["Luis", 25]],
        title="Personas",
    )
    assert "**Personas**" in md
    assert "| Nombre | Edad |" in md
    assert "| Ana | 30 |" in md


def test_chunker_does_not_split_table() -> None:
    table = table_to_markdown(
        ["Producto", "Precio"],
        [["A", "10"], ["B", "20"], ["C", "30"]],
        title="Precios",
    )
    long_prefix = ("Introducción al catálogo. " * 40).strip()
    long_suffix = ("Notas finales del documento. " * 40).strip()
    document = _doc(
        ContentBlock(kind=ContentKind.TEXT, text=long_prefix),
        ContentBlock(kind=ContentKind.TABLE, text=table),
        ContentBlock(kind=ContentKind.TEXT, text=long_suffix),
    )

    chunker = TableAwareChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk(document)
    assert chunks

    table_chunks = [c for c in chunks if "| Producto | Precio |" in c.content]
    assert table_chunks
    for chunk in table_chunks:
        # Tabla completa (cabecera + filas) sin cortar a medias.
        assert "| Producto | Precio |" in chunk.content
        assert "| --- | --- |" in chunk.content
        assert "| A | 10 |" in chunk.content
        assert "| B | 20 |" in chunk.content
        assert "| C | 30 |" in chunk.content
        assert "t0" not in chunk.content


def test_chunker_restores_multiple_tables() -> None:
    t1 = table_to_markdown(["X"], [["1"]], title="T1")
    t2 = table_to_markdown(["Y"], [["2"]], title="T2")
    document = _doc(
        ContentBlock(kind=ContentKind.TEXT, text="Antes"),
        ContentBlock(kind=ContentKind.TABLE, text=t1),
        ContentBlock(kind=ContentKind.TEXT, text="Medio"),
        ContentBlock(kind=ContentKind.TABLE, text=t2),
        ContentBlock(kind=ContentKind.TEXT, text="Después"),
    )
    chunks = TableAwareChunker(chunk_size=500, chunk_overlap=50).chunk(document)
    joined = "\n".join(c.content for c in chunks)
    assert "**T1**" in joined
    assert "**T2**" in joined
    assert "| X |" in joined
    assert "| Y |" in joined
    assert "t0" not in joined
    assert "t1" not in joined
