"""Formato de contexto RAG y citas (compartido por chat clásico y agente)."""

from __future__ import annotations

from chatbot.domain.documents import RetrievedChunk

CITATION_INSTRUCTIONS = (
    "Cuando uses información de un documento, cítalo en línea una sola vez "
    "con exactamente este formato (sin alterar los campos ni añadir espacios "
    "extra alrededor del signo = salvo el de index): "
    "<index = N, source=rag, title=NOMBRE_ARCHIVO, id=DOCUMENT_ID>. "
    "Cada documento tiene un único índice: no repitas la misma cita ni cites "
    "el mismo id más de una vez. No inventes índices, títulos ni ids."
)


def document_index_map(
    retrieved: list[RetrievedChunk],
    *,
    existing: dict[str, tuple[int, str]] | None = None,
) -> dict[str, tuple[int, str]]:
    """Asigna un índice único por document_id (orden de primera aparición)."""
    mapping: dict[str, tuple[int, str]] = dict(existing or {})
    next_index = max((idx for idx, _ in mapping.values()), default=0) + 1
    for item in retrieved:
        doc_id = item.chunk.document_id
        if doc_id in mapping:
            continue
        filename = str(item.chunk.metadata.get("filename", "documento"))
        mapping[doc_id] = (next_index, filename)
        next_index += 1
    return mapping


def build_context(
    retrieved: list[RetrievedChunk],
    *,
    mapping: dict[str, tuple[int, str]] | None = None,
) -> str:
    if not retrieved:
        return "(Sin fragmentos relevantes recuperados.)"

    doc_map = mapping if mapping is not None else document_index_map(retrieved)
    parts = [
        CITATION_INSTRUCTIONS,
        "",
        "Documentos (cita cada uno como máximo una vez):",
    ]
    for doc_id, (index, filename) in doc_map.items():
        citation = f"<index = {index}, source=rag, title={filename}, id={doc_id}>"
        parts.append(f"- {citation}")
    parts.append("")

    for item in retrieved:
        doc_id = item.chunk.document_id
        index, filename = doc_map[doc_id]
        citation = f"<index = {index}, source=rag, title={filename}, id={doc_id}>"
        parts.append(
            f"[Fragmento doc={index} | score={item.score:.3f} | cita={citation}]"
        )
        parts.append(item.chunk.content)
        parts.append("")
    return "\n".join(parts).strip()


def sources_payload(
    retrieved: list[RetrievedChunk],
    *,
    mapping: dict[str, tuple[int, str]] | None = None,
) -> tuple[dict[str, str | int], ...]:
    doc_map = mapping if mapping is not None else document_index_map(retrieved)
    sources: list[dict[str, str | int]] = []
    for doc_id, (index, filename) in doc_map.items():
        sources.append(
            {
                "index": index,
                "source": "rag",
                "title": filename,
                "id": doc_id,
            }
        )
    return tuple(sources)
