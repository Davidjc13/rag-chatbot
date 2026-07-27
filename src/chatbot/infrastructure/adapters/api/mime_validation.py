"""Validación de MIME, extensión y magic bytes para ingestión segura."""

from __future__ import annotations

from pathlib import Path

from chatbot.domain.exceptions import InvalidMimeTypeError, ValidationError

# MIME allowlist → extensiones permitidas
_MIME_TO_EXTENSIONS: dict[str, frozenset[str]] = {
    "application/pdf": frozenset({".pdf"}),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": frozenset(
        {".docx"}
    ),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": frozenset({".xlsx"}),
    "application/vnd.ms-excel.sheet.macroenabled.12": frozenset({".xlsm"}),
}

_ALLOWED_MIME_TYPES = frozenset(_MIME_TO_EXTENSIONS)


def normalize_mime(content_type: str | None) -> str:
    """Normaliza Content-Type (minúsculas, sin parámetros)."""
    if not content_type:
        return ""
    return content_type.split(";", maxsplit=1)[0].strip().lower()


def _extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def _assert_magic_bytes(data: bytes, mime: str) -> None:
    if mime == "application/pdf":
        if not data.startswith(b"%PDF"):
            raise InvalidMimeTypeError(
                "El contenido no coincide con un PDF válido (magic bytes)"
            )
        return

    # DOCX / XLSX / XLSM son ZIP (OOXML)
    if not data.startswith(b"PK"):
        raise InvalidMimeTypeError(
            "El contenido no coincide con un documento Office Open XML (magic bytes)"
        )


def validate_upload(*, filename: str, content_type: str | None, data: bytes) -> str:
    """
    Valida MIME, extensión y magic bytes.

    Devuelve el MIME normalizado si todo es correcto.
    """
    name = (filename or "").strip()
    if not name:
        raise ValidationError("El nombre del fichero es obligatorio")
    if not data:
        raise ValidationError("El fichero está vacío")

    mime = normalize_mime(content_type)
    if mime not in _ALLOWED_MIME_TYPES:
        raise InvalidMimeTypeError(
            f"Tipo MIME no permitido: {content_type or '(vacío)'}. "
            f"Permitidos: {', '.join(sorted(_ALLOWED_MIME_TYPES))}"
        )

    extension = _extension_of(name)
    allowed_exts = _MIME_TO_EXTENSIONS[mime]
    if extension not in allowed_exts:
        raise InvalidMimeTypeError(
            f"La extensión '{extension or '(ninguna)'}' no coincide con el MIME '{mime}'"
        )

    _assert_magic_bytes(data, mime)
    return mime
