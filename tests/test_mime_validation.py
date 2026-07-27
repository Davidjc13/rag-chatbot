"""Tests de validación MIME en ingestión."""

from __future__ import annotations

import io

import pytest
from docx import Document

from chatbot.domain.exceptions import InvalidMimeTypeError, ValidationError
from chatbot.infrastructure.adapters.api.mime_validation import (
    normalize_mime,
    validate_upload,
)


def _docx_bytes() -> bytes:
    document = Document()
    document.add_paragraph("Hola")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_normalize_mime_strips_params() -> None:
    assert normalize_mime("application/pdf; charset=binary") == "application/pdf"


def test_validate_upload_accepts_docx() -> None:
    data = _docx_bytes()
    mime = validate_upload(
        filename="policy.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        data=data,
    )
    assert "wordprocessingml" in mime


def test_validate_upload_rejects_wrong_mime() -> None:
    with pytest.raises(InvalidMimeTypeError):
        validate_upload(
            filename="policy.docx",
            content_type="text/plain",
            data=_docx_bytes(),
        )


def test_validate_upload_rejects_mime_extension_mismatch() -> None:
    with pytest.raises(InvalidMimeTypeError, match="no coincide"):
        validate_upload(
            filename="policy.pdf",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            data=_docx_bytes(),
        )


def test_validate_upload_rejects_bad_magic() -> None:
    with pytest.raises(InvalidMimeTypeError, match="magic"):
        validate_upload(
            filename="fake.pdf",
            content_type="application/pdf",
            data=b"not-a-pdf",
        )


def test_validate_upload_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        validate_upload(
            filename="empty.pdf",
            content_type="application/pdf",
            data=b"",
        )


def test_validate_upload_accepts_pdf_magic() -> None:
    mime = validate_upload(
        filename="doc.pdf",
        content_type="application/pdf",
        data=b"%PDF-1.4 fake content",
    )
    assert mime == "application/pdf"
