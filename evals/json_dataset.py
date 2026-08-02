"""Carga y validación de datasets de evaluación en formato JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from evals.bioasq import BioASQPassage, BioASQSample, SanitizeStats

_DATASET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


class JsonPassageSpec(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1)


class JsonSampleSpec(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    relevant_passage_ids: list[str] = Field(..., min_length=1)

    @field_validator("relevant_passage_ids")
    @classmethod
    def _non_empty_ids(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("relevant_passage_ids no puede estar vacío")
        return cleaned


class JsonDatasetSpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=2000)
    passages: list[JsonPassageSpec] = Field(..., min_length=1)
    samples: list[JsonSampleSpec] = Field(..., min_length=1)


@dataclass(frozen=True, slots=True)
class JsonDatasetImport:
    dataset_id: str
    name: str
    description: str | None
    passages: dict[int, BioASQPassage]
    samples: list[BioASQSample]
    id_mapping: dict[str, int]
    sanitize_stats: SanitizeStats


def slugify_dataset_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower())
    slug = slug.strip("-_")
    if not slug or not _DATASET_ID_RE.match(slug):
        raise ValueError(
            "dataset_id inválido; usa minúsculas, números, guiones y guiones bajos (máx. 63)"
        )
    return slug


def load_json_dataset_spec(
    payload: dict[str, Any] | str | Path,
    *,
    dataset_id: str | None = None,
) -> JsonDatasetSpec:
    if isinstance(payload, Path):
        raw = json.loads(payload.read_text(encoding="utf-8"))
    elif isinstance(payload, str):
        raw = json.loads(payload)
    else:
        raw = payload
    spec = JsonDatasetSpec.model_validate(raw)
    if dataset_id is not None:
        slugify_dataset_id(dataset_id)
    return spec


def parse_json_dataset(
    payload: dict[str, Any] | str | Path,
    *,
    dataset_id: str | None = None,
) -> JsonDatasetImport:
    """Convierte un dataset JSON a pasajes/muestras con IDs enteros internos."""
    spec = load_json_dataset_spec(payload, dataset_id=dataset_id)
    resolved_id = slugify_dataset_id(dataset_id or spec.name)

    passage_key_to_int: dict[str, int] = {}
    passages: dict[int, BioASQPassage] = {}
    for index, passage in enumerate(spec.passages, start=1):
        key = passage.id.strip()
        if key in passage_key_to_int:
            raise ValueError(f"ID de pasaje duplicado: {key!r}")
        passage_key_to_int[key] = index
        passages[index] = BioASQPassage(id=index, text=passage.text.strip())

    samples: list[BioASQSample] = []
    dropped_refs = 0
    for index, sample in enumerate(spec.samples, start=1):
        relevant: list[int] = []
        for ref in sample.relevant_passage_ids:
            pid = passage_key_to_int.get(ref.strip())
            if pid is None:
                dropped_refs += 1
                continue
            relevant.append(pid)
        if not relevant:
            continue
        samples.append(
            BioASQSample(
                id=index,
                question=sample.question.strip(),
                ground_truth=sample.answer.strip(),
                relevant_passage_ids=tuple(dict.fromkeys(relevant)),
            )
        )

    if not samples:
        raise ValueError("Ninguna muestra válida tras resolver relevant_passage_ids")

    stats = SanitizeStats(
        input_samples=len(spec.samples),
        kept_samples=len(samples),
        dropped_samples=len(spec.samples) - len(samples),
        dropped_passage_refs=dropped_refs,
        skipped_nan_passages=0,
    )
    return JsonDatasetImport(
        dataset_id=resolved_id,
        name=spec.name.strip(),
        description=spec.description.strip() if spec.description else None,
        passages=passages,
        samples=samples,
        id_mapping=passage_key_to_int,
        sanitize_stats=stats,
    )


def dataset_template_path() -> Path:
    """Ruta a la plantilla JSON (versionada; no depende de evals/data/)."""
    here = Path(__file__).resolve().parent
    candidates = (
        here / "templates" / "dataset.template.json",
        here / "data" / "dataset.template.json",
        Path(__file__).resolve().parents[2]
        / "src"
        / "chatbot"
        / "infrastructure"
        / "adapters"
        / "api"
        / "static"
        / "dataset.template.json",
        Path(__file__).resolve().parents[1]
        / "chatbot"
        / "infrastructure"
        / "adapters"
        / "api"
        / "static"
        / "dataset.template.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    return here / "templates" / "dataset.template.json"
