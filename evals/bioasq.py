"""Carga y normalización del dataset rag-mini-bioasq (Hugging Face)."""

from __future__ import annotations

import ast
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HF_DATASET = "rag-datasets/rag-mini-bioasq"
QA_CONFIG = "question-answer-passages"
CORPUS_CONFIG = "text-corpus"


@dataclass(frozen=True, slots=True)
class BioASQSample:
    id: int
    question: str
    ground_truth: str
    relevant_passage_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BioASQPassage:
    id: int
    text: str


@dataclass(frozen=True, slots=True)
class SanitizeStats:
    input_samples: int
    kept_samples: int
    dropped_samples: int
    dropped_passage_refs: int
    skipped_nan_passages: int = 0


def is_usable_passage_text(value: Any) -> bool:
    """True si el pasaje tiene texto usable (filtra None / NaN / vacíos)."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    # numpy scalars (p.ej. np.float64('nan'))
    item = getattr(value, "item", None)
    if callable(item):
        try:
            raw = item()
        except (ValueError, TypeError):
            raw = value
        else:
            if isinstance(raw, float) and math.isnan(raw):
                return False
            value = raw
    text = str(value).strip()
    if not text:
        return False
    return text.casefold() not in {"nan", "none", "null", "<na>"}


def parse_passage_ids(raw: Any) -> tuple[int, ...]:
    """Parsea `relevant_passage_ids` (lista, JSON o literal Python)."""
    if raw is None:
        return ()
    if isinstance(raw, float) and math.isnan(raw):
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw if x is not None and not (
            isinstance(x, float) and math.isnan(x)
        ))
    text = str(raw).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return ()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"relevant_passage_ids inválido: {raw!r}")
    return tuple(
        int(x)
        for x in parsed
        if x is not None and not (isinstance(x, float) and math.isnan(x))
    )


def load_bioasq_qa(
    *,
    limit: int | None = None,
    cache_dir: str | Path | None = None,
) -> list[BioASQSample]:
    """Descarga el split de preguntas/respuestas y opcionalmente limita el tamaño."""
    from datasets import load_dataset

    dataset = load_dataset(
        HF_DATASET,
        QA_CONFIG,
        split="test",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit debe ser >= 1")
        dataset = dataset.select(range(min(limit, len(dataset))))

    samples: list[BioASQSample] = []
    for row in dataset:
        question = str(row["question"] or "").strip()
        answer = str(row["answer"] or "").strip()
        if not question or not answer or question.casefold() == "nan":
            continue
        if answer.casefold() == "nan":
            continue
        ids = parse_passage_ids(row["relevant_passage_ids"])
        if not ids:
            continue
        samples.append(
            BioASQSample(
                id=int(row["id"]),
                question=question,
                ground_truth=answer,
                relevant_passage_ids=ids,
            )
        )
    logger.info("Cargadas %s muestras QA de %s", len(samples), HF_DATASET)
    return samples


def load_bioasq_corpus(
    *,
    passage_ids: set[int] | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[dict[int, BioASQPassage], int]:
    """
    Descarga el corpus de pasajes.

    Si `passage_ids` se indica, solo se materializan esos ids.
    Omite pasajes NaN/vacíos. Devuelve (passages, skipped_nan_count).
    """
    from datasets import load_dataset

    dataset = load_dataset(
        HF_DATASET,
        CORPUS_CONFIG,
        split="passages",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    wanted = passage_ids
    passages: dict[int, BioASQPassage] = {}
    skipped_nan = 0
    for row in dataset:
        pid = int(row["id"])
        if wanted is not None and pid not in wanted:
            continue
        text = row.get("passage")
        if not is_usable_passage_text(text):
            skipped_nan += 1
            continue
        passages[pid] = BioASQPassage(id=pid, text=str(text).strip())
    logger.info(
        "Cargados %s pasajes del corpus (omitidos NaN/vacíos: %s)",
        len(passages),
        skipped_nan,
    )
    return passages, skipped_nan


def collect_relevant_ids(samples: list[BioASQSample]) -> set[int]:
    ids: set[int] = set()
    for sample in samples:
        ids.update(sample.relevant_passage_ids)
    return ids


def sanitize_samples_against_corpus(
    samples: list[BioASQSample],
    available_passage_ids: set[int],
    *,
    skipped_nan_passages: int = 0,
) -> tuple[list[BioASQSample], SanitizeStats]:
    """
    Recorta `relevant_passage_ids` a pasajes existentes y descarta muestras sin gold usable.

    Así recall/hit no penalizan por ids gold que apuntan a NaN del corpus.
    """
    cleaned: list[BioASQSample] = []
    dropped_refs = 0
    dropped_samples = 0
    for sample in samples:
        valid = tuple(pid for pid in sample.relevant_passage_ids if pid in available_passage_ids)
        dropped_refs += len(sample.relevant_passage_ids) - len(valid)
        if not valid:
            dropped_samples += 1
            continue
        if valid != sample.relevant_passage_ids:
            cleaned.append(replace(sample, relevant_passage_ids=valid))
        else:
            cleaned.append(sample)

    stats = SanitizeStats(
        input_samples=len(samples),
        kept_samples=len(cleaned),
        dropped_samples=dropped_samples,
        dropped_passage_refs=dropped_refs,
        skipped_nan_passages=skipped_nan_passages,
    )
    logger.info(
        "Sanitizado BioASQ: kept=%s/%s samples, refs_nan_omitidos=%s, corpus_nan=%s",
        stats.kept_samples,
        stats.input_samples,
        stats.dropped_passage_refs,
        stats.skipped_nan_passages,
    )
    return cleaned, stats


def load_bioasq_eval_set(
    *,
    limit: int,
    cache_dir: str | Path | None = None,
    pool_multiplier: int = 5,
) -> tuple[list[BioASQSample], dict[int, BioASQPassage], SanitizeStats]:
    """
    Carga QA + corpus completo usable, elimina refs a pasajes NaN y corta a `limit`.

    Devuelve (samples, corpus_sin_nan, stats). El caller añade distractores desde corpus.
    """
    if limit < 1:
        raise ValueError("limit debe ser >= 1")

    pool = max(limit * max(1, pool_multiplier), limit)
    candidates = load_bioasq_qa(limit=pool, cache_dir=cache_dir)
    corpus, skipped_nan = load_bioasq_corpus(passage_ids=None, cache_dir=cache_dir)
    cleaned, stats = sanitize_samples_against_corpus(
        candidates,
        set(corpus),
        skipped_nan_passages=skipped_nan,
    )
    samples = cleaned[:limit]
    if not samples:
        raise ValueError(
            "No quedaron muestras QA con pasajes gold válidos tras filtrar NaN del corpus"
        )
    if len(samples) < limit:
        logger.warning(
            "Solo %s/%s muestras válidas tras filtrar NaN (pool=%s)",
            len(samples),
            limit,
            pool,
        )
    return samples, corpus, stats
