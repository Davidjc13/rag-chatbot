"""Tests del repositorio y servicio de evaluaciones."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from evals.bioasq import BioASQPassage, BioASQSample, SanitizeStats
from evals.domain import EvalSuite, EvalSuiteConfig
from chatbot.application.services.eval_service import EvalService
from chatbot.infrastructure.adapters.persistence.postgres.eval_repository import (
    PostgresEvalRepository,
    _suite_config_from_dict,
    _suite_config_to_dict,
)
from chatbot.infrastructure.adapters.persistence.postgres.models import EvalSuiteModel


def test_suite_config_roundtrip() -> None:
    config = EvalSuiteConfig(limit=15, distractors=30, top_k=5, generate=True, ragas=True)
    raw = _suite_config_to_dict(config)
    restored = _suite_config_from_dict(raw)
    assert restored == config


def test_row_to_suite() -> None:
    row = EvalSuiteModel(
        id="suite-1",
        name="Test suite",
        description="desc",
        dataset_id="bioasq",
        config=_suite_config_to_dict(EvalSuiteConfig(limit=10)),
        sample_ids=[1, 2, 3],
        created_at=datetime.now(UTC),
    )
    suite = PostgresEvalRepository._row_to_suite(row)
    assert suite.id == "suite-1"
    assert suite.sample_ids == (1, 2, 3)
    assert suite.config.limit == 10


@pytest.mark.asyncio
async def test_eval_service_start_run_validates_ragas() -> None:
    repo = AsyncMock()
    service = EvalService(repository=repo)

    with pytest.raises(ValueError, match="ragas requiere generate"):
        await service.start_run(config=EvalSuiteConfig(ragas=True, generate=False))


@pytest.mark.asyncio
async def test_eval_service_start_run_validates_deepeval() -> None:
    repo = AsyncMock()
    service = EvalService(repository=repo)

    with pytest.raises(ValueError, match="deepeval requiere generate"):
        await service.start_run(config=EvalSuiteConfig(deepeval=True, generate=False))


@pytest.mark.asyncio
async def test_eval_service_build_passage_index() -> None:
    repo = AsyncMock()
    service = EvalService(repository=repo)
    samples = [
        BioASQSample(id=1, question="q", ground_truth="a", relevant_passage_ids=(10,)),
    ]
    corpus = {
        10: BioASQPassage(id=10, text="gold"),
        20: BioASQPassage(id=20, text="noise"),
        30: BioASQPassage(id=30, text="noise2"),
    }
    passages = service._build_passage_index(
        samples,
        corpus,
        EvalSuiteConfig(distractors=1, seed=42),
    )
    assert 10 in passages
    assert len(passages) == 2


@pytest.mark.asyncio
async def test_eval_service_load_samples_from_db_when_imported() -> None:
    repo = AsyncMock()
    repo.is_dataset_imported.return_value = True
    repo.load_bioasq_eval_set.return_value = (
        [BioASQSample(id=1, question="q", ground_truth="a", relevant_passage_ids=(1,))],
        {1: BioASQPassage(id=1, text="p")},
        SanitizeStats(1, 1, 0, 0),
    )
    service = EvalService(repository=repo)
    samples, corpus, stats = await service._load_samples(
        suite=None,
        config=EvalSuiteConfig(limit=5),
        cache_dir=None,
        use_db=True,
    )
    assert len(samples) == 1
    assert stats.kept_samples == 1
    repo.load_bioasq_eval_set.assert_awaited_once()


def test_eval_suite_config_asdict() -> None:
    config = EvalSuiteConfig(limit=3, generate=True)
    payload = asdict(config)
    assert payload["limit"] == 3
    assert payload["generate"] is True
