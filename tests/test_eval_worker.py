"""Tests de cola de evaluaciones y worker."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatbot.application.services.eval_service import EvalService
from chatbot.application.services.eval_worker import EvalJobWorker
from chatbot.core.env import Env
from evals.domain import BIOASQ_DATASET_ID, EvalRunSummary, EvalSuiteConfig


def _run_summary(*, run_id: str = "run-1", status: str = "queued") -> EvalRunSummary:
    now = datetime.now(UTC)
    return EvalRunSummary(
        id=run_id,
        suite_id=None,
        dataset_id=BIOASQ_DATASET_ID,
        name="demo",
        status=status,  # type: ignore[arg-type]
        mode="retrieval",
        config={},
        retrieval_metrics=None,
        ragas_metrics=None,
        deepeval_metrics=None,
        experiment_id=None,
        variant_label=None,
        error=None,
        started_at=now,
        finished_at=None,
    )


@pytest.mark.asyncio
async def test_start_run_enqueues_without_executing() -> None:
    repo = AsyncMock()
    queued = _run_summary()
    repo.create_run.return_value = queued
    service = EvalService(repository=repo)
    result = await service.start_run(config=EvalSuiteConfig())
    assert result.id == "run-1"
    repo.create_run.assert_awaited_once()
    assert service._running == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_eval_worker_claims_and_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_WORKER_ENABLED", "true")
    monkeypatch.setenv("EVAL_WORKER_POLL_SECONDS", "0.05")
    Env.reset()
    env = Env.get_instance()

    repo = AsyncMock()
    repo.requeue_expired_leases.return_value = 1
    claimed = _run_summary(status="running")
    repo.claim_next_run.side_effect = [claimed, None]
    repo.get_suite.return_value = None

    service = MagicMock()
    service.execute_run = AsyncMock()

    worker = EvalJobWorker(service=service, repository=repo, env=env)
    task = asyncio.create_task(worker.run_forever())
    await asyncio.sleep(0.2)
    worker.stop()
    await task

    repo.requeue_expired_leases.assert_awaited()
    service.execute_run.assert_awaited()
    Env.reset()
