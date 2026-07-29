"""Repositorio PostgreSQL para datasets de evaluación, suites y runs."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from evals.bioasq import (
    HF_DATASET,
    BioASQPassage,
    BioASQSample,
    SanitizeStats,
    load_bioasq_corpus,
    load_bioasq_qa,
    sanitize_samples_against_corpus,
)
from evals.domain import (
    BIOASQ_DATASET_ID,
    EvalDatasetStatus,
    EvalRunSample,
    EvalRunStatus,
    EvalRunSummary,
    EvalSuite,
    EvalSuiteConfig,
)
from evals.pipeline import RagSampleResult
from chatbot.infrastructure.adapters.persistence.postgres.models import (
    EvalDatasetModel,
    EvalPassageModel,
    EvalQASampleModel,
    EvalRunModel,
    EvalRunResultModel,
    EvalSuiteModel,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500


def _suite_config_from_dict(raw: dict[str, Any]) -> EvalSuiteConfig:
    return EvalSuiteConfig(
        limit=int(raw.get("limit", 20)),
        distractors=int(raw.get("distractors", 50)),
        top_k=raw.get("top_k"),
        seed=int(raw.get("seed", 42)),
        generate=bool(raw.get("generate", False)),
        ragas=bool(raw.get("ragas", False)),
        ragas_timeout=int(raw.get("ragas_timeout", 600)),
    )


def _suite_config_to_dict(config: EvalSuiteConfig) -> dict[str, Any]:
    return {
        "limit": config.limit,
        "distractors": config.distractors,
        "top_k": config.top_k,
        "seed": config.seed,
        "generate": config.generate,
        "ragas": config.ragas,
        "ragas_timeout": config.ragas_timeout,
    }


class PostgresEvalRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_dataset_status(self, dataset_id: str = BIOASQ_DATASET_ID) -> EvalDatasetStatus | None:
        async with self._session_factory() as session:
            row = await session.get(EvalDatasetModel, dataset_id)
            if row is None:
                return None
            return EvalDatasetStatus(
                dataset_id=row.id,
                name=row.name,
                hf_source=row.hf_source,
                passage_count=row.passage_count,
                qa_count=row.qa_count,
                imported_at=row.imported_at,
                import_stats=dict(row.import_stats or {}),
            )

    async def is_dataset_imported(self, dataset_id: str = BIOASQ_DATASET_ID) -> bool:
        status = await self.get_dataset_status(dataset_id)
        return status is not None and status.passage_count > 0

    async def import_bioasq(
        self,
        *,
        cache_dir: str | Path | None = None,
        force: bool = False,
    ) -> EvalDatasetStatus:
        if not force and await self.is_dataset_imported():
            status = await self.get_dataset_status()
            assert status is not None
            logger.info(
                "Dataset BioASQ ya importado (%s pasajes, %s QA)",
                status.passage_count,
                status.qa_count,
            )
            return status

        qa_samples = load_bioasq_qa(limit=None, cache_dir=cache_dir)
        corpus, skipped_nan = load_bioasq_corpus(passage_ids=None, cache_dir=cache_dir)
        cleaned, stats = sanitize_samples_against_corpus(
            qa_samples,
            set(corpus),
            skipped_nan_passages=skipped_nan,
        )

        async with self._session_factory() as session:
            if force:
                await session.execute(
                    delete(EvalPassageModel).where(EvalPassageModel.dataset_id == BIOASQ_DATASET_ID)
                )
                await session.execute(
                    delete(EvalQASampleModel).where(
                        EvalQASampleModel.dataset_id == BIOASQ_DATASET_ID
                    )
                )
                await session.execute(
                    delete(EvalDatasetModel).where(EvalDatasetModel.id == BIOASQ_DATASET_ID)
                )
                await session.flush()

            dataset = EvalDatasetModel(
                id=BIOASQ_DATASET_ID,
                name="BioASQ mini",
                hf_source=HF_DATASET,
                passage_count=0,
                qa_count=0,
                import_stats={
                    "input_samples": stats.input_samples,
                    "kept_samples": stats.kept_samples,
                    "dropped_samples": stats.dropped_samples,
                    "dropped_passage_refs": stats.dropped_passage_refs,
                    "skipped_nan_passages": stats.skipped_nan_passages,
                },
                imported_at=datetime.now(UTC),
            )
            session.add(dataset)
            await session.flush()

            passage_items = list(corpus.values())
            for start in range(0, len(passage_items), _BATCH_SIZE):
                batch = passage_items[start : start + _BATCH_SIZE]
                session.add_all(
                    [
                        EvalPassageModel(
                            dataset_id=BIOASQ_DATASET_ID,
                            passage_id=p.id,
                            text=p.text,
                        )
                        for p in batch
                    ]
                )
                await session.flush()

            for start in range(0, len(cleaned), _BATCH_SIZE):
                batch = cleaned[start : start + _BATCH_SIZE]
                session.add_all(
                    [
                        EvalQASampleModel(
                            dataset_id=BIOASQ_DATASET_ID,
                            sample_id=s.id,
                            question=s.question,
                            ground_truth=s.ground_truth,
                            relevant_passage_ids=list(s.relevant_passage_ids),
                        )
                        for s in batch
                    ]
                )
                await session.flush()

            dataset.passage_count = len(corpus)
            dataset.qa_count = len(cleaned)
            await session.commit()

        logger.info(
            "BioASQ importado en Postgres: %s pasajes, %s QA",
            len(corpus),
            len(cleaned),
        )
        status = await self.get_dataset_status()
        assert status is not None
        return status

    async def load_bioasq_corpus_from_db(
        self,
        *,
        passage_ids: set[int] | None = None,
        dataset_id: str = BIOASQ_DATASET_ID,
    ) -> dict[int, BioASQPassage]:
        async with self._session_factory() as session:
            query = select(EvalPassageModel).where(EvalPassageModel.dataset_id == dataset_id)
            if passage_ids is not None:
                query = query.where(EvalPassageModel.passage_id.in_(passage_ids))
            result = await session.execute(query)
            rows = result.scalars().all()
            return {row.passage_id: BioASQPassage(id=row.passage_id, text=row.text) for row in rows}

    async def load_bioasq_qa_from_db(
        self,
        *,
        limit: int | None = None,
        sample_ids: list[int] | None = None,
        dataset_id: str = BIOASQ_DATASET_ID,
    ) -> list[BioASQSample]:
        async with self._session_factory() as session:
            query = (
                select(EvalQASampleModel)
                .where(EvalQASampleModel.dataset_id == dataset_id)
                .order_by(EvalQASampleModel.sample_id)
            )
            if sample_ids is not None:
                query = query.where(EvalQASampleModel.sample_id.in_(sample_ids))
            if limit is not None:
                query = query.limit(limit)
            result = await session.execute(query)
            rows = result.scalars().all()
            return [
                BioASQSample(
                    id=row.sample_id,
                    question=row.question,
                    ground_truth=row.ground_truth,
                    relevant_passage_ids=tuple(row.relevant_passage_ids),
                )
                for row in rows
            ]

    async def load_bioasq_eval_set(
        self,
        *,
        limit: int,
        pool_multiplier: int = 5,
        dataset_id: str = BIOASQ_DATASET_ID,
    ) -> tuple[list[BioASQSample], dict[int, BioASQPassage], SanitizeStats]:
        if limit < 1:
            raise ValueError("limit debe ser >= 1")

        status = await self.get_dataset_status(dataset_id)
        if status is None or status.passage_count == 0:
            raise ValueError(
                "Dataset BioASQ no importado en Postgres. "
                "Ejecuta la importación desde la UI o POST /api/v1/evals/datasets/bioasq/import"
            )

        pool = max(limit * max(1, pool_multiplier), limit)
        candidates = await self.load_bioasq_qa_from_db(limit=pool, dataset_id=dataset_id)
        corpus = await self.load_bioasq_corpus_from_db(dataset_id=dataset_id)
        cleaned, stats = sanitize_samples_against_corpus(
            candidates,
            set(corpus),
            skipped_nan_passages=int(status.import_stats.get("skipped_nan_passages", 0)),
        )
        samples = cleaned[:limit]
        if not samples:
            raise ValueError("No quedaron muestras QA válidas en Postgres")
        return samples, corpus, stats

    async def create_suite(
        self,
        *,
        name: str,
        description: str | None,
        config: EvalSuiteConfig,
        sample_ids: list[int] | None = None,
        dataset_id: str = BIOASQ_DATASET_ID,
    ) -> EvalSuite:
        if sample_ids is None:
            pool = max(config.limit * 5, config.limit)
            qa = await self.load_bioasq_qa_from_db(limit=pool, dataset_id=dataset_id)
            sample_ids = [s.id for s in qa[: config.limit]]

        suite_id = str(uuid4())
        async with self._session_factory() as session:
            session.add(
                EvalSuiteModel(
                    id=suite_id,
                    name=name,
                    description=description,
                    dataset_id=dataset_id,
                    config=_suite_config_to_dict(config),
                    sample_ids=sample_ids,
                )
            )
            await session.commit()

        suite = await self.get_suite(suite_id)
        assert suite is not None
        return suite

    async def list_suites(self, *, dataset_id: str | None = None) -> list[EvalSuite]:
        async with self._session_factory() as session:
            query = select(EvalSuiteModel).order_by(EvalSuiteModel.created_at.desc())
            if dataset_id is not None:
                query = query.where(EvalSuiteModel.dataset_id == dataset_id)
            result = await session.execute(query)
            rows = result.scalars().all()
            return [self._row_to_suite(row) for row in rows]

    async def get_suite(self, suite_id: str) -> EvalSuite | None:
        async with self._session_factory() as session:
            row = await session.get(EvalSuiteModel, suite_id)
            if row is None:
                return None
            return self._row_to_suite(row)

    async def update_suite(
        self,
        suite_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        config: EvalSuiteConfig | None = None,
        sample_ids: list[int] | None = None,
    ) -> EvalSuite | None:
        async with self._session_factory() as session:
            row = await session.get(EvalSuiteModel, suite_id)
            if row is None:
                return None
            if name is not None:
                row.name = name
            if description is not None:
                row.description = description
            if config is not None:
                row.config = _suite_config_to_dict(config)
            if sample_ids is not None:
                row.sample_ids = sample_ids
            await session.commit()
        return await self.get_suite(suite_id)

    async def delete_suite(self, suite_id: str) -> bool:
        async with self._session_factory() as session:
            row = await session.get(EvalSuiteModel, suite_id)
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def create_run(
        self,
        *,
        dataset_id: str,
        suite_id: str | None,
        name: str | None,
        mode: str,
        config: dict[str, Any],
    ) -> EvalRunSummary:
        run_id = str(uuid4())
        async with self._session_factory() as session:
            session.add(
                EvalRunModel(
                    id=run_id,
                    suite_id=suite_id,
                    dataset_id=dataset_id,
                    name=name,
                    status="pending",
                    mode=mode,
                    config=config,
                )
            )
            await session.commit()
        run = await self.get_run(run_id)
        assert run is not None
        return run

    async def update_run_status(
        self,
        run_id: str,
        *,
        status: EvalRunStatus,
        retrieval_metrics: dict[str, Any] | None = None,
        ragas_metrics: dict[str, Any] | None = None,
        error: str | None = None,
        finished: bool = False,
    ) -> None:
        async with self._session_factory() as session:
            row = await session.get(EvalRunModel, run_id)
            if row is None:
                return
            row.status = status
            if retrieval_metrics is not None:
                row.retrieval_metrics = retrieval_metrics
            if ragas_metrics is not None:
                row.ragas_metrics = ragas_metrics
            if error is not None:
                row.error = error
            if finished:
                row.finished_at = datetime.now(UTC)
            await session.commit()

    async def save_run_results(self, run_id: str, results: list[RagSampleResult]) -> None:
        async with self._session_factory() as session:
            await session.execute(
                delete(EvalRunResultModel).where(EvalRunResultModel.run_id == run_id)
            )
            session.add_all(
                [
                    EvalRunResultModel(
                        run_id=run_id,
                        sample_id=r.sample_id,
                        question=r.question,
                        ground_truth=r.ground_truth,
                        answer=r.answer,
                        contexts=list(r.contexts),
                        retrieved_passage_ids=list(r.retrieved_passage_ids),
                        scores=list(r.scores),
                    )
                    for r in results
                ]
            )
            await session.commit()

    async def list_runs(self, *, limit: int = 50) -> list[EvalRunSummary]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EvalRunModel).order_by(EvalRunModel.started_at.desc()).limit(limit)
            )
            rows = result.scalars().all()
            return [self._row_to_run(row) for row in rows]

    async def get_run(self, run_id: str) -> EvalRunSummary | None:
        async with self._session_factory() as session:
            row = await session.get(EvalRunModel, run_id)
            if row is None:
                return None
            return self._row_to_run(row)

    async def get_run_samples(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[EvalRunSample], int]:
        async with self._session_factory() as session:
            total = await session.scalar(
                select(func.count(EvalRunResultModel.id)).where(  # pylint: disable=not-callable
                    EvalRunResultModel.run_id == run_id
                )
            )
            result = await session.execute(
                select(EvalRunResultModel)
                .where(EvalRunResultModel.run_id == run_id)
                .order_by(EvalRunResultModel.sample_id)
                .offset(offset)
                .limit(limit)
            )
            rows = result.scalars().all()
            samples = [
                EvalRunSample(
                    sample_id=row.sample_id,
                    question=row.question,
                    ground_truth=row.ground_truth,
                    answer=row.answer,
                    contexts=tuple(row.contexts or []),
                    retrieved_passage_ids=tuple(row.retrieved_passage_ids or []),
                    scores=tuple(row.scores or []),
                )
                for row in rows
            ]
            return samples, int(total or 0)

    @staticmethod
    def _row_to_suite(row: EvalSuiteModel) -> EvalSuite:
        return EvalSuite(
            id=row.id,
            name=row.name,
            dataset_id=row.dataset_id,
            description=row.description,
            config=_suite_config_from_dict(dict(row.config or {})),
            sample_ids=tuple(row.sample_ids or []),
            created_at=row.created_at,
        )

    @staticmethod
    def _row_to_run(row: EvalRunModel) -> EvalRunSummary:
        return EvalRunSummary(
            id=row.id,
            suite_id=row.suite_id,
            dataset_id=row.dataset_id,
            name=row.name,
            status=row.status,  # type: ignore[arg-type]
            mode=row.mode,  # type: ignore[arg-type]
            config=dict(row.config or {}),
            retrieval_metrics=dict(row.retrieval_metrics) if row.retrieval_metrics else None,
            ragas_metrics=dict(row.ragas_metrics) if row.ragas_metrics else None,
            error=row.error,
            started_at=row.started_at,
            finished_at=row.finished_at,
        )
