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
    EvalComparisonResult,
    EvalComparisonSample,
    EvalDatasetStatus,
    EvalExperiment,
    EvalRunSample,
    EvalRunStatus,
    EvalRunSummary,
    EvalSuite,
    EvalSuiteConfig,
)
from evals.json_dataset import JsonDatasetImport, parse_json_dataset
from evals.pipeline import RagSampleResult
from chatbot.infrastructure.adapters.persistence.postgres.models import (
    EvalDatasetModel,
    EvalExperimentModel,
    EvalPassageModel,
    EvalQASampleModel,
    EvalRunModel,
    EvalRunResultModel,
    EvalSuiteModel,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500


def _suite_config_from_dict(raw: dict[str, Any]) -> EvalSuiteConfig:
    metrics_raw = raw.get("deepeval_metrics")
    if isinstance(metrics_raw, list):
        deepeval_metrics = tuple(str(item) for item in metrics_raw)
    elif isinstance(metrics_raw, tuple):
        deepeval_metrics = tuple(str(item) for item in metrics_raw)
    else:
        deepeval_metrics = ("answer_relevancy", "faithfulness", "contextual_relevancy")
    return EvalSuiteConfig(
        limit=int(raw.get("limit", 20)),
        distractors=int(raw.get("distractors", 50)),
        top_k=raw.get("top_k"),
        seed=int(raw.get("seed", 42)),
        generate=bool(raw.get("generate", False)),
        ragas=bool(raw.get("ragas", False)),
        ragas_timeout=int(raw.get("ragas_timeout", 600)),
        deepeval=bool(raw.get("deepeval", False)),
        deepeval_timeout=int(raw.get("deepeval_timeout", 600)),
        deepeval_metrics=deepeval_metrics,
        llm_model=raw.get("llm_model"),
        llm_provider=raw.get("llm_provider"),
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
        "deepeval": config.deepeval,
        "deepeval_timeout": config.deepeval_timeout,
        "deepeval_metrics": list(config.deepeval_metrics),
        "llm_model": config.llm_model,
        "llm_provider": config.llm_provider,
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

    async def list_datasets(self) -> list[EvalDatasetStatus]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EvalDatasetModel).order_by(EvalDatasetModel.imported_at.desc())
            )
            rows = result.scalars().all()
            return [
                EvalDatasetStatus(
                    dataset_id=row.id,
                    name=row.name,
                    hf_source=row.hf_source,
                    passage_count=row.passage_count,
                    qa_count=row.qa_count,
                    imported_at=row.imported_at,
                    import_stats=dict(row.import_stats or {}),
                )
                for row in rows
            ]

    async def import_json_dataset(
        self,
        payload: dict[str, Any],
        *,
        dataset_id: str | None = None,
        force: bool = False,
    ) -> EvalDatasetStatus:
        imported = parse_json_dataset(payload, dataset_id=dataset_id)
        if not force and await self.is_dataset_imported(imported.dataset_id):
            status = await self.get_dataset_status(imported.dataset_id)
            assert status is not None
            return status
        await self._persist_json_import(imported, force=force)
        status = await self.get_dataset_status(imported.dataset_id)
        assert status is not None
        return status

    async def _persist_json_import(self, imported: JsonDatasetImport, *, force: bool) -> None:
        dataset_id = imported.dataset_id
        async with self._session_factory() as session:
            if force:
                await session.execute(
                    delete(EvalPassageModel).where(EvalPassageModel.dataset_id == dataset_id)
                )
                await session.execute(
                    delete(EvalQASampleModel).where(EvalQASampleModel.dataset_id == dataset_id)
                )
                await session.execute(
                    delete(EvalDatasetModel).where(EvalDatasetModel.id == dataset_id)
                )
                await session.flush()

            dataset = EvalDatasetModel(
                id=dataset_id,
                name=imported.name,
                hf_source="json",
                passage_count=len(imported.passages),
                qa_count=len(imported.samples),
                import_stats={
                    "source": "json",
                    "description": imported.description,
                    "id_mapping": imported.id_mapping,
                    "input_samples": imported.sanitize_stats.input_samples,
                    "kept_samples": imported.sanitize_stats.kept_samples,
                    "dropped_samples": imported.sanitize_stats.dropped_samples,
                    "dropped_passage_refs": imported.sanitize_stats.dropped_passage_refs,
                },
                imported_at=datetime.now(UTC),
            )
            session.add(dataset)
            await session.flush()

            passage_items = list(imported.passages.values())
            for start in range(0, len(passage_items), _BATCH_SIZE):
                batch = passage_items[start : start + _BATCH_SIZE]
                session.add_all(
                    [
                        EvalPassageModel(
                            dataset_id=dataset_id,
                            passage_id=p.id,
                            text=p.text,
                        )
                        for p in batch
                    ]
                )
                await session.flush()

            for start in range(0, len(imported.samples), _BATCH_SIZE):
                batch = imported.samples[start : start + _BATCH_SIZE]
                session.add_all(
                    [
                        EvalQASampleModel(
                            dataset_id=dataset_id,
                            sample_id=s.id,
                            question=s.question,
                            ground_truth=s.ground_truth,
                            relevant_passage_ids=list(s.relevant_passage_ids),
                        )
                        for s in batch
                    ]
                )
                await session.flush()
            await session.commit()

    async def load_dataset_eval_set(
        self,
        dataset_id: str,
        *,
        limit: int,
        pool_multiplier: int = 5,
    ) -> tuple[list[BioASQSample], dict[int, BioASQPassage], SanitizeStats]:
        if limit < 1:
            raise ValueError("limit debe ser >= 1")

        status = await self.get_dataset_status(dataset_id)
        if status is None or status.passage_count == 0:
            raise ValueError(f"Dataset {dataset_id!r} no importado en Postgres")

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
            raise ValueError(f"No quedaron muestras QA válidas para {dataset_id!r}")
        return samples, corpus, stats

    async def load_bioasq_eval_set(
        self,
        *,
        limit: int,
        pool_multiplier: int = 5,
        dataset_id: str = BIOASQ_DATASET_ID,
    ) -> tuple[list[BioASQSample], dict[int, BioASQPassage], SanitizeStats]:
        return await self.load_dataset_eval_set(
            dataset_id,
            limit=limit,
            pool_multiplier=pool_multiplier,
        )

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
        experiment_id: str | None = None,
        variant_label: str | None = None,
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
                    experiment_id=experiment_id,
                    variant_label=variant_label,
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
        deepeval_metrics: dict[str, Any] | None = None,
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
            if deepeval_metrics is not None:
                row.deepeval_metrics = deepeval_metrics
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

    async def create_experiment(
        self,
        *,
        name: str,
        suite_id: str,
        dataset_id: str,
        run_a_id: str,
        run_b_id: str,
        experiment_id: str | None = None,
    ) -> EvalExperiment:
        resolved_id = experiment_id or str(uuid4())
        async with self._session_factory() as session:
            session.add(
                EvalExperimentModel(
                    id=resolved_id,
                    name=name,
                    suite_id=suite_id,
                    dataset_id=dataset_id,
                    run_a_id=run_a_id,
                    run_b_id=run_b_id,
                )
            )
            await session.commit()
        experiment = await self.get_experiment(resolved_id)
        assert experiment is not None
        return experiment

    async def get_experiment(self, experiment_id: str) -> EvalExperiment | None:
        async with self._session_factory() as session:
            row = await session.get(EvalExperimentModel, experiment_id)
            if row is None:
                return None
            return EvalExperiment(
                id=row.id,
                name=row.name,
                suite_id=row.suite_id,
                dataset_id=row.dataset_id,
                run_a_id=row.run_a_id,
                run_b_id=row.run_b_id,
                created_at=row.created_at,
            )

    async def list_experiments(self, *, limit: int = 50) -> list[EvalExperiment]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EvalExperimentModel)
                .order_by(EvalExperimentModel.created_at.desc())
                .limit(limit)
            )
            rows = result.scalars().all()
            return [
                EvalExperiment(
                    id=row.id,
                    name=row.name,
                    suite_id=row.suite_id,
                    dataset_id=row.dataset_id,
                    run_a_id=row.run_a_id,
                    run_b_id=row.run_b_id,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    async def delete_run(self, run_id: str) -> bool:
        async with self._session_factory() as session:
            row = await session.get(EvalRunModel, run_id)
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def clear_runs_and_experiments(self) -> tuple[int, int]:
        async with self._session_factory() as session:
            experiments_count = await session.scalar(
                select(func.count(EvalExperimentModel.id))  # pylint: disable=not-callable
            )
            runs_count = await session.scalar(
                select(func.count(EvalRunModel.id))  # pylint: disable=not-callable
            )
            await session.execute(delete(EvalExperimentModel))
            await session.execute(delete(EvalRunModel))
            await session.commit()
            return int(runs_count or 0), int(experiments_count or 0)

    async def compare_runs(self, run_a_id: str, run_b_id: str) -> EvalComparisonResult:
        run_a = await self.get_run(run_a_id)
        run_b = await self.get_run(run_b_id)
        if run_a is None or run_b is None:
            raise ValueError("Uno o ambos runs no existen")

        samples_a, _ = await self.get_run_samples(run_a_id, offset=0, limit=10_000)
        samples_b, _ = await self.get_run_samples(run_b_id, offset=0, limit=10_000)
        index_b = {item.sample_id: item for item in samples_b}

        comparison_samples: list[EvalComparisonSample] = []
        wins_b = 0
        comparable = 0
        for sample_a in samples_a:
            sample_b = index_b.get(sample_a.sample_id)
            if sample_b is None:
                continue
            score_a = max(sample_a.scores) if sample_a.scores else 0.0
            score_b = max(sample_b.scores) if sample_b.scores else 0.0
            comparable += 1
            if score_b > score_a:
                wins_b += 1
            comparison_samples.append(
                EvalComparisonSample(
                    sample_id=sample_a.sample_id,
                    question=sample_a.question,
                    ground_truth=sample_a.ground_truth,
                    answer_a=sample_a.answer,
                    answer_b=sample_b.answer,
                    retrieved_a=sample_a.retrieved_passage_ids,
                    retrieved_b=sample_b.retrieved_passage_ids,
                    hit_a=score_a > 0,
                    hit_b=score_b > 0,
                )
            )

        return EvalComparisonResult(
            run_a_id=run_a_id,
            run_b_id=run_b_id,
            run_a_name=run_a.name,
            run_b_name=run_b.name,
            retrieval_delta=_metric_delta(run_a.retrieval_metrics, run_b.retrieval_metrics),
            ragas_delta=_metric_delta(run_a.ragas_metrics, run_b.ragas_metrics),
            deepeval_delta=_metric_delta(run_a.deepeval_metrics, run_b.deepeval_metrics),
            win_rates={
                "retrieval_score_b": wins_b / comparable if comparable else None,
            },
            samples=tuple(comparison_samples),
        )

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
            deepeval_metrics=dict(row.deepeval_metrics) if row.deepeval_metrics is not None else None,
            experiment_id=row.experiment_id,
            variant_label=row.variant_label,
            error=row.error,
            started_at=row.started_at,
            finished_at=row.finished_at,
        )


def _metric_delta(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, float | None]:
    keys = set()
    if left:
        keys.update(left.keys())
    if right:
        keys.update(right.keys())
    delta: dict[str, float | None] = {}
    for key in sorted(keys):
        if key == "sanitize":
            continue
        left_val = _as_float((left or {}).get(key))
        right_val = _as_float((right or {}).get(key))
        if left_val is None or right_val is None:
            delta[key] = None
        else:
            delta[key] = right_val - left_val
    return delta


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
