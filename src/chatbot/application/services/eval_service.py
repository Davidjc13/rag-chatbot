"""Servicio de aplicación para evaluaciones RAGAS / DeepEval / BioASQ."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from chatbot.core.env import Env
from evals.bioasq import BioASQPassage, BioASQSample, collect_relevant_ids, load_bioasq_eval_set
from evals.deepeval_runner import evaluate_with_deepeval
from evals.domain import (
    BIOASQ_DATASET_ID,
    EvalComparisonResult,
    EvalDatasetStatus,
    EvalExperiment,
    EvalRunSample,
    EvalRunSummary,
    EvalSuite,
    EvalSuiteConfig,
)
from evals.pipeline import BioASQRagPipeline
from evals.ragas_runner import evaluate_with_ragas
from evals.retrieval_metrics import compute_retrieval_metrics
from chatbot.infrastructure.adapters.persistence.postgres.eval_repository import (
    PostgresEvalRepository,
)

logger = logging.getLogger(__name__)


class EvalService:
    def __init__(
        self,
        *,
        repository: PostgresEvalRepository,
        env: Env | None = None,
    ) -> None:
        self._repo = repository
        self._env = env or Env.get_instance()
        self._running: set[str] = set()

    async def list_datasets(self) -> list[EvalDatasetStatus]:
        return await self._repo.list_datasets()

    async def get_bioasq_status(self) -> EvalDatasetStatus | None:
        return await self._repo.get_dataset_status(BIOASQ_DATASET_ID)

    async def get_dataset_status(self, dataset_id: str) -> EvalDatasetStatus | None:
        return await self._repo.get_dataset_status(dataset_id)

    async def import_bioasq(
        self,
        *,
        cache_dir: str | Path | None = None,
        force: bool = False,
    ) -> EvalDatasetStatus:
        return await self._repo.import_bioasq(cache_dir=cache_dir, force=force)

    async def import_json_dataset(
        self,
        payload: dict[str, Any],
        *,
        dataset_id: str | None = None,
        force: bool = False,
    ) -> EvalDatasetStatus:
        return await self._repo.import_json_dataset(
            payload,
            dataset_id=dataset_id,
            force=force,
        )

    async def list_suites(self, *, dataset_id: str | None = None) -> list[EvalSuite]:
        return await self._repo.list_suites(dataset_id=dataset_id)

    async def get_suite(self, suite_id: str) -> EvalSuite | None:
        return await self._repo.get_suite(suite_id)

    async def create_suite(
        self,
        *,
        name: str,
        description: str | None = None,
        config: EvalSuiteConfig | None = None,
        sample_ids: list[int] | None = None,
        dataset_id: str = BIOASQ_DATASET_ID,
    ) -> EvalSuite:
        return await self._repo.create_suite(
            name=name,
            description=description,
            config=config or EvalSuiteConfig(),
            sample_ids=sample_ids,
            dataset_id=dataset_id,
        )

    async def update_suite(
        self,
        suite_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        config: EvalSuiteConfig | None = None,
        sample_ids: list[int] | None = None,
    ) -> EvalSuite | None:
        return await self._repo.update_suite(
            suite_id,
            name=name,
            description=description,
            config=config,
            sample_ids=sample_ids,
        )

    async def delete_suite(self, suite_id: str) -> bool:
        return await self._repo.delete_suite(suite_id)

    async def list_runs(self, *, limit: int = 50) -> list[EvalRunSummary]:
        return await self._repo.list_runs(limit=limit)

    async def get_run(self, run_id: str) -> EvalRunSummary | None:
        return await self._repo.get_run(run_id)

    async def get_run_samples(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[EvalRunSample], int]:
        return await self._repo.get_run_samples(run_id, offset=offset, limit=limit)

    async def compare_runs(self, run_a_id: str, run_b_id: str) -> EvalComparisonResult:
        return await self._repo.compare_runs(run_a_id, run_b_id)

    async def list_experiments(self, *, limit: int = 50) -> list[EvalExperiment]:
        return await self._repo.list_experiments(limit=limit)

    async def delete_run(self, run_id: str) -> bool:
        return await self._repo.delete_run(run_id)

    async def clear_runs_and_experiments(self) -> tuple[int, int]:
        return await self._repo.clear_runs_and_experiments()

    async def start_ab_test(
        self,
        *,
        suite_id: str,
        name: str | None,
        variant_a_name: str,
        variant_b_name: str,
        variant_a_config: EvalSuiteConfig | None = None,
        variant_b_config: EvalSuiteConfig | None = None,
    ) -> EvalExperiment:
        suite = await self._repo.get_suite(suite_id)
        if suite is None:
            raise ValueError(f"Suite no encontrada: {suite_id}")

        base_config = suite.config
        config_a = _merge_config(base_config, variant_a_config)
        config_b = _merge_config(base_config, variant_b_config)
        _validate_run_config(config_a)
        _validate_run_config(config_b)

        experiment_id = str(uuid4())
        run_a = await self._repo.create_run(
            dataset_id=suite.dataset_id,
            suite_id=suite_id,
            name=variant_a_name,
            mode=_resolve_mode(config_a),
            config=_config_to_dict(config_a),
            experiment_id=experiment_id,
            variant_label="A",
        )
        run_b = await self._repo.create_run(
            dataset_id=suite.dataset_id,
            suite_id=suite_id,
            name=variant_b_name,
            mode=_resolve_mode(config_b),
            config=_config_to_dict(config_b),
            experiment_id=experiment_id,
            variant_label="B",
        )

        experiment = await self._repo.create_experiment(
            name=name or f"A/B {suite.name}",
            suite_id=suite_id,
            dataset_id=suite.dataset_id,
            run_a_id=run_a.id,
            run_b_id=run_b.id,
            experiment_id=experiment_id,
        )

        asyncio.create_task(
            self._execute_run(run_a.id, suite=suite, config=config_a, cache_dir=None, use_db=True)
        )
        asyncio.create_task(
            self._execute_run(run_b.id, suite=suite, config=config_b, cache_dir=None, use_db=True)
        )
        return experiment

    async def start_run(
        self,
        *,
        suite_id: str | None = None,
        name: str | None = None,
        config: EvalSuiteConfig | None = None,
        cache_dir: str | Path | None = None,
        use_db: bool = True,
        dataset_id: str | None = None,
        experiment_id: str | None = None,
        variant_label: str | None = None,
    ) -> EvalRunSummary:
        suite: EvalSuite | None = None
        run_config = config

        if suite_id is not None:
            suite = await self._repo.get_suite(suite_id)
            if suite is None:
                raise ValueError(f"Suite no encontrada: {suite_id}")
            run_config = _merge_config(suite.config, config)

        if run_config is None:
            run_config = EvalSuiteConfig()

        _validate_run_config(run_config)
        resolved_dataset = dataset_id or (suite.dataset_id if suite else BIOASQ_DATASET_ID)

        run = await self._repo.create_run(
            dataset_id=resolved_dataset,
            suite_id=suite_id,
            name=name or (suite.name if suite else None),
            mode=_resolve_mode(run_config),
            config=_config_to_dict(run_config),
            experiment_id=experiment_id,
            variant_label=variant_label,
        )

        asyncio.create_task(
            self._execute_run(
                run.id,
                suite=suite,
                config=run_config,
                cache_dir=cache_dir,
                use_db=use_db,
            )
        )
        return run

    async def _execute_run(
        self,
        run_id: str,
        *,
        suite: EvalSuite | None,
        config: EvalSuiteConfig,
        cache_dir: str | Path | None,
        use_db: bool,
    ) -> None:
        if run_id in self._running:
            return
        self._running.add(run_id)
        dataset_id = suite.dataset_id if suite else BIOASQ_DATASET_ID
        try:
            await self._repo.update_run_status(run_id, status="running")
            samples, corpus, sanitize_stats = await self._load_samples(
                suite=suite,
                config=config,
                cache_dir=cache_dir,
                use_db=use_db,
                dataset_id=dataset_id,
            )
            passages = self._build_passage_index(samples, corpus, config)
            pipeline = BioASQRagPipeline.from_env(
                self._env,
                top_k=config.top_k,
                with_llm=config.generate,
                llm_model=config.llm_model,
                llm_provider=config.llm_provider,
            )
            await pipeline.index_passages(passages)
            results = await pipeline.run_samples(samples, generate=config.generate)

            retrieval = compute_retrieval_metrics(samples, results)
            retrieval_payload = {
                "hit_at_k": retrieval.hit_at_k,
                "recall_at_k": retrieval.recall_at_k,
                "mrr": retrieval.mrr,
                "sample_count": retrieval.sample_count,
                "k": pipeline.top_k,
                "sanitize": {
                    "input_samples": sanitize_stats.input_samples,
                    "kept_samples": sanitize_stats.kept_samples,
                    "dropped_samples": sanitize_stats.dropped_samples,
                    "dropped_passage_refs": sanitize_stats.dropped_passage_refs,
                    "skipped_nan_passages": sanitize_stats.skipped_nan_passages,
                    "evaluated_samples": len(samples),
                },
            }

            ragas_metrics: dict[str, Any] | None = None
            deepeval_metrics: dict[str, Any] | None = None
            output_dir = Path(f"evals/results/runs/{run_id}")

            if config.ragas:
                summary = await asyncio.to_thread(
                    evaluate_with_ragas,
                    results,
                    env=self._env,
                    output_dir=output_dir,
                    timeout_seconds=config.ragas_timeout,
                )
                ragas_metrics = summary.metrics

            if config.deepeval:
                summary = await asyncio.to_thread(
                    evaluate_with_deepeval,
                    results,
                    env=self._env,
                    output_dir=output_dir,
                    timeout_seconds=config.deepeval_timeout,
                    metric_names=config.deepeval_metrics,
                )
                deepeval_metrics = summary.metrics

            await self._repo.save_run_results(run_id, results)
            await self._repo.update_run_status(
                run_id,
                status="completed",
                retrieval_metrics=retrieval_payload,
                ragas_metrics=ragas_metrics,
                deepeval_metrics=deepeval_metrics,
                finished=True,
            )
            logger.info("Eval run %s completado", run_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Eval run %s falló", run_id)
            await self._repo.update_run_status(
                run_id,
                status="failed",
                error=str(exc),
                finished=True,
            )
        finally:
            self._running.discard(run_id)

    async def _load_samples(
        self,
        *,
        suite: EvalSuite | None,
        config: EvalSuiteConfig,
        cache_dir: str | Path | None,
        use_db: bool,
        dataset_id: str,
    ) -> tuple[list[BioASQSample], dict[int, BioASQPassage], Any]:
        if suite is not None and suite.sample_ids:
            if use_db and await self._repo.is_dataset_imported(dataset_id):
                samples = await self._repo.load_bioasq_qa_from_db(
                    sample_ids=list(suite.sample_ids),
                    dataset_id=dataset_id,
                )
                corpus = await self._repo.load_bioasq_corpus_from_db(dataset_id=dataset_id)
                from evals.bioasq import sanitize_samples_against_corpus

                status = await self._repo.get_dataset_status(dataset_id)
                skipped = int((status.import_stats if status else {}).get("skipped_nan_passages", 0))
                samples, stats = sanitize_samples_against_corpus(
                    samples, set(corpus), skipped_nan_passages=skipped
                )
                return samples, corpus, stats

        if use_db and await self._repo.is_dataset_imported(dataset_id):
            return await self._repo.load_dataset_eval_set(dataset_id, limit=config.limit)

        if dataset_id != BIOASQ_DATASET_ID:
            raise ValueError(
                f"Dataset {dataset_id!r} requiere importación en Postgres antes de evaluar"
            )
        return load_bioasq_eval_set(limit=config.limit, cache_dir=cache_dir)

    @staticmethod
    def _build_passage_index(
        samples: list[BioASQSample],
        corpus: dict[int, BioASQPassage],
        config: EvalSuiteConfig,
    ) -> dict[int, BioASQPassage]:
        relevant_ids = collect_relevant_ids(samples)
        passages = {pid: corpus[pid] for pid in relevant_ids if pid in corpus}
        if config.distractors > 0:
            candidates = [pid for pid in corpus if pid not in relevant_ids]
            rng = random.Random(config.seed)
            for pid in rng.sample(candidates, k=min(config.distractors, len(candidates))):
                passages[pid] = corpus[pid]
        return passages


def _merge_config(base: EvalSuiteConfig, override: EvalSuiteConfig | None) -> EvalSuiteConfig:
    if override is None:
        return base
    return EvalSuiteConfig(
        limit=base.limit,
        distractors=base.distractors,
        top_k=override.top_k if override.top_k is not None else base.top_k,
        seed=base.seed,
        generate=override.generate,
        ragas=override.ragas,
        ragas_timeout=base.ragas_timeout,
        deepeval=override.deepeval,
        deepeval_timeout=base.deepeval_timeout,
        deepeval_metrics=override.deepeval_metrics or base.deepeval_metrics,
        llm_model=override.llm_model or base.llm_model,
        llm_provider=override.llm_provider or base.llm_provider,
    )


def _validate_run_config(config: EvalSuiteConfig) -> None:
    if config.ragas and not config.generate:
        raise ValueError("ragas requiere generate=True")
    if config.deepeval and not config.generate:
        raise ValueError("deepeval requiere generate=True")


def _resolve_mode(config: EvalSuiteConfig) -> str:
    if config.ragas and config.deepeval:
        return "full"
    if config.deepeval:
        return "deepeval"
    if config.ragas:
        return "ragas"
    if config.generate:
        return "generate"
    return "retrieval"


def _config_to_dict(config: EvalSuiteConfig) -> dict[str, Any]:
    return asdict(config)
