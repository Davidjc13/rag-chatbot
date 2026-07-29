"""Servicio de aplicación para evaluaciones RAGAS / BioASQ."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chatbot.core.env import Env
from evals.bioasq import BioASQPassage, BioASQSample, collect_relevant_ids, load_bioasq_eval_set
from evals.domain import (
    BIOASQ_DATASET_ID,
    EvalDatasetStatus,
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

    async def get_bioasq_status(self) -> EvalDatasetStatus | None:
        return await self._repo.get_dataset_status(BIOASQ_DATASET_ID)

    async def import_bioasq(
        self,
        *,
        cache_dir: str | Path | None = None,
        force: bool = False,
    ) -> EvalDatasetStatus:
        return await self._repo.import_bioasq(cache_dir=cache_dir, force=force)

    async def list_suites(self) -> list[EvalSuite]:
        return await self._repo.list_suites(dataset_id=BIOASQ_DATASET_ID)

    async def get_suite(self, suite_id: str) -> EvalSuite | None:
        return await self._repo.get_suite(suite_id)

    async def create_suite(
        self,
        *,
        name: str,
        description: str | None = None,
        config: EvalSuiteConfig | None = None,
        sample_ids: list[int] | None = None,
    ) -> EvalSuite:
        return await self._repo.create_suite(
            name=name,
            description=description,
            config=config or EvalSuiteConfig(),
            sample_ids=sample_ids,
            dataset_id=BIOASQ_DATASET_ID,
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

    async def start_run(
        self,
        *,
        suite_id: str | None = None,
        name: str | None = None,
        config: EvalSuiteConfig | None = None,
        cache_dir: str | Path | None = None,
        use_db: bool = True,
    ) -> EvalRunSummary:
        suite: EvalSuite | None = None
        run_config = config or EvalSuiteConfig()

        if suite_id is not None:
            suite = await self._repo.get_suite(suite_id)
            if suite is None:
                raise ValueError(f"Suite no encontrada: {suite_id}")
            run_config = suite.config

        if run_config.ragas and not run_config.generate:
            raise ValueError("--ragas requiere generate=True")

        mode = "ragas" if run_config.ragas else ("generate" if run_config.generate else "retrieval")
        run = await self._repo.create_run(
            dataset_id=BIOASQ_DATASET_ID,
            suite_id=suite_id,
            name=name or (suite.name if suite else None),
            mode=mode,
            config=_config_to_dict(run_config),
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
        try:
            await self._repo.update_run_status(run_id, status="running")
            samples, corpus, sanitize_stats = await self._load_samples(
                suite=suite,
                config=config,
                cache_dir=cache_dir,
                use_db=use_db,
            )
            passages = self._build_passage_index(samples, corpus, config)
            pipeline = BioASQRagPipeline.from_env(
                self._env,
                top_k=config.top_k,
                with_llm=config.generate,
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
            if config.ragas:
                summary = evaluate_with_ragas(
                    results,
                    env=self._env,
                    output_dir=Path(f"evals/results/bioasq/runs/{run_id}"),
                    timeout_seconds=config.ragas_timeout,
                )
                ragas_metrics = summary.metrics

            await self._repo.save_run_results(run_id, results)
            await self._repo.update_run_status(
                run_id,
                status="completed",
                retrieval_metrics=retrieval_payload,
                ragas_metrics=ragas_metrics,
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
            await _shutdown_http_clients()

    async def _load_samples(
        self,
        *,
        suite: EvalSuite | None,
        config: EvalSuiteConfig,
        cache_dir: str | Path | None,
        use_db: bool,
    ) -> tuple[list[BioASQSample], dict[int, BioASQPassage], Any]:
        if suite is not None and suite.sample_ids:
            if use_db and await self._repo.is_dataset_imported():
                samples = await self._repo.load_bioasq_qa_from_db(sample_ids=list(suite.sample_ids))
                corpus = await self._repo.load_bioasq_corpus_from_db()
                from evals.bioasq import sanitize_samples_against_corpus

                status = await self._repo.get_dataset_status()
                skipped = int((status.import_stats if status else {}).get("skipped_nan_passages", 0))
                samples, stats = sanitize_samples_against_corpus(samples, set(corpus), skipped_nan_passages=skipped)
                return samples, corpus, stats

        if use_db and await self._repo.is_dataset_imported():
            return await self._repo.load_bioasq_eval_set(limit=config.limit)

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


def _config_to_dict(config: EvalSuiteConfig) -> dict[str, Any]:
    return asdict(config)


async def _shutdown_http_clients() -> None:
    try:
        import litellm
        from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients

        await close_litellm_async_clients()
        for attr in ("module_level_aclient", "aclient_session"):
            handler = getattr(litellm, attr, None)
            if handler is None:
                continue
            close = getattr(handler, "close", None)
            if callable(close):
                result = close()
                if hasattr(result, "__await__"):
                    await result
    except Exception:  # noqa: BLE001
        logger.debug("No se pudieron cerrar clientes LiteLLM", exc_info=True)
