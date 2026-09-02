"""Worker in-process para runs de evaluación encolados en Postgres."""

from __future__ import annotations

import asyncio
import logging

from chatbot.application.services.eval_service import EvalService
from chatbot.core.env import Env
from chatbot.infrastructure.adapters.persistence.postgres.eval_repository import (
    PostgresEvalRepository,
    _suite_config_from_dict,
)

logger = logging.getLogger(__name__)


class EvalJobWorker:
    def __init__(
        self,
        *,
        service: EvalService,
        repository: PostgresEvalRepository,
        env: Env,
    ) -> None:
        self._service = service
        self._repo = repository
        self._env = env
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        if not self._env.eval_worker_enabled:
            logger.info("Eval worker desactivado (EVAL_WORKER_ENABLED=false)")
            return
        await self._repo.requeue_expired_leases()
        poll = max(self._env.eval_worker_poll_seconds, 0.5)
        lease = self._env.eval_job_lease_seconds
        logger.info("Eval worker iniciado (poll=%.1fs, lease=%ss)", poll, lease)
        while not self._stop.is_set():
            try:
                claimed = await self._repo.claim_next_run(lease_seconds=lease)
            except Exception:  # noqa: BLE001
                logger.exception("Eval worker: error reclamando run")
                claimed = None
            if claimed is None:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=poll)
                except TimeoutError:
                    continue
                break
            suite = None
            if claimed.suite_id:
                suite = await self._repo.get_suite(claimed.suite_id)
            config = _suite_config_from_dict(dict(claimed.config or {}))
            use_db = bool(claimed.config.get("use_db", True)) if claimed.config else True
            cache_raw = claimed.config.get("cache_dir") if claimed.config else None
            try:
                await self._service.execute_run(
                    claimed.id,
                    suite=suite,
                    config=config,
                    cache_dir=cache_raw,
                    use_db=use_db,
                    lease_seconds=lease,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Eval worker: fallo ejecutando %s", claimed.id)

    def stop(self) -> None:
        self._stop.set()
