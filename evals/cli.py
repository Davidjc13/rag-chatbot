"""CLI: evaluación BioASQ (retrieval por defecto; generación/RAGAS opcionales)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

from chatbot.core.env import Env
from chatbot.infrastructure.config.logging_config import setup_logging

from evals.bioasq import (
    collect_relevant_ids,
    load_bioasq_eval_set,
)
from evals.json_dataset import dataset_template_path, parse_json_dataset
from evals.pipeline import BioASQRagPipeline
from evals.deepeval_runner import evaluate_with_deepeval
from evals.ragas_runner import evaluate_with_ragas, results_to_dicts
from evals.retrieval_metrics import compute_retrieval_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description=(
            "Evalúa retrieval (BioASQ o JSON). Opcional: --generate, --ragas, --deepeval."
        ),
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Usa evals/templates/dataset.template.json (smoke test rápido).",
    )
    parser.add_argument(
        "--json-dataset",
        type=Path,
        default=None,
        help="Ruta a un dataset JSON (mismo formato que la plantilla).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Número de preguntas QA a evaluar (default: 10).",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=50,
        help="Pasajes distractores aleatorios además de los relevantes (default: 50).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override de RAG_TOP_K para retrieval.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio de salida para métricas y artefactos.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("evals/data/hf-cache"),
        help="Caché local de Hugging Face datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para distractores aleatorios.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="También genera respuestas con el LLM (por defecto solo retrieval).",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help=(
            "Calcula métricas RAGAS con juez LLM (requiere --generate). "
            "Con qwen3:4b local suele fallar el parseo JSON."
        ),
    )
    parser.add_argument(
        "--deepeval",
        action="store_true",
        help="Calcula métricas DeepEval con juez LiteLLM/Ollama (requiere --generate).",
    )
    parser.add_argument(
        "--deepeval-timeout",
        type=int,
        default=600,
        help="Timeout (s) por job del juez DeepEval (default: 600).",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=600,
        help="Timeout (s) por job del juez RAGAS (default: 600).",
    )
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="Carga QA/corpus desde PostgreSQL (requiere importación previa).",
    )
    parser.add_argument(
        "--import-db",
        action="store_true",
        help="Importa el dataset BioASQ a PostgreSQL antes de evaluar.",
    )
    parser.add_argument(
        "--force-import",
        action="store_true",
        help="Reimporta el dataset aunque ya exista en PostgreSQL.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    env = Env.get_instance()
    setup_logging(level=args.log_level, json_logs=False)

    if args.template and args.json_dataset is not None:
        logging.error("Usa solo uno: --template o --json-dataset")
        return 2

    if args.ragas and not args.generate:
        logging.error("--ragas requiere --generate (hace falta la respuesta del modelo)")
        return 2

    if args.deepeval and not args.generate:
        logging.error("--deepeval requiere --generate (hace falta la respuesta del modelo)")
        return 2

    use_json = args.template or args.json_dataset is not None
    if use_json and (args.import_db or args.use_db or args.force_import):
        logging.error("--template/--json-dataset no usan Postgres; omite --use-db/--import-db")
        return 2

    output_dir = args.output_dir or (
        Path("evals/results/template") if use_json else Path("evals/results/bioasq")
    )
    args.output_dir = output_dir

    if use_json:
        json_path = args.json_dataset if args.json_dataset is not None else dataset_template_path()
        if not json_path.is_file():
            logging.error("Dataset JSON no encontrado: %s", json_path)
            return 1
        imported = parse_json_dataset(json_path)
        samples = imported.samples[: args.limit]
        corpus = imported.passages
        sanitize_stats = imported.sanitize_stats
        logging.info(
            "Dataset JSON %r: %s pasajes, %s muestras (evaluando %s)",
            imported.name,
            len(corpus),
            len(imported.samples),
            len(samples),
        )
    elif args.import_db or args.use_db or args.force_import:
        from chatbot.infrastructure.adapters.persistence.postgres.engine import (
            create_engine,
            create_session_factory,
        )
        from chatbot.infrastructure.adapters.persistence.postgres.eval_repository import (
            PostgresEvalRepository,
        )
        from chatbot.infrastructure.adapters.persistence.postgres.schema import init_schema

        engine = create_engine(env.database_url)
        session_factory = create_session_factory(engine)
        await init_schema(engine, embedding_dimension=env.embedding_dimension)
        repo = PostgresEvalRepository(session_factory)

        if args.import_db or args.force_import:
            await repo.import_bioasq(cache_dir=args.cache_dir, force=args.force_import)
            logging.info("Dataset BioASQ importado en PostgreSQL")

        if args.use_db or args.import_db:
            samples, corpus, sanitize_stats = await repo.load_bioasq_eval_set(
                limit=args.limit,
            )
        else:
            samples, corpus, sanitize_stats = load_bioasq_eval_set(
                limit=args.limit,
                cache_dir=args.cache_dir,
            )
        await engine.dispose()
    else:
        samples, corpus, sanitize_stats = load_bioasq_eval_set(
            limit=args.limit,
            cache_dir=args.cache_dir,
        )

    relevant_ids = collect_relevant_ids(samples)
    passages = {pid: corpus[pid] for pid in relevant_ids}

    if args.distractors > 0:
        candidates = [pid for pid in corpus if pid not in relevant_ids]
        rng = random.Random(args.seed)
        for pid in rng.sample(candidates, k=min(args.distractors, len(candidates))):
            passages[pid] = corpus[pid]

    missing = relevant_ids - set(passages)
    if missing:
        logging.error(
            "Inconsistencia interna: faltan %s pasajes gold tras sanitizar: %s",
            len(missing),
            sorted(missing)[:10],
        )
        return 1

    pipeline = BioASQRagPipeline.from_env(
        env,
        top_k=args.top_k,
        with_llm=args.generate,
    )

    indexed = await pipeline.index_passages(passages)
    logging.info("Índice listo: %s pasajes", indexed)

    results = await pipeline.run_samples(samples, generate=args.generate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rag_path = args.output_dir / "rag_outputs.json"
    rag_path.write_text(
        json.dumps(results_to_dicts(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Salida RAG/retrieval en %s", rag_path)

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
    retrieval_path = args.output_dir / "retrieval_metrics.json"
    retrieval_path.write_text(
        json.dumps(retrieval_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Métricas de retrieval: %s", retrieval_payload)

    payload: dict[str, object] = {
        "sample_count": len(results),
        "mode": _resolve_cli_mode(args),
        "retrieval_metrics": retrieval_payload,
        "rag_outputs": str(rag_path),
    }

    if args.ragas:
        summary = evaluate_with_ragas(
            results,
            env=env,
            output_dir=args.output_dir,
            timeout_seconds=args.ragas_timeout,
        )
        payload["ragas_metrics"] = summary.metrics
        payload["details"] = summary.details_path

    if args.deepeval:
        summary = evaluate_with_deepeval(
            results,
            env=env,
            output_dir=args.output_dir,
            timeout_seconds=args.deepeval_timeout,
        )
        payload["deepeval_metrics"] = summary.metrics
        payload["deepeval_details"] = summary.details_path

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _resolve_cli_mode(args: argparse.Namespace) -> str:
    if args.ragas and args.deepeval:
        return "full"
    if args.deepeval:
        return "deepeval"
    if args.ragas:
        return "ragas"
    if args.generate:
        return "generate"
    return "retrieval"


async def _shutdown_http_clients() -> None:
    """Cierra clientes HTTP de LiteLLM (httpx + transport aiohttp)."""
    try:
        import litellm
        from litellm.llms.custom_httpx.async_client_cleanup import (
            close_litellm_async_clients,
        )

        await close_litellm_async_clients()

        # close_litellm_async_clients no cierra siempre module_level_aclient,
        # que es el que deja el "Unclosed client session" de aiohttp.
        for attr in ("module_level_aclient", "aclient_session"):
            handler = getattr(litellm, attr, None)
            if handler is None:
                continue
            close = getattr(handler, "close", None)
            if callable(close):
                result = close()
                if hasattr(result, "__await__"):
                    await result
                continue
            client = getattr(handler, "client", None)
            if client is not None and hasattr(client, "aclose"):
                if not getattr(client, "is_closed", False):
                    await client.aclose()
    except Exception:  # noqa: BLE001
        logging.debug("No se pudieron cerrar clientes LiteLLM", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        logging.error("Cancelado por el usuario")
        return 130


async def _run(args: argparse.Namespace) -> int:
    try:
        return await _async_main(args)
    finally:
        await _shutdown_http_clients()


if __name__ == "__main__":
    sys.exit(main())
