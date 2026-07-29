"""Ejecución de métricas RAGAS sobre resultados del pipeline BioASQ."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evals.pipeline import RagSampleResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RagasSummary:
    sample_count: int
    metrics: dict[str, float | None]
    details_path: str | None = None


def build_ragas_records(results: list[RagSampleResult]) -> list[dict[str, Any]]:
    """Convierte resultados del pipeline al esquema SingleTurn de RAGAS."""
    records: list[dict[str, Any]] = []
    for item in results:
        records.append(
            {
                "user_input": item.question,
                "response": item.answer,
                "retrieved_contexts": list(item.contexts),
                "reference": item.ground_truth,
                "sample_id": item.sample_id,
                "retrieved_passage_ids": list(item.retrieved_passage_ids),
                "retrieval_scores": list(item.scores),
            }
        )
    return records


def _ollama_base_url(env: Any) -> str:
    if getattr(env, "llm_provider", "") == "ollama":
        return env.ollama_base_url
    return env.litellm_api_base or env.ollama_base_url


def _chat_model_name(env: Any) -> str:
    model = env.active_model
    return model.removeprefix("ollama/")


def _embedding_model_name(env: Any) -> str:
    model = env.litellm_embedding_model
    return model.removeprefix("ollama/")


def create_ragas_judge(env: Any) -> tuple[Any, Any]:
    """Crea LLM y embeddings juez para RAGAS (Ollama vía LangChain)."""
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    base_url = _ollama_base_url(env)
    llm = LangchainLLMWrapper(
        ChatOllama(
            model=_chat_model_name(env),
            base_url=base_url,
            temperature=0.0,
            # Evita el modo "thinking" de Qwen3 (ralentiza y dispara timeouts).
            reasoning=False,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=_embedding_model_name(env),
            base_url=base_url,
        )
    )
    return llm, embeddings


def evaluate_with_ragas(
    results: list[RagSampleResult],
    *,
    env: Any,
    output_dir: str | Path,
    timeout_seconds: int = 600,
) -> RagasSummary:
    """Calcula métricas básicas de RAGAS y persiste resultados."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    if not results:
        raise ValueError("No hay resultados para evaluar")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = build_ragas_records(results)
    samples_path = out / "samples.json"
    samples_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
            }
            for r in records
        ]
    )
    llm, embeddings = create_ragas_judge(env)
    logger.info("Ejecutando RAGAS sobre %s muestras", len(results))
    evaluation = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=True,
        # Ollama local: pocas workers + timeout alto evita TimeoutError en el juez.
        run_config=RunConfig(timeout=timeout_seconds, max_workers=2, max_retries=2),
        batch_size=1,
    )

    metrics = _extract_metric_means(evaluation)
    summary = {
        "dataset": "rag-datasets/rag-mini-bioasq",
        "sample_count": len(results),
        "metrics": metrics,
        "samples_file": str(samples_path),
    }
    summary_path = out / "metrics.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Métricas RAGAS guardadas en %s: %s", summary_path, metrics)
    return RagasSummary(
        sample_count=len(results),
        metrics=metrics,
        details_path=str(summary_path),
    )


def _finite_or_none(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _extract_metric_means(evaluation: Any) -> dict[str, float | None]:
    raw: dict[str, Any]
    if hasattr(evaluation, "to_pandas"):
        frame = evaluation.to_pandas()
        numeric = frame.select_dtypes(include="number")
        raw = {col: float(numeric[col].mean()) for col in numeric.columns}
    elif isinstance(evaluation, dict):
        raw = evaluation
    else:
        raw = dict(evaluation)

    metrics: dict[str, float | None] = {}
    for key, value in raw.items():
        try:
            metrics[str(key)] = _finite_or_none(float(value))
        except (TypeError, ValueError):
            continue
    return metrics


def results_to_dicts(results: list[RagSampleResult]) -> list[dict[str, Any]]:
    return [asdict(item) for item in results]
