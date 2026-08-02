"""Ejecución de métricas DeepEval sobre resultados del pipeline (juez vía LiteLLM/Ollama)."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.pipeline import RagSampleResult

logger = logging.getLogger(__name__)

DEFAULT_METRICS = ("answer_relevancy", "faithfulness", "contextual_relevancy")


def configure_deepeval_runtime(timeout_seconds: int | None = None) -> Path:
    """Asegura rutas escribibles para caché/telemetría (Docker, CI, etc.)."""
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")
    os.environ.setdefault("DEEPEVAL_NO_INSPECT_PROMPT", "1")

    task_timeout = timeout_seconds
    if task_timeout is None:
        raw_override = os.getenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE")
        if raw_override:
            try:
                task_timeout = int(float(raw_override))
            except ValueError:
                logger.warning(
                    "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE inválido: %s",
                    raw_override,
                )
    if task_timeout is not None and task_timeout > 0:
        # DeepEval usa 180s por defecto; con Ollama local (qwen3) suele ser insuficiente.
        os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = str(task_timeout)

    raw = os.getenv("DEEPEVAL_CACHE_FOLDER")
    if raw:
        cache_dir = Path(os.path.expanduser(os.path.expandvars(raw)))
    else:
        cache_dir = Path(os.getenv("HOME", "/tmp")) / ".cache" / "deepeval"
        os.environ["DEEPEVAL_CACHE_FOLDER"] = str(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass(frozen=True, slots=True)
class DeepEvalSummary:
    sample_count: int
    metrics: dict[str, float | None]
    details_path: str | None = None


def _chat_model_name(env: Any) -> str:
    model = env.active_model
    return model.removeprefix("ollama/")


def _litellm_model_name(env: Any) -> str:
    model = env.active_model
    if not model.startswith("ollama/") and env.llm_provider == "litellm":
        return model
    if not model.startswith("ollama/"):
        return f"ollama/{model}"
    return model


def _litellm_api_base(env: Any) -> str:
    if getattr(env, "llm_provider", "") == "ollama":
        return env.ollama_base_url
    return env.litellm_api_base or env.ollama_base_url


def _judge_model_name(env: Any) -> str:
    override = os.getenv("DEEPEVAL_JUDGE_MODEL")
    if override:
        return override.removeprefix("ollama/")
    return _chat_model_name(env)


def _split_ollama_chat_kwargs(
    generation_kwargs: dict[str, Any],
    *,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separa kwargs de chat Ollama: `think` va al top-level, el resto en `options`."""
    gen = dict(generation_kwargs)
    top_level: dict[str, Any] = {}
    think = gen.pop("think", None)
    if think is not None:
        top_level["think"] = think
    options = {"temperature": temperature, **gen}
    return top_level, options


def _ollama_chat_response(
    chat_model: Any,
    *,
    model_name: str,
    messages: list[dict[str, Any]],
    schema: Any,
    generation_kwargs: dict[str, Any],
    temperature: float,
) -> Any:
    top_level, options = _split_ollama_chat_kwargs(
        generation_kwargs,
        temperature=temperature,
    )
    return chat_model.chat(
        model=model_name,
        messages=messages,
        format=schema.model_json_schema() if schema else None,
        options=options,
        **top_level,
    )


def create_deepeval_model(env: Any) -> Any:
    """Crea el juez DeepEval (Ollama nativo; mejor JSON estructurado que LiteLLM local)."""
    configure_deepeval_runtime()
    base_url = _litellm_api_base(env)
    model_name = _judge_model_name(env)

    from deepeval.models import OllamaModel
    from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array

    ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

    class FixedOllamaModel(OllamaModel):
        """OllamaModel con `think` en top-level (qwen3 devuelve content vacío si va en options)."""

        def generate(self, prompt: str, schema: Any = None) -> tuple[Any, float]:
            from deepeval.models.llms.ollama_model import retry_ollama

            @retry_ollama
            def _generate() -> tuple[Any, float]:
                chat_model = self.load_model()
                if check_if_multimodal(prompt):
                    multimodal = convert_to_multi_modal_array(prompt)
                    messages = self.generate_messages(multimodal)
                else:
                    messages = [{"role": "user", "content": prompt}]

                response = _ollama_chat_response(
                    chat_model,
                    model_name=self.name,
                    messages=messages,
                    schema=schema,
                    generation_kwargs=self.generation_kwargs,
                    temperature=self.temperature,
                )
                content = response.message.content
                if schema:
                    return schema.model_validate_json(content), 0
                return content, 0

            return _generate()

        async def a_generate(self, prompt: str, schema: Any = None) -> tuple[Any, float]:
            from deepeval.models.llms.ollama_model import retry_ollama

            @retry_ollama
            async def _a_generate() -> tuple[Any, float]:
                chat_model = self.load_model(async_mode=True)
                if check_if_multimodal(prompt):
                    multimodal = convert_to_multi_modal_array(prompt)
                    messages = self.generate_messages(multimodal)
                else:
                    messages = [{"role": "user", "content": prompt}]

                top_level, options = _split_ollama_chat_kwargs(
                    self.generation_kwargs,
                    temperature=self.temperature,
                )
                response = await chat_model.chat(
                    model=self.name,
                    messages=messages,
                    format=schema.model_json_schema() if schema else None,
                    options=options,
                    **top_level,
                )
                content = response.message.content
                if schema:
                    return schema.model_validate_json(content), 0
                return content, 0

            return await _a_generate()

    return FixedOllamaModel(
        model=model_name,
        base_url=base_url,
        temperature=0.0,
        timeout=ollama_timeout,
        generation_kwargs={"think": False, "num_predict": 2048},
    )


def _build_metrics(model: Any, metric_names: tuple[str, ...]) -> list[Any]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )

    registry = {
        "answer_relevancy": lambda: AnswerRelevancyMetric(model=model, threshold=0.5),
        "faithfulness": lambda: FaithfulnessMetric(model=model, threshold=0.5),
        "contextual_relevancy": lambda: ContextualRelevancyMetric(model=model, threshold=0.5),
    }
    metrics: list[Any] = []
    for name in metric_names:
        factory = registry.get(name)
        if factory is None:
            logger.warning("Métrica DeepEval desconocida, se omite: %s", name)
            continue
        metrics.append(factory())
    if not metrics:
        raise ValueError(f"No hay métricas DeepEval válidas en {metric_names!r}")
    return metrics


def build_deepeval_test_cases(results: list[RagSampleResult]) -> list[Any]:
    from deepeval.test_case import LLMTestCase

    cases: list[Any] = []
    for item in results:
        cases.append(
            LLMTestCase(
                input=item.question,
                actual_output=item.answer or "",
                expected_output=item.ground_truth,
                retrieval_context=list(item.contexts),
            )
        )
    return cases


def evaluate_with_deepeval(
    results: list[RagSampleResult],
    *,
    env: Any,
    output_dir: str | Path,
    timeout_seconds: int = 600,
    metric_names: tuple[str, ...] = DEFAULT_METRICS,
) -> DeepEvalSummary:
    """Calcula métricas DeepEval y persiste resultados."""
    cache_dir = configure_deepeval_runtime(timeout_seconds=timeout_seconds)
    logger.debug(
        "DeepEval cache en %s (per_task_timeout=%ss)",
        cache_dir,
        os.getenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "180"),
    )
    from deepeval import evaluate

    if not results:
        raise ValueError("No hay resultados para evaluar")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "sample_id": item.sample_id,
            "input": item.question,
            "actual_output": item.answer,
            "expected_output": item.ground_truth,
            "retrieval_context": list(item.contexts),
        }
        for item in results
    ]
    samples_path = out / "deepeval_samples.json"
    samples_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model = create_deepeval_model(env)
    metrics = _build_metrics(model, metric_names)
    test_cases = build_deepeval_test_cases(results)

    logger.info(
        "Ejecutando DeepEval sobre %s muestras (timeout=%ss)",
        len(results),
        timeout_seconds,
    )
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig

    evaluation = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False, max_concurrent=1),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
        ),
        error_config=ErrorConfig(ignore_errors=True),
    )

    metric_values = _extract_metric_means(evaluation, metrics)
    summary = {
        "sample_count": len(results),
        "metrics": metric_values,
        "metric_names": list(metric_names),
        "samples_file": str(samples_path),
        "timeout_seconds": timeout_seconds,
    }
    summary_path = out / "deepeval_metrics.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Métricas DeepEval guardadas en %s: %s", summary_path, metric_values)
    return DeepEvalSummary(
        sample_count=len(results),
        metrics=metric_values,
        details_path=str(summary_path),
    )


def _finite_or_none(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _normalize_metric_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _extract_metric_means(evaluation: Any, metrics: list[Any]) -> dict[str, Any]:
    """Extrae medias por métrica desde el resultado de deepeval.evaluate."""
    totals: dict[str, list[float]] = {}
    errors: list[str] = []

    if hasattr(evaluation, "test_results") and evaluation.test_results:
        for result in evaluation.test_results:
            for metric_data in getattr(result, "metrics_data", []) or []:
                raw_name = str(getattr(metric_data, "name", "") or "").strip()
                if not raw_name:
                    continue
                key = _normalize_metric_key(raw_name)
                score = getattr(metric_data, "score", None)
                if score is not None:
                    try:
                        totals.setdefault(key, []).append(float(score))
                    except (TypeError, ValueError):
                        pass
                    continue
                error = getattr(metric_data, "error", None)
                if error:
                    errors.append(f"{key}: {error}")

    if totals:
        payload: dict[str, Any] = {
            key: _finite_or_none(sum(values) / len(values))
            for key, values in totals.items()
        }
        if errors:
            payload["_warnings"] = errors
        return payload

    if errors:
        return {"_error": errors[0], "_warnings": errors}

    # Fallback: usar nombres de las métricas configuradas con score agregado del objeto.
    raw_scores: dict[str, Any] = {}
    for metric in metrics:
        name = _normalize_metric_key(str(getattr(metric, "name", metric.__class__.__name__)))
        score = getattr(metric, "score", None)
        if score is not None:
            try:
                raw_scores[name] = _finite_or_none(float(score))
            except (TypeError, ValueError):
                continue
    return raw_scores
