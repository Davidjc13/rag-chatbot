"""Tests para importación JSON y runner DeepEval."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from evals.deepeval_runner import (
    build_deepeval_test_cases,
    configure_deepeval_runtime,
    create_deepeval_model,
    _extract_metric_means,
    _split_ollama_chat_kwargs,
)
from deepeval.models import OllamaModel
from evals.domain import EvalSuiteConfig
from evals.json_dataset import parse_json_dataset, slugify_dataset_id
from evals.pipeline import RagSampleResult
from chatbot.infrastructure.adapters.persistence.postgres.eval_repository import (
    _suite_config_from_dict,
    _suite_config_to_dict,
)


SAMPLE_JSON = {
    "name": "Test dataset",
    "passages": [
        {"id": "a", "text": "Texto A"},
        {"id": "b", "text": "Texto B"},
    ],
    "samples": [
        {
            "id": "q1",
            "question": "Pregunta?",
            "answer": "Respuesta",
            "relevant_passage_ids": ["a"],
        }
    ],
}


def test_slugify_dataset_id() -> None:
    assert slugify_dataset_id("Mi Dataset 2024") == "mi-dataset-2024"


def test_parse_json_dataset_maps_ids() -> None:
    imported = parse_json_dataset(SAMPLE_JSON, dataset_id="custom-set")
    assert imported.dataset_id == "custom-set"
    assert len(imported.passages) == 2
    assert len(imported.samples) == 1
    assert imported.samples[0].relevant_passage_ids == (1,)


def test_suite_config_deepeval_roundtrip() -> None:
    config = EvalSuiteConfig(generate=True, deepeval=True, llm_model="ollama/qwen2.5:3b")
    restored = _suite_config_from_dict(_suite_config_to_dict(config))
    assert restored.deepeval is True
    assert restored.llm_model == "ollama/qwen2.5:3b"


def test_build_deepeval_test_cases() -> None:
    results = [
        RagSampleResult(
            sample_id=1,
            question="Q",
            ground_truth="GT",
            answer="A",
            contexts=("ctx",),
            retrieved_passage_ids=(1,),
            scores=(0.9,),
        )
    ]
    cases = build_deepeval_test_cases(results)
    assert len(cases) == 1
    assert cases[0].input == "Q"
    assert cases[0].actual_output == "A"


def test_split_ollama_chat_kwargs_moves_think_to_top_level() -> None:
    top_level, options = _split_ollama_chat_kwargs(
        {"think": False, "num_predict": 2048},
        temperature=0.0,
    )
    assert top_level == {"think": False}
    assert options == {"temperature": 0.0, "num_predict": 2048}


def test_create_deepeval_model_uses_fixed_ollama_subclass() -> None:
    env = MagicMock()
    env.llm_provider = "litellm"
    env.active_model = "qwen3:4b"
    env.litellm_api_base = "http://localhost:11434"
    env.ollama_base_url = "http://localhost:11434"

    with patch.dict("os.environ", {}, clear=False):
        model = create_deepeval_model(env)
    assert isinstance(model, OllamaModel)
    assert model.name == "qwen3:4b"
    assert model.base_url == "http://localhost:11434"
    assert model.generation_kwargs == {"think": False, "num_predict": 2048}


def test_extract_metric_means_normalizes_keys() -> None:
    metric_data = MagicMock()
    metric_data.name = "Answer Relevancy"
    metric_data.score = 0.85
    metric_data.error = None
    result = MagicMock()
    result.metrics_data = [metric_data]
    evaluation = MagicMock()
    evaluation.test_results = [result]

    extracted = _extract_metric_means(evaluation, [])
    assert extracted["answer_relevancy"] == 0.85


def test_configure_deepeval_runtime_uses_writable_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPEVAL_CACHE_FOLDER", raising=False)
    monkeypatch.delenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = configure_deepeval_runtime()
    assert cache.is_dir()
    assert cache == tmp_path / ".cache" / "deepeval"


def test_configure_deepeval_runtime_sets_task_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPEVAL_CACHE_FOLDER", raising=False)
    monkeypatch.delenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    configure_deepeval_runtime(timeout_seconds=900)
    assert os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] == "900"
