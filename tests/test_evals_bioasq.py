"""Tests del flujo de evaluación BioASQ + RAGAS (sin LLM real)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.bioasq import BioASQPassage, BioASQSample, parse_passage_ids
from evals.pipeline import BioASQRagPipeline, RagSampleResult
from evals.ragas_runner import build_ragas_records, results_to_dicts


class _FakeEmbeddings:
    model_name = "fake-embed"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            # Embedding trivial: longitud normalizada + hash de caracteres.
            base = float(len(text) % 97) / 97.0
            vectors.append([base, 1.0 - base, float(sum(ord(c) for c in text[:20]) % 50) / 50.0])
        return vectors


class _FakeLLM:
    model_name = "fake-llm"

    async def generate(self, messages, *, system_prompt=None):
        from chatbot.domain.entities import Message, Role

        q = messages[-1].content if messages else ""
        return Message(role=Role.ASSISTANT, content=f"Respuesta sintética a: {q}")

    async def generate_stream(self, messages, *, system_prompt=None):
        if False:  # pragma: no cover
            yield None

    async def health_check(self) -> bool:
        return True


def test_parse_passage_ids_string_list() -> None:
    assert parse_passage_ids("[1, 2, 3]") == (1, 2, 3)


def test_parse_passage_ids_json() -> None:
    assert parse_passage_ids(json.dumps([10, 20])) == (10, 20)


def test_parse_passage_ids_native_list() -> None:
    assert parse_passage_ids([7, 8]) == (7, 8)


def test_is_usable_passage_text_filters_nan() -> None:
    from evals.bioasq import is_usable_passage_text

    assert is_usable_passage_text("hello") is True
    assert is_usable_passage_text(None) is False
    assert is_usable_passage_text(float("nan")) is False
    assert is_usable_passage_text("nan") is False
    assert is_usable_passage_text("  NaN  ") is False
    assert is_usable_passage_text("") is False


def test_sanitize_samples_drops_nan_refs() -> None:
    from evals.bioasq import sanitize_samples_against_corpus

    samples = [
        BioASQSample(
            id=1,
            question="q1",
            ground_truth="a1",
            relevant_passage_ids=(10, 99, 20),  # 99 missing/nan
        ),
        BioASQSample(
            id=2,
            question="q2",
            ground_truth="a2",
            relevant_passage_ids=(55,),  # all missing → drop sample
        ),
    ]
    cleaned, stats = sanitize_samples_against_corpus(samples, {10, 20})
    assert len(cleaned) == 1
    assert cleaned[0].relevant_passage_ids == (10, 20)
    assert stats.dropped_samples == 1
    assert stats.dropped_passage_refs == 2  # 99 and 55


@pytest.mark.asyncio
async def test_pipeline_index_and_run(tmp_path: Path) -> None:
    from chatbot.infrastructure.adapters.persistence.memory_vector_store import (
        InMemoryVectorStore,
    )

    passages = {
        101: BioASQPassage(id=101, text="EGFR ligands include EGF and TGF-alpha."),
        202: BioASQPassage(id=202, text="Unrelated text about orthopedics."),
    }
    sample = BioASQSample(
        id=1,
        question="Which ligands interact with EGFR?",
        ground_truth="EGF and TGF-alpha",
        relevant_passage_ids=(101,),
    )
    pipeline = BioASQRagPipeline(
        llm=_FakeLLM(),
        embeddings=_FakeEmbeddings(),
        vector_store=InMemoryVectorStore(),
        top_k=2,
    )
    indexed = await pipeline.index_passages(passages)
    assert indexed == 2

    retrieved_only = await pipeline.retrieve_sample(sample)
    assert retrieved_only.answer == ""
    assert 101 in retrieved_only.retrieved_passage_ids

    result = await pipeline.run_sample(sample)
    assert result.sample_id == 1
    assert result.answer.startswith("Respuesta sintética")
    assert len(result.contexts) == 2
    assert 101 in result.retrieved_passage_ids


def test_build_ragas_records() -> None:
    results = [
        RagSampleResult(
            sample_id=0,
            question="Q?",
            ground_truth="A",
            answer="B",
            contexts=("ctx",),
            retrieved_passage_ids=(1,),
            scores=(0.9,),
        )
    ]
    records = build_ragas_records(results)
    assert records[0]["user_input"] == "Q?"
    assert records[0]["response"] == "B"
    assert records[0]["retrieved_contexts"] == ["ctx"]
    assert records[0]["reference"] == "A"
    assert results_to_dicts(results)[0]["sample_id"] == 0


def test_retrieval_metrics_hit_and_mrr() -> None:
    from evals.retrieval_metrics import compute_retrieval_metrics

    samples = [
        BioASQSample(
            id=1,
            question="q1",
            ground_truth="a1",
            relevant_passage_ids=(10, 20),
        ),
        BioASQSample(
            id=2,
            question="q2",
            ground_truth="a2",
            relevant_passage_ids=(30,),
        ),
    ]
    results = [
        RagSampleResult(
            sample_id=1,
            question="q1",
            ground_truth="a1",
            answer="x",
            contexts=("c",),
            retrieved_passage_ids=(99, 10),
            scores=(0.9, 0.8),
        ),
        RagSampleResult(
            sample_id=2,
            question="q2",
            ground_truth="a2",
            answer="y",
            contexts=("c",),
            retrieved_passage_ids=(40, 50),
            scores=(0.7, 0.6),
        ),
    ]
    metrics = compute_retrieval_metrics(samples, results)
    assert metrics.sample_count == 2
    assert metrics.hit_at_k == 0.5
    assert metrics.recall_at_k == 0.25  # 1/2 and 0/1
    assert metrics.mrr == 0.25  # 1/2 and 0
