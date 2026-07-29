"""Métricas de retrieval deterministas (sin juez LLM)."""

from __future__ import annotations

from dataclasses import dataclass

from evals.bioasq import BioASQSample
from evals.pipeline import RagSampleResult


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    hit_at_k: float
    recall_at_k: float
    mrr: float
    sample_count: int


def compute_retrieval_metrics(
    samples: list[BioASQSample],
    results: list[RagSampleResult],
) -> RetrievalMetrics:
    """
    Compara pasajes recuperados con `relevant_passage_ids` del dataset.

    - hit_at_k: fracción de queries con ≥1 pasaje relevante en top-k
    - recall_at_k: media de |retrieved ∩ relevant| / |relevant|
    - mrr: mean reciprocal rank del primer pasaje relevante
    """
    by_id = {sample.id: sample for sample in samples}
    hits = 0
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    for result in results:
        sample = by_id.get(result.sample_id)
        if sample is None:
            continue
        relevant = set(sample.relevant_passage_ids)
        if not relevant:
            continue
        retrieved = list(result.retrieved_passage_ids)
        overlap = relevant.intersection(retrieved)
        if overlap:
            hits += 1
        recalls.append(len(overlap) / len(relevant))
        rr = 0.0
        for rank, pid in enumerate(retrieved, start=1):
            if pid in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    n = len(recalls)
    if n == 0:
        return RetrievalMetrics(hit_at_k=0.0, recall_at_k=0.0, mrr=0.0, sample_count=0)
    return RetrievalMetrics(
        hit_at_k=hits / n,
        recall_at_k=sum(recalls) / n,
        mrr=sum(reciprocal_ranks) / n,
        sample_count=n,
    )
