"""Retrieval vectorial compartido por chat clásico y el agente ReAct."""

from __future__ import annotations

from chatbot.domain.documents import RetrievedChunk
from chatbot.domain.ports import EmbeddingPort, VectorStorePort
from chatbot.domain.retrieval import RETRIEVAL_BACKEND_POSTGRES


def _rrf_scores(ranked_lists: list[list[RetrievedChunk]], *, k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            chunk_id = item.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    peak = max(values.values())
    if peak <= 0:
        return {key: 0.0 for key in values}
    return {key: value / peak for key, value in values.items()}


def _fuse_hybrid(  # pylint: disable=too-many-locals
    dense: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    *,
    top_k: int,
    alpha: float,
) -> list[RetrievedChunk]:
    by_id: dict[str, RetrievedChunk] = {}
    dense_scores = {item.chunk.id: item.score for item in dense}
    lexical_scores = {item.chunk.id: item.score for item in lexical}
    for item in dense + lexical:
        by_id[item.chunk.id] = item

    rrf = _rrf_scores([dense, lexical] if lexical else [dense])
    dense_n = _normalize(dense_scores)
    lexical_n = _normalize(lexical_scores)
    clamped = min(max(alpha, 0.0), 1.0)

    fused: list[RetrievedChunk] = []
    for chunk_id, chunk in by_id.items():
        combined = (
            clamped * dense_n.get(chunk_id, 0.0)
            + (1.0 - clamped) * lexical_n.get(chunk_id, 0.0)
            + 0.01 * rrf.get(chunk_id, 0.0)
        )
        fused.append(RetrievedChunk(chunk=chunk.chunk, score=combined))
    fused.sort(key=lambda item: item.score, reverse=True)
    return fused[:top_k]


async def retrieve_chunks(  # pylint: disable=too-many-arguments,too-many-locals
    query: str,
    *,
    embeddings: EmbeddingPort | None,
    vector_store: VectorStorePort | None,
    retrieval_backend: str | None = None,
    top_k: int,
    document_id: str | None = None,
    hybrid: bool = False,
    candidates: int = 20,
    hybrid_alpha: float = 0.7,
) -> list[RetrievedChunk]:
    if embeddings is None or vector_store is None:
        return []
    vectors = await embeddings.embed([query])
    if not vectors:
        return []

    pool = max(top_k, candidates if hybrid else top_k)
    search_backend = getattr(vector_store, "search_backend", None)
    if callable(search_backend) and retrieval_backend:
        dense = await search_backend(
            retrieval_backend,
            vectors[0],
            top_k=pool,
            document_id=document_id,
        )
    else:
        dense = await vector_store.search(
            vectors[0],
            top_k=pool,
            document_id=document_id,
        )

    backend = (retrieval_backend or RETRIEVAL_BACKEND_POSTGRES).lower()
    use_hybrid = hybrid and backend == RETRIEVAL_BACKEND_POSTGRES
    if not use_hybrid:
        return dense[:top_k]

    lexical = await vector_store.keyword_search(
        query,
        top_k=pool,
        document_id=document_id,
    )
    if not lexical:
        return dense[:top_k]
    return _fuse_hybrid(dense, lexical, top_k=top_k, alpha=hybrid_alpha)
