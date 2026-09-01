"""Retrieval vectorial compartido por chat clásico y el agente ReAct."""

from __future__ import annotations

from chatbot.domain.documents import RetrievedChunk
from chatbot.domain.ports import EmbeddingPort, VectorStorePort


async def retrieve_chunks(  # pylint: disable=too-many-arguments
    query: str,
    *,
    embeddings: EmbeddingPort | None,
    vector_store: VectorStorePort | None,
    retrieval_backend: str | None = None,
    top_k: int,
    document_id: str | None = None,
) -> list[RetrievedChunk]:
    if embeddings is None or vector_store is None:
        return []
    vectors = await embeddings.embed([query])
    if not vectors:
        return []
    search_backend = getattr(vector_store, "search_backend", None)
    if callable(search_backend) and retrieval_backend:
        return await search_backend(
            retrieval_backend,
            vectors[0],
            top_k=top_k,
            document_id=document_id,
        )
    return await vector_store.search(
        vectors[0],
        top_k=top_k,
        document_id=document_id,
    )
