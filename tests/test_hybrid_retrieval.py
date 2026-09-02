"""Tests de retrieval híbrido (RRF + keyword en memoria)."""

from __future__ import annotations

import pytest

from chatbot.application.services.rag_retrieval import retrieve_chunks
from chatbot.domain.documents import DocumentChunk
from chatbot.infrastructure.adapters.llm.embedding_adapter import MockEmbeddingAdapter
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_hybrid_retrieval_prefers_lexical_overlap() -> None:
    embeddings = MockEmbeddingAdapter()
    store = InMemoryVectorStore()
    dense_vec = await embeddings.embed(["vectorial"])
    keyword_vec = await embeddings.embed(["otro tema"])
    await store.upsert(
        [
            DocumentChunk(
                id="dense",
                document_id="doc-1",
                content="texto genérico sin la palabra clave buscada",
                metadata={"filename": "a.txt", "format": "txt"},
                embedding=dense_vec[0],
            ),
            DocumentChunk(
                id="lexical",
                document_id="doc-1",
                content="plazo de devolución devolución devolución",
                metadata={"filename": "b.txt", "format": "txt"},
                embedding=keyword_vec[0],
            ),
        ]
    )

    hybrid = await retrieve_chunks(
        "devolución",
        embeddings=embeddings,
        vector_store=store,
        top_k=1,
        hybrid=True,
        candidates=4,
        hybrid_alpha=0.2,
    )
    assert hybrid
    assert hybrid[0].chunk.id == "lexical"

    dense_only = await retrieve_chunks(
        "devolución",
        embeddings=embeddings,
        vector_store=store,
        top_k=2,
        hybrid=False,
    )
    assert len(dense_only) == 2
