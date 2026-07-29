"""Pipeline RAG aislado para evaluación (índice en memoria + retrieve + generate)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chatbot.core.env import Env
from chatbot.domain.documents import DocumentChunk, DocumentFormat, RetrievedChunk
from chatbot.domain.entities import Message, Role
from chatbot.domain.ports import EmbeddingPort, LLMPort, VectorStorePort
from chatbot.infrastructure.adapters.llm.embedding_adapter import LiteLLMEmbeddingAdapter
from chatbot.infrastructure.adapters.llm.llm_factory import LLMFactory
from chatbot.infrastructure.adapters.persistence.memory_vector_store import InMemoryVectorStore

from evals.bioasq import BioASQPassage, BioASQSample

logger = logging.getLogger(__name__)

_EVAL_SYSTEM = (
    "Eres un asistente biomédico. Responde de forma concisa usando únicamente "
    "el contexto recuperado. Si el contexto no basta, dilo explícitamente.\n\n"
    "Contexto:\n{context}"
)


def _embedding_api_base(env: Env) -> str | None:
    if env.llm_provider == "ollama":
        return env.ollama_base_url
    return env.litellm_api_base


@dataclass(frozen=True, slots=True)
class RagSampleResult:
    sample_id: int
    question: str
    ground_truth: str
    answer: str
    contexts: tuple[str, ...]
    retrieved_passage_ids: tuple[int, ...]
    scores: tuple[float, ...]


class BioASQRagPipeline:
    """Indexa pasajes BioASQ y ejecuta retrieve (+ generate opcional) sobre muestras QA."""

    def __init__(
        self,
        *,
        llm: LLMPort | None,
        embeddings: EmbeddingPort,
        vector_store: VectorStorePort,
        top_k: int = 4,
        embed_batch_size: int = 32,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._vector_store = vector_store
        self.top_k = top_k
        self._embed_batch_size = embed_batch_size

    @classmethod
    def from_env(
        cls,
        env: Env | None = None,
        *,
        top_k: int | None = None,
        with_llm: bool = False,
    ) -> BioASQRagPipeline:
        settings = env or Env.get_instance()
        llm = LLMFactory.create(settings) if with_llm else None
        embeddings = LiteLLMEmbeddingAdapter(
            model=settings.litellm_embedding_model,
            api_base=_embedding_api_base(settings),
            api_key=settings.litellm_api_key,
            timeout_seconds=settings.litellm_embedding_timeout_seconds,
        )
        return cls(
            llm=llm,
            embeddings=embeddings,
            vector_store=InMemoryVectorStore(),
            top_k=top_k if top_k is not None else settings.rag_top_k,
        )

    async def index_passages(self, passages: dict[int, BioASQPassage]) -> int:
        if not passages:
            raise ValueError("No hay pasajes para indexar")

        items = list(passages.values())
        total = 0
        batch_size = max(1, self._embed_batch_size)
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            vectors = await self._embeddings.embed([p.text for p in batch])
            chunks: list[DocumentChunk] = []
            for passage, vector in zip(batch, vectors, strict=True):
                chunks.append(
                    DocumentChunk(
                        document_id=str(passage.id),
                        content=passage.text,
                        embedding=vector,
                        metadata={
                            "filename": f"passage_{passage.id}.txt",
                            "format": DocumentFormat.PDF.value,
                            "passage_id": passage.id,
                            "source": "rag-mini-bioasq",
                        },
                    )
                )
            await self._vector_store.upsert(chunks)
            total += len(chunks)
            logger.info("Indexados %s/%s pasajes", total, len(items))
        return total

    async def retrieve_sample(self, sample: BioASQSample) -> RagSampleResult:
        """Solo retrieval: no llama al LLM de chat."""
        vectors = await self._embeddings.embed([sample.question])
        retrieved = await self._vector_store.search(vectors[0], top_k=self.top_k)
        return self._to_result(sample, retrieved, answer="")

    async def run_sample(self, sample: BioASQSample) -> RagSampleResult:
        """Retrieval + generación de respuesta."""
        vectors = await self._embeddings.embed([sample.question])
        retrieved = await self._vector_store.search(vectors[0], top_k=self.top_k)
        answer = await self._generate(sample.question, retrieved)
        return self._to_result(sample, retrieved, answer=answer)

    async def run_samples(
        self,
        samples: list[BioASQSample],
        *,
        generate: bool = False,
    ) -> list[RagSampleResult]:
        results: list[RagSampleResult] = []
        for index, sample in enumerate(samples, start=1):
            mode = "RAG" if generate else "retrieve"
            logger.info("%s %s/%s (id=%s)", mode, index, len(samples), sample.id)
            if generate:
                results.append(await self.run_sample(sample))
            else:
                results.append(await self.retrieve_sample(sample))
        return results

    @staticmethod
    def _to_result(
        sample: BioASQSample,
        retrieved: list[RetrievedChunk],
        *,
        answer: str,
    ) -> RagSampleResult:
        return RagSampleResult(
            sample_id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            answer=answer,
            contexts=tuple(item.chunk.content for item in retrieved),
            retrieved_passage_ids=tuple(
                int(item.chunk.metadata.get("passage_id", item.chunk.document_id))
                for item in retrieved
            ),
            scores=tuple(item.score for item in retrieved),
        )

    async def _generate(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        if self._llm is None:
            raise RuntimeError("LLM no configurado; usa with_llm=True o --generate")
        if retrieved:
            context = "\n\n".join(
                f"[{i}] {item.chunk.content}" for i, item in enumerate(retrieved, start=1)
            )
        else:
            context = "(Sin fragmentos relevantes recuperados.)"
        system_prompt = _EVAL_SYSTEM.format(context=context)
        message = await self._llm.generate(
            [Message(role=Role.USER, content=question)],
            system_prompt=system_prompt,
        )
        return message.content.strip()
