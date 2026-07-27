"""Adaptador de embeddings vía LiteLLM."""

from __future__ import annotations

import logging

from litellm import aembedding
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    Timeout,
)

from chatbot.domain.exceptions import LLMProviderError, LLMUnavailableError
from chatbot.domain.ports import EmbeddingPort

logger = logging.getLogger(__name__)

_PROVIDER = "litellm-embedding"


class LiteLLMEmbeddingAdapter(EmbeddingPort):
    def __init__(
        self,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._model = model
        self._api_base = api_base.rstrip("/") if api_base else None
        self._api_key = api_key or None
        self._timeout = timeout_seconds

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        kwargs: dict[str, object] = {
            "model": self._model,
            "input": texts,
            "timeout": self._timeout,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key

        try:
            response = await aembedding(**kwargs)
        except (APIConnectionError, Timeout) as exc:
            raise LLMUnavailableError(
                f"Embeddings LiteLLM no disponibles ({self._model}): {exc}",
                provider=_PROVIDER,
            ) from exc
        except AuthenticationError as exc:
            raise LLMProviderError(
                f"Autenticación fallida en embeddings LiteLLM: {exc}",
                provider=_PROVIDER,
            ) from exc
        except (RateLimitError, BadRequestError, APIError) as exc:
            raise LLMProviderError(str(exc), provider=_PROVIDER) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error inesperado en LiteLLMEmbeddingAdapter")
            raise LLMProviderError(str(exc), provider=_PROVIDER) from exc

        try:
            data = response["data"] if isinstance(response, dict) else response.data
            # Ordenar por index por si el proveedor reordena.
            indexed = sorted(
                data,
                key=lambda item: item["index"] if isinstance(item, dict) else item.index,
            )
            vectors: list[list[float]] = []
            for item in indexed:
                embedding = item["embedding"] if isinstance(item, dict) else item.embedding
                vectors.append([float(x) for x in embedding])
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise LLMProviderError(
                f"Respuesta inesperada de embeddings LiteLLM: {response!r}",
                provider=_PROVIDER,
            ) from exc

        if len(vectors) != len(texts):
            raise LLMProviderError(
                f"Se esperaban {len(texts)} embeddings y se recibieron {len(vectors)}",
                provider=_PROVIDER,
            )
        return vectors


class MockEmbeddingAdapter(EmbeddingPort):
    """Embeddings deterministas para tests (sin red)."""

    def __init__(self, *, model: str = "mock-embed", dimensions: int = 8) -> None:
        self._model = model
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) or 1
            base = [(seed * (i + 1) % 97) / 97.0 for i in range(self._dimensions)]
            norm = sum(x * x for x in base) ** 0.5 or 1.0
            vectors.append([x / norm for x in base])
        return vectors
