"""Adaptador LiteLLM — implementación unificada del puerto LLM."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from litellm import acompletion
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    Timeout,
)

from chatbot.domain.entities import Message, Role
from chatbot.domain.exceptions import LLMProviderError, LLMUnavailableError
from chatbot.domain.ports import LLMPort

logger = logging.getLogger(__name__)

_PROVIDER = "litellm"


class LiteLLMAdapter(LLMPort):
    """Cliente asíncrono vía LiteLLM (OpenAI, Ollama, Anthropic, etc.)."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        temperature: float = 0.7,
    ) -> None:
        self._model = model
        self._api_base = api_base.rstrip("/") if api_base else None
        self._api_key = api_key or None
        self._timeout = timeout_seconds
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def _build_messages(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None,
    ) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        if system_prompt:
            payload.append({"role": "system", "content": system_prompt})
        for message in messages:
            payload.append({"role": message.role.value, "content": message.content})
        return payload

    def _completion_kwargs(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None,
        stream: bool,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": self._model,
            "messages": self._build_messages(messages, system_prompt=system_prompt),
            "temperature": self._temperature,
            "timeout": self._timeout,
            "stream": stream,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return kwargs

    def _map_exception(self, exc: Exception) -> Exception:
        if isinstance(exc, (APIConnectionError, Timeout)):
            return LLMUnavailableError(
                f"No se pudo completar la petición a LiteLLM ({self._model}): {exc}",
                provider=_PROVIDER,
            )
        if isinstance(exc, AuthenticationError):
            return LLMProviderError(
                f"Autenticación fallida en LiteLLM: {exc}",
                provider=_PROVIDER,
            )
        if isinstance(exc, (RateLimitError, BadRequestError, APIError)):
            return LLMProviderError(str(exc), provider=_PROVIDER)
        return LLMProviderError(str(exc), provider=_PROVIDER)

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> Message:
        kwargs = self._completion_kwargs(messages, system_prompt=system_prompt, stream=False)

        logger.debug(
            "Llamando a LiteLLM",
            extra={"model": self._model, "api_base": self._api_base},
        )

        try:
            response = await acompletion(**kwargs)
        except (APIConnectionError, Timeout) as exc:
            logger.exception("LiteLLM no disponible (%s)", self._model)
            raise self._map_exception(exc) from exc
        except AuthenticationError as exc:
            logger.error("Autenticación LiteLLM fallida: %s", exc)
            raise self._map_exception(exc) from exc
        except (RateLimitError, BadRequestError, APIError) as exc:
            logger.error("Error de proveedor LiteLLM: %s", exc)
            raise self._map_exception(exc) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error inesperado en LiteLLMAdapter")
            raise self._map_exception(exc) from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise LLMProviderError(
                f"Respuesta inesperada de LiteLLM: {response!r}",
                provider=_PROVIDER,
            ) from exc

        if not content or not str(content).strip():
            raise LLMProviderError(
                "LiteLLM devolvió una respuesta vacía",
                provider=_PROVIDER,
            )

        return Message(role=Role.ASSISTANT, content=str(content).strip())

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        kwargs = self._completion_kwargs(messages, system_prompt=system_prompt, stream=True)

        logger.debug(
            "Llamando a LiteLLM (stream)",
            extra={"model": self._model, "api_base": self._api_base},
        )

        stream_errors = (
            APIConnectionError,
            Timeout,
            AuthenticationError,
            RateLimitError,
            BadRequestError,
            APIError,
        )
        try:
            response = await acompletion(**kwargs)
        except stream_errors as exc:
            logger.exception("Error iniciando stream LiteLLM (%s)", self._model)
            raise self._map_exception(exc) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error inesperado iniciando stream LiteLLM")
            raise self._map_exception(exc) from exc

        produced = False
        try:
            async for chunk in response:
                try:
                    delta = chunk.choices[0].delta.content
                except (AttributeError, IndexError, KeyError, TypeError):
                    continue
                if delta:
                    produced = True
                    yield str(delta)
        except stream_errors as exc:
            logger.exception("Error durante stream LiteLLM (%s)", self._model)
            raise self._map_exception(exc) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error inesperado durante stream LiteLLM")
            raise self._map_exception(exc) from exc

        if not produced:
            raise LLMProviderError(
                "LiteLLM devolvió un stream vacío",
                provider=_PROVIDER,
            )

    async def health_check(self) -> bool:
        kwargs: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "timeout": min(self._timeout, 10.0),
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key

        try:
            await acompletion(**kwargs)
            return True
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            logger.warning("Health check de LiteLLM falló", exc_info=True)
            return False
