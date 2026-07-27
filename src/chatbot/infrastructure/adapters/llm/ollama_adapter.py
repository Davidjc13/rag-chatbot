"""Adaptador Ollama — implementación del puerto LLM."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from chatbot.core.http import AsyncHttpClient
from chatbot.domain.entities import Message, Role
from chatbot.domain.exceptions import LLMProviderError, LLMUnavailableError
from chatbot.domain.ports import LLMPort

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMPort):
    """Cliente HTTP hacia la API de chat de Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        temperature: float = 0.7,
        http: AsyncHttpClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._http = http or AsyncHttpClient.get_instance()

    @property
    def model_name(self) -> str:
        return self._model

    def _build_payload(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None,
        stream: bool,
    ) -> dict[str, object]:
        payload_messages: list[dict[str, str]] = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        for message in messages:
            payload_messages.append({"role": message.role.value, "content": message.content})
        return {
            "model": self._model,
            "messages": payload_messages,
            "stream": stream,
            "options": {"temperature": self._temperature},
        }

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> Message:
        body = self._build_payload(messages, system_prompt=system_prompt, stream=False)
        url = f"{self._base_url}/api/chat"
        logger.debug("Llamando a Ollama", extra={"url": url, "model": self._model})

        try:
            response = await self._http.post(url, json=body, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError as exc:
            logger.exception("Ollama no disponible en %s", self._base_url)
            raise LLMUnavailableError(
                f"No se pudo conectar con Ollama en {self._base_url}",
                provider="ollama",
            ) from exc
        except httpx.TimeoutException as exc:
            logger.exception("Timeout al consultar Ollama")
            raise LLMUnavailableError(
                f"Timeout tras {self._timeout}s consultando Ollama",
                provider="ollama",
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            logger.error("Error HTTP de Ollama: %s", detail)
            raise LLMProviderError(
                f"Ollama respondió con error HTTP {exc.response.status_code}: {detail}",
                provider="ollama",
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error inesperado en OllamaAdapter")
            raise LLMProviderError(str(exc), provider="ollama") from exc

        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMProviderError(
                f"Respuesta inesperada de Ollama: {data!r}",
                provider="ollama",
            ) from exc

        return Message(role=Role.ASSISTANT, content=content.strip())

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        body = self._build_payload(messages, system_prompt=system_prompt, stream=True)
        url = f"{self._base_url}/api/chat"
        logger.debug("Llamando a Ollama (stream)", extra={"url": url, "model": self._model})

        produced = False
        try:
            async with self._http.stream(
                "POST",
                url,
                json=body,
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMProviderError(
                            f"Chunk NDJSON inválido de Ollama: {line!r}",
                            provider="ollama",
                        ) from exc
                    content = (data.get("message") or {}).get("content")
                    if content:
                        produced = True
                        yield str(content)
        except httpx.ConnectError as exc:
            logger.exception("Ollama no disponible en %s", self._base_url)
            raise LLMUnavailableError(
                f"No se pudo conectar con Ollama en {self._base_url}",
                provider="ollama",
            ) from exc
        except httpx.TimeoutException as exc:
            logger.exception("Timeout al consultar Ollama (stream)")
            raise LLMUnavailableError(
                f"Timeout tras {self._timeout}s consultando Ollama",
                provider="ollama",
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            logger.error("Error HTTP de Ollama (stream): %s", detail)
            raise LLMProviderError(
                f"Ollama respondió con error HTTP {exc.response.status_code}: {detail}",
                provider="ollama",
            ) from exc
        except LLMProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error inesperado en OllamaAdapter.stream")
            raise LLMProviderError(str(exc), provider="ollama") from exc

        if not produced:
            raise LLMProviderError(
                "Ollama devolvió un stream vacío",
                provider="ollama",
            )

    async def health_check(self) -> bool:
        try:
            response = await self._http.get(
                f"{self._base_url}/api/tags",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            logger.warning("Health check de Ollama falló", exc_info=True)
            return False
