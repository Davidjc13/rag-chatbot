"""Adaptador Ollama — implementación del puerto LLM."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from chatbot.core.http import AsyncHttpClient
from chatbot.domain.entities import Message, Role
from chatbot.domain.exceptions import LLMProviderError, LLMUnavailableError
from chatbot.domain.llm_stream import LLMDelta
from chatbot.domain.ports import LLMPort

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMPort):
    """Cliente HTTP hacia la API de chat de Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 300.0,
        temperature: float = 0.7,
        think: bool = True,
        num_predict: int = 4096,
        think_max_chars: int = 600,
        http: AsyncHttpClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._think = think
        self._num_predict = num_predict
        self._think_max_chars = think_max_chars
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
            "think": self._think,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._num_predict,
            },
        }

    def _map_http_error(self, exc: Exception, *, stream: bool = False) -> Exception:
        suffix = " (stream)" if stream else ""
        if isinstance(exc, httpx.ConnectError):
            logger.exception("Ollama no disponible en %s", self._base_url)
            return LLMUnavailableError(
                f"No se pudo conectar con Ollama en {self._base_url}",
                provider="ollama",
            )
        if isinstance(exc, httpx.TimeoutException):
            logger.exception("Timeout al consultar Ollama%s", suffix)
            return LLMUnavailableError(
                f"Timeout tras {self._timeout}s consultando Ollama",
                provider="ollama",
            )
        if isinstance(exc, httpx.HTTPStatusError):
            detail = exc.response.text
            logger.error("Error HTTP de Ollama%s: %s", suffix, detail)
            return LLMProviderError(
                f"Ollama respondió con error HTTP {exc.response.status_code}: {detail}",
                provider="ollama",
            )
        logger.exception("Error inesperado en OllamaAdapter%s", suffix)
        return LLMProviderError(str(exc), provider="ollama")

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> Message:
        body = self._build_payload(messages, system_prompt=system_prompt, stream=False)
        url = f"{self._base_url}/api/chat"
        try:
            response = await self._http.post(url, json=body, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"]
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            raise self._map_http_error(exc) from exc
        except (KeyError, TypeError) as exc:
            raise LLMProviderError(
                f"Respuesta inesperada de Ollama: {data!r}",
                provider="ollama",
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._map_http_error(exc) from exc

        return Message(role=Role.ASSISTANT, content=str(content).strip())

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[LLMDelta]:
        body = self._build_payload(messages, system_prompt=system_prompt, stream=True)
        url = f"{self._base_url}/api/chat"
        produced_content = False
        thinking_chars = 0
        thinking_truncated = False

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

                    message = data.get("message") or {}
                    thinking = message.get("thinking")
                    if thinking and self._think and not thinking_truncated:
                        piece = str(thinking)
                        remaining = self._think_max_chars - thinking_chars
                        if remaining <= 0:
                            thinking_truncated = True
                        else:
                            if len(piece) > remaining:
                                piece = piece[:remaining] + "…"
                                thinking_truncated = True
                            thinking_chars += len(piece)
                            yield LLMDelta(text=piece, kind="thinking")

                    content = message.get("content")
                    if content:
                        produced_content = True
                        yield LLMDelta(text=str(content), kind="content")
        except LLMProviderError:
            raise
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            raise self._map_http_error(exc, stream=True) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._map_http_error(exc, stream=True) from exc

        if not produced_content:
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
