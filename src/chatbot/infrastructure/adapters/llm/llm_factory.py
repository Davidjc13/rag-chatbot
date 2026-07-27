"""Factory de adaptadores LLM (patrón Factory Method)."""

from __future__ import annotations

import logging
from typing import Literal

from chatbot.core.env import Env
from chatbot.domain.exceptions import ConfigurationError
from chatbot.domain.ports import LLMPort
from chatbot.infrastructure.adapters.llm.litellm_adapter import LiteLLMAdapter
from chatbot.infrastructure.adapters.llm.mock_adapter import MockLLMAdapter
from chatbot.infrastructure.adapters.llm.ollama_adapter import OllamaAdapter

logger = logging.getLogger(__name__)

LLMProviderName = Literal["litellm", "ollama", "mock"]


class LLMFactory:
    """
    Crea la implementación concreta de LLMPort según configuración.

    Facilita cambiar de modelo/proveedor (LiteLLM, Ollama, mock…)
    sin tocar la capa de aplicación.
    """

    @staticmethod
    def create(env: Env | None = None) -> LLMPort:
        settings = env or Env.get_instance()
        provider = settings.llm_provider
        model = settings.active_model
        logger.info(
            "Creando adaptador LLM",
            extra={"provider": provider, "model": model},
        )

        if provider == "litellm":
            return LiteLLMAdapter(
                model=settings.litellm_model,
                api_base=settings.litellm_api_base,
                api_key=settings.litellm_api_key,
                timeout_seconds=settings.litellm_timeout_seconds,
                temperature=settings.litellm_temperature,
            )

        if provider == "ollama":
            return OllamaAdapter(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                timeout_seconds=settings.ollama_timeout_seconds,
                temperature=settings.ollama_temperature,
                think=settings.ollama_think,
                num_predict=settings.ollama_num_predict,
                think_max_chars=settings.ollama_think_max_chars,
            )

        if provider == "mock":
            return MockLLMAdapter(model=model)

        raise ConfigurationError(f"Proveedor LLM no soportado: {provider}")

    @staticmethod
    def create_from_name(  # pylint: disable=too-many-arguments
        name: LLMProviderName,
        *,
        model: str = "ollama/qwen2.5:3b",
        api_base: str | None = "http://localhost:11434",
        api_key: str | None = None,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        temperature: float = 0.7,
    ) -> LLMPort:
        if name == "litellm":
            return LiteLLMAdapter(
                model=model,
                api_base=api_base,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
            )
        if name == "ollama":
            return OllamaAdapter(
                base_url=base_url,
                model=model.removeprefix("ollama/"),
                timeout_seconds=timeout_seconds,
                temperature=temperature,
            )
        if name == "mock":
            return MockLLMAdapter(model=model)
        raise ConfigurationError(f"Proveedor LLM no soportado: {name}")
