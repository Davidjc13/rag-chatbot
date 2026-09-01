"""Factory de chat models LangChain para el agente ReAct."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from chatbot.core.env import Env
from chatbot.domain.exceptions import ConfigurationError


class BindableFakeChatModel(FakeMessagesListChatModel):
    """FakeChatModel con bind_tools no-op (tests y LLM_PROVIDER=mock)."""

    def bind_tools(self, tools: object, **kwargs: object) -> BindableFakeChatModel:
        return self


class ChatModelFactory:
    """Crea el chat model de LangChain alineado con LLM_PROVIDER."""

    def __init__(self, env: Env) -> None:
        self._env = env

    def create(self, *, model: str | None = None) -> BaseChatModel:
        env = self._env
        selected = (model or "").strip() or env.active_model
        provider = env.llm_provider

        if provider == "mock":
            return BindableFakeChatModel(
                responses=[
                    AIMessage(
                        content="El proveedor mock no ejecuta el agente ReAct."
                    )
                ]
            )

        if provider == "ollama" or selected.startswith("ollama/"):
            name = selected.removeprefix("ollama/")
            base_url = (
                env.ollama_base_url
                if provider == "ollama"
                else (env.litellm_api_base or env.ollama_base_url)
            )
            return ChatOllama(
                model=name,
                base_url=base_url,
                temperature=(
                    env.ollama_temperature
                    if provider == "ollama"
                    else env.litellm_temperature
                ),
                num_predict=env.ollama_num_predict,
                reasoning=env.ollama_think if provider == "ollama" else False,
            )

        if provider == "litellm":
            from langchain_community.chat_models import (  # pylint: disable=import-outside-toplevel
                ChatLiteLLM,
            )

            return ChatLiteLLM(
                model=selected,
                api_base=env.litellm_api_base,
                api_key=env.litellm_api_key or "sk-none",
                temperature=env.litellm_temperature,
            )

        raise ConfigurationError(f"Proveedor LLM no soportado para el agente: {provider}")
