"""Tests de la factory LLM."""

import pytest

from chatbot.domain.exceptions import ConfigurationError
from chatbot.infrastructure.adapters.llm.litellm_adapter import LiteLLMAdapter
from chatbot.infrastructure.adapters.llm.llm_factory import LLMFactory
from chatbot.infrastructure.adapters.llm.mock_adapter import MockLLMAdapter
from chatbot.infrastructure.adapters.llm.ollama_adapter import OllamaAdapter
from chatbot.infrastructure.config.settings import Settings


def test_factory_creates_litellm() -> None:
    settings = Settings(
        llm_provider="litellm",
        litellm_model="ollama/llama3.2:3b",
        litellm_api_base="http://localhost:11434",
    )
    adapter = LLMFactory.create(settings)
    assert isinstance(adapter, LiteLLMAdapter)
    assert adapter.model_name == "ollama/llama3.2:3b"


def test_factory_creates_ollama() -> None:
    settings = Settings(llm_provider="ollama", ollama_model="llama3.2:3b")
    adapter = LLMFactory.create(settings)
    assert isinstance(adapter, OllamaAdapter)
    assert adapter.model_name == "llama3.2:3b"


def test_factory_creates_mock() -> None:
    settings = Settings(llm_provider="mock", litellm_model="fake")
    adapter = LLMFactory.create(settings)
    assert isinstance(adapter, MockLLMAdapter)


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigurationError):
        LLMFactory.create_from_name("unknown")  # type: ignore[arg-type]
