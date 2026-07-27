"""Tests de la factory LLM."""

import pytest

from chatbot.core.env import Env
from chatbot.domain.exceptions import ConfigurationError
from chatbot.infrastructure.adapters.llm.litellm_adapter import LiteLLMAdapter
from chatbot.infrastructure.adapters.llm.llm_factory import LLMFactory
from chatbot.infrastructure.adapters.llm.mock_adapter import MockLLMAdapter
from chatbot.infrastructure.adapters.llm.ollama_adapter import OllamaAdapter


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    Env.reset()
    yield
    Env.reset()


def test_factory_creates_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "litellm")
    monkeypatch.setenv("LITELLM_MODEL", "ollama/llama3.2:3b")
    monkeypatch.setenv("LITELLM_API_BASE", "http://localhost:11434")
    adapter = LLMFactory.create(Env.get_instance())
    assert isinstance(adapter, LiteLLMAdapter)
    assert adapter.model_name == "ollama/llama3.2:3b"


def test_factory_creates_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    adapter = LLMFactory.create(Env.get_instance())
    assert isinstance(adapter, OllamaAdapter)
    assert adapter.model_name == "llama3.2:3b"


def test_factory_creates_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LITELLM_MODEL", "fake")
    adapter = LLMFactory.create(Env.get_instance())
    assert isinstance(adapter, MockLLMAdapter)


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigurationError):
        LLMFactory.create_from_name("unknown")  # type: ignore[arg-type]
