"""Tests del adaptador LiteLLM."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from litellm.exceptions import APIConnectionError, Timeout

from chatbot.domain.entities import Message, Role
from chatbot.domain.exceptions import LLMProviderError, LLMUnavailableError
from chatbot.infrastructure.adapters.llm.litellm_adapter import LiteLLMAdapter


def _completion_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


@pytest.fixture
def adapter() -> LiteLLMAdapter:
    return LiteLLMAdapter(
        model="ollama/qwen2.5:3b",
        api_base="http://localhost:11434",
        temperature=0.2,
        timeout_seconds=30.0,
    )


@pytest.mark.asyncio
async def test_generate_returns_assistant_message(adapter: LiteLLMAdapter) -> None:
    with patch(
        "chatbot.infrastructure.adapters.llm.litellm_adapter.acompletion",
        new_callable=AsyncMock,
        return_value=_completion_response("  Hola mundo  "),
    ) as mock_completion:
        reply = await adapter.generate(
            [Message(role=Role.USER, content="Hola")],
            system_prompt="Sé breve.",
        )

    assert reply.role == Role.ASSISTANT
    assert reply.content == "Hola mundo"
    kwargs = mock_completion.await_args.kwargs
    assert kwargs["model"] == "ollama/qwen2.5:3b"
    assert kwargs["api_base"] == "http://localhost:11434"
    assert kwargs["messages"][0] == {"role": "system", "content": "Sé breve."}
    assert kwargs["messages"][1] == {"role": "user", "content": "Hola"}


@pytest.mark.asyncio
async def test_generate_maps_connection_errors(adapter: LiteLLMAdapter) -> None:
    with patch(
        "chatbot.infrastructure.adapters.llm.litellm_adapter.acompletion",
        new_callable=AsyncMock,
        side_effect=APIConnectionError(message="down", llm_provider="ollama", model="qwen"),
    ):
        with pytest.raises(LLMUnavailableError):
            await adapter.generate([Message(role=Role.USER, content="Hola")])


@pytest.mark.asyncio
async def test_generate_maps_timeout(adapter: LiteLLMAdapter) -> None:
    with patch(
        "chatbot.infrastructure.adapters.llm.litellm_adapter.acompletion",
        new_callable=AsyncMock,
        side_effect=Timeout(message="slow", model="qwen", llm_provider="ollama"),
    ):
        with pytest.raises(LLMUnavailableError):
            await adapter.generate([Message(role=Role.USER, content="Hola")])


@pytest.mark.asyncio
async def test_generate_rejects_empty_content(adapter: LiteLLMAdapter) -> None:
    with patch(
        "chatbot.infrastructure.adapters.llm.litellm_adapter.acompletion",
        new_callable=AsyncMock,
        return_value=_completion_response("   "),
    ):
        with pytest.raises(LLMProviderError, match="vacía"):
            await adapter.generate([Message(role=Role.USER, content="Hola")])


@pytest.mark.asyncio
async def test_generate_stream_yields_deltas(adapter: LiteLLMAdapter) -> None:
    async def fake_stream(**_kwargs: object):
        for piece in ("Ho", "la"):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])

    with patch(
        "chatbot.infrastructure.adapters.llm.litellm_adapter.acompletion",
        new_callable=AsyncMock,
        side_effect=fake_stream,
    ):
        chunks = [
            chunk
            async for chunk in adapter.generate_stream(
                [Message(role=Role.USER, content="Hola")],
                system_prompt="Sé breve.",
            )
        ]

    assert chunks == ["Ho", "la"]


@pytest.mark.asyncio
async def test_health_check_ok(adapter: LiteLLMAdapter) -> None:
    with patch(
        "chatbot.infrastructure.adapters.llm.litellm_adapter.acompletion",
        new_callable=AsyncMock,
        return_value=_completion_response("ok"),
    ):
        assert await adapter.health_check() is True


@pytest.mark.asyncio
async def test_health_check_failure(adapter: LiteLLMAdapter) -> None:
    with patch(
        "chatbot.infrastructure.adapters.llm.litellm_adapter.acompletion",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        assert await adapter.health_check() is False
