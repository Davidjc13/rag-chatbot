"""Tests del tracer Langfuse."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from chatbot.domain.ports import ChatGenerationTrace
from chatbot.infrastructure.adapters.observability.langfuse_tracer import LangfuseTracer
from chatbot.infrastructure.adapters.observability.noop_tracer import NoOpTracer

_SAMPLE_TRACE = ChatGenerationTrace(
    user_query="¿Plazo?",
    chunk_ids=("c1", "c2"),
    chunk_scores=(0.9, 0.7),
    model_response="30 días",
    model="qwen3:4b",
    conversation_id="conv-abc",
    duration_ms=250,
    retrieval_duration_ms=45,
    mode="stream",
    retrieval_backend="postgres",
    input_tokens=100,
    output_tokens=50,
)


def test_noop_tracer_does_not_raise() -> None:
    tracer = NoOpTracer()
    tracer.record_chat_generation(_SAMPLE_TRACE)


def test_langfuse_tracer_sends_nested_observations() -> None:
    root = MagicMock()
    root.__enter__ = MagicMock(return_value=root)
    root.__exit__ = MagicMock(return_value=False)
    retrieval = MagicMock()
    retrieval.__enter__ = MagicMock(return_value=retrieval)
    retrieval.__exit__ = MagicMock(return_value=False)
    generation = MagicMock()
    generation.__enter__ = MagicMock(return_value=generation)
    generation.__exit__ = MagicMock(return_value=False)

    client = MagicMock()
    client.start_as_current_observation.side_effect = [root, retrieval, generation]

    with patch("langfuse.Langfuse", return_value=client):
        tracer = LangfuseTracer(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://localhost:3000",
        )
        tracer.record_chat_generation(_SAMPLE_TRACE)

    assert client.start_as_current_observation.call_count == 3
    root.update.assert_called_once()
    root_kwargs = root.update.call_args.kwargs
    assert root_kwargs["input"] == "¿Plazo?"
    assert root_kwargs["output"] == "30 días"
    retrieval.update.assert_called_once()
    retrieval_kwargs = retrieval.update.call_args.kwargs
    assert retrieval_kwargs["input"] == "¿Plazo?"
    assert retrieval_kwargs["output"] == [
        {"chunk_id": "c1", "score": 0.9},
        {"chunk_id": "c2", "score": 0.7},
    ]
    assert retrieval_kwargs["metadata"]["duration_ms"] == 45
    generation.update.assert_called_once()
    gen_kwargs = generation.update.call_args.kwargs
    assert gen_kwargs["usage_details"] == {"input": 100, "output": 50}
    client.flush.assert_called_once()
