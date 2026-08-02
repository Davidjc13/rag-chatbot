"""Adaptador Langfuse para trazas de chat RAG."""

from __future__ import annotations

import logging

from langfuse import propagate_attributes

from chatbot.domain.ports import ChatGenerationTrace, TracingPort

logger = logging.getLogger(__name__)

_TRACE_NAME = {"stream": "rag-chat-stream", "sync": "rag-chat-sync"}


class LangfuseTracer(TracingPort):
    """Envía trazas RAG (retrieval + generación) a Langfuse."""

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
    ) -> None:
        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host.rstrip("/"),
        )

    def record_chat_generation(self, trace: ChatGenerationTrace) -> None:
        try:
            usage: dict[str, int] = {}
            if trace.input_tokens is not None:
                usage["input"] = trace.input_tokens
            if trace.output_tokens is not None:
                usage["output"] = trace.output_tokens

            retrieval_output = [
                {"chunk_id": chunk_id, "score": score}
                for chunk_id, score in zip(trace.chunk_ids, trace.chunk_scores, strict=False)
            ]

            with propagate_attributes(
                session_id=trace.conversation_id,
                metadata={
                    "conversation_id": trace.conversation_id,
                    "mode": trace.mode,
                },
            ):
                with self._client.start_as_current_observation(
                    as_type="span",
                    name=_TRACE_NAME[trace.mode],
                ) as root:
                    root.update(
                        input=trace.user_query,
                        output=trace.model_response,
                    )

                    with self._client.start_as_current_observation(
                        as_type="span",
                        name="rag-retrieval",
                    ) as retrieval:
                        retrieval.update(
                            input=trace.user_query,
                            output=retrieval_output,
                            metadata={
                                "chunk_ids": list(trace.chunk_ids),
                                "retrieval_backend": trace.retrieval_backend,
                                "duration_ms": trace.retrieval_duration_ms,
                            },
                        )

                    with self._client.start_as_current_observation(
                        as_type="generation",
                        name="llm-response",
                        model=trace.model,
                    ) as generation:
                        generation.update(
                            input=trace.user_query,
                            output=trace.model_response,
                            metadata={
                                "chunk_ids": list(trace.chunk_ids),
                                "duration_ms": trace.duration_ms,
                            },
                            usage_details=usage or None,
                        )

            self._client.flush()
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            logger.exception("Error enviando traza a Langfuse")
