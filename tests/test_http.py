"""Tests del singleton AsyncHttpClient."""

from __future__ import annotations

import httpx
import pytest

from chatbot.core.http import AsyncHttpClient


@pytest.fixture(autouse=True)
async def _reset_http() -> None:
    client = AsyncHttpClient.get_instance()
    await client.aclose()
    AsyncHttpClient.reset()
    yield
    client = AsyncHttpClient.get_instance()
    await client.aclose()
    AsyncHttpClient.reset()


def test_http_is_singleton() -> None:
    a = AsyncHttpClient.get_instance(timeout_seconds=5.0)
    b = AsyncHttpClient.get_instance(timeout_seconds=99.0)
    assert a is b


@pytest.mark.asyncio
async def test_http_get_with_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    http = AsyncHttpClient.get_instance(timeout_seconds=5.0)
    await http.aclose()
    http._client = httpx.AsyncClient(transport=transport, timeout=5.0)  # noqa: SLF001

    response = await http.get("http://example.test/ping")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
