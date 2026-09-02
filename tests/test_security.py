"""Tests de autenticación Bearer y rate limiting."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from chatbot.core.env import Env
from chatbot.infrastructure.adapters.api.security import SecurityMiddleware, _limiter


@pytest.fixture(autouse=True)
def _reset_env_and_limiter() -> None:
    Env.reset()
    _limiter._hits.clear()  # noqa: SLF001
    yield
    Env.reset()
    _limiter._hits.clear()  # noqa: SLF001


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityMiddleware)

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/models")
    async def models() -> dict[str, str]:
        return {"ok": "yes"}

    @app.post("/api/v1/chat")
    async def chat() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/static/index.html")
    async def static_page() -> dict[str, str]:
        return {"ok": "static"}

    return app


@pytest.mark.asyncio
async def test_health_and_static_are_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_API_KEY", "secret")
    Env.reset()
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/api/v1/health")
        static = await client.get("/static/index.html")
        models = await client.get("/api/v1/models")
    assert health.status_code == 200
    assert static.status_code == 200
    assert models.status_code == 401


@pytest.mark.asyncio
async def test_valid_bearer_allows_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_API_KEY", "secret")
    Env.reset()
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/v1/models",
            headers={"Authorization": "Bearer secret"},
        )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_auth_disabled_skips_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    Env.reset()
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/models")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("RATE_LIMIT_CHAT", "2")
    Env.reset()
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post("/api/v1/chat")
        second = await client.post("/api/v1/chat")
        third = await client.post("/api/v1/chat")
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["code"] == "rate_limited"
