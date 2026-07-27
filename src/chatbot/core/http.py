"""Singleton de cliente HTTP asíncrono compartido."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

from chatbot.core.env import Env


class AsyncHttpClient:
    """Cliente httpx.AsyncClient único reutilizable en toda la app."""

    _instance: AsyncHttpClient | None = None
    _lock = threading.Lock()

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        timeout = timeout_seconds
        if timeout is None:
            timeout = Env.get_instance().http_timeout_seconds
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._client_lock = threading.Lock()

    @classmethod
    def get_instance(cls, *, timeout_seconds: float | None = None) -> AsyncHttpClient:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(timeout_seconds=timeout_seconds)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Resetea el singleton (tests). No cierra el cliente; usar aclose()."""
        with cls._lock:
            cls._instance = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    @property
    def client(self) -> httpx.AsyncClient:
        return self._ensure_client()

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        return await self.client.get(
            url,
            headers=headers,
            params=params,
            timeout=timeout if timeout is not None else self._timeout,
        )

    async def post(
        self,
        url: str,
        *,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        return await self.client.post(
            url,
            json=json,
            headers=headers,
            timeout=timeout if timeout is not None else self._timeout,
        )

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[httpx.Response]:
        async with self.client.stream(
            method,
            url,
            json=json,
            headers=headers,
            timeout=timeout if timeout is not None else self._timeout,
        ) as response:
            yield response

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
