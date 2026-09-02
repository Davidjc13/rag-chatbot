"""Autenticación por API key y rate limiting en memoria."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from hmac import compare_digest
from threading import Lock

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from chatbot.core.env import Env

logger = logging.getLogger(__name__)

_HEALTH_PATH = "/api/v1/health"
_WINDOW_SECONDS = 60.0


class SlidingWindowLimiter:
    """Ventana deslizante en memoria (no se comparte entre réplicas)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, limit: int) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic()
        cutoff = now - _WINDOW_SECONDS
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


_limiter = SlidingWindowLimiter()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", maxsplit=1)[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _rate_limit_bucket(method: str, path: str) -> tuple[str, str] | None:
    normalized = path.rstrip("/") or "/"
    if method == "POST" and normalized in {
        "/api/v1/chat",
        "/api/v1/chat/stream",
        "/api/v1/transcribe",
    }:
        return "chat", "rate_limit_chat"
    if method in {"POST", "PUT"} and (
        normalized == "/api/v1/documents" or normalized.startswith("/api/v1/documents/")
    ):
        return "upload", "rate_limit_upload"
    if method == "POST" and normalized.startswith("/api/v1/evals"):
        return "eval", "rate_limit_eval"
    return None


def _limit_value(env: Env, attr: str) -> int:
    return int(getattr(env, attr))


class SecurityMiddleware(BaseHTTPMiddleware):
    """Protege /api/v1 salvo health: Bearer token opcional y rate limit."""

    def __init__(self, app: ASGIApp, *, env: Env | None = None) -> None:
        super().__init__(app)
        self._env = env

    def _settings(self) -> Env:
        return self._env or Env.get_instance()

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)
        if request.method == "GET" and path.rstrip("/") == _HEALTH_PATH:
            return await call_next(request)

        env = self._settings()
        bucket = _rate_limit_bucket(request.method, path)
        if bucket is not None:
            name, attr = bucket
            limit = _limit_value(env, attr)
            key = f"{_client_ip(request)}:{name}"
            if not _limiter.allow(key, limit):
                logger.warning("Rate limit excedido: %s %s", request.method, path)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Demasiadas peticiones. Inténtalo más tarde.",
                        "code": "rate_limited",
                    },
                )

        if env.auth_enabled:
            api_key = env.auth_api_key
            if not api_key:
                logger.error("AUTH_ENABLED=true pero AUTH_API_KEY no está definida")
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Autenticación mal configurada",
                        "code": "auth_misconfigured",
                    },
                )
            header = request.headers.get("authorization") or ""
            prefix = "Bearer "
            provided = header[len(prefix) :] if header.startswith(prefix) else ""
            if not provided or not compare_digest(provided, api_key):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "API key inválida o ausente",
                        "code": "unauthorized",
                    },
                )

        return await call_next(request)
