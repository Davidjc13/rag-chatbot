"""Aplicación FastAPI (adaptador primario / driving adapter)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from chatbot.core.env import Env
from chatbot.infrastructure.adapters.api.exception_handlers import register_exception_handlers
from chatbot.infrastructure.adapters.api.routes import router
from chatbot.infrastructure.config.logging_config import setup_logging
from chatbot.infrastructure.container import AppContainer

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(container: AppContainer | None = None) -> FastAPI:
    env = Env.get_instance()
    setup_logging(level=env.log_level, json_logs=env.log_json)
    app_container = container or AppContainer.get_instance()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await app_container.startup()
        app.state.container = app_container
        app.state.settings = app_container.env
        app.state.llm = app_container.llm
        app.state.chat_service = app_container.chat_service
        app.state.ingestion_service = app_container.ingestion_service
        app.state.eval_service = app_container.eval_service
        try:
            yield
        finally:
            await app_container.shutdown()

    app = FastAPI(
        title=env.app_name,
        version="0.1.0",
        description="Chatbot RAG hexagonal con LiteLLM e ingestión de documentos",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url="/static/index.html")

        @app.get("/documents", include_in_schema=False)
        async def documents_page() -> FileResponse:
            return FileResponse(_STATIC_DIR / "documents.html")

        @app.get("/evals", include_in_schema=False)
        async def evals_page() -> FileResponse:
            return FileResponse(_STATIC_DIR / "evals.html")

    return app
