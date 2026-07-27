"""Punto de entrada de la aplicación."""

from __future__ import annotations

import uvicorn

from chatbot.infrastructure.adapters.api.app import create_app
from chatbot.infrastructure.config.settings import get_settings

app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "chatbot.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    run()
