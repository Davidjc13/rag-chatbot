"""Punto de entrada de la aplicación."""

from __future__ import annotations

import uvicorn

from chatbot.core.env import Env
from chatbot.infrastructure.adapters.api.app import create_app

app = create_app()


def run() -> None:
    env = Env.get_instance()
    uvicorn.run(
        "chatbot.main:app",
        host=env.host,
        port=env.port,
        reload=env.app_env == "development",
    )


if __name__ == "__main__":
    run()
