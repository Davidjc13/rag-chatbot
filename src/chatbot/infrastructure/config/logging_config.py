"""Configuración centralizada de logging."""

from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger.json import JsonFormatter


class _ColorFormatter(logging.Formatter):
    """Formato legible para desarrollo local."""

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        return (
            f"{self.formatTime(record)} | {record.levelname:<8} | "
            f"{record.name} | {record.message}"
        )


def setup_logging(*, level: str = "INFO", json_logs: bool = False) -> None:
    """Configura el logging de la aplicación una sola vez."""
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())

    if json_logs:
        formatter: logging.Formatter = JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    else:
        formatter = _ColorFormatter()

    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Silenciar ruido de librerías
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_extra(**kwargs: Any) -> dict[str, Any]:
    return kwargs
