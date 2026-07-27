"""Utilidades core compartidas (singletons)."""

from chatbot.core.env import Env
from chatbot.core.http import AsyncHttpClient

__all__ = ["AsyncHttpClient", "Env"]
