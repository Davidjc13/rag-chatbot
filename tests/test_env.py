"""Tests del singleton Env lazy."""

from __future__ import annotations

import pytest

from chatbot.core.env import Env


@pytest.fixture(autouse=True)
def _reset_env() -> None:
    Env.reset()
    yield
    Env.reset()


def test_env_is_singleton() -> None:
    a = Env.get_instance()
    b = Env.get_instance()
    assert a is b


def test_env_caches_on_first_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "First")
    env = Env.get_instance()
    assert env.app_name == "First"
    monkeypatch.setenv("APP_NAME", "Second")
    # Cacheado: no relee
    assert env.app_name == "First"
    assert env.get("APP_NAME") == "First"


def test_env_does_not_cache_unused_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNUSED_KEY", "before")
    env = Env.get_instance()
    _ = env.log_level  # toca otra clave
    monkeypatch.setenv("UNUSED_KEY", "after")
    assert env.get("UNUSED_KEY") == "after"


def test_typed_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_MIN_SCORE", "0.5")
    monkeypatch.setenv("LOG_JSON", "true")
    env = Env.get_instance()
    assert env.rag_top_k == 7
    assert env.rag_min_score == 0.5
    assert env.log_json is True


def test_empty_optional_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "")
    env = Env.get_instance()
    assert env.litellm_api_key is None


def test_stt_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    env = Env.get_instance()
    assert env.stt_enabled is True
    assert env.stt_provider == "faster_whisper"
    assert env.stt_model == "base"
    assert env.stt_language == "es"
    assert env.stt_max_audio_seconds == 60


def test_vector_backend_defaults_and_neo4j_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_BACKEND", "neo4j")
    monkeypatch.setenv("NEO4J_URI", "bolt://graph:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    env = Env.get_instance()
    assert env.vector_backend == "neo4j"
    assert env.neo4j_uri == "bolt://graph:7687"
    assert env.neo4j_username == "neo"
    assert env.neo4j_password == "secret"
