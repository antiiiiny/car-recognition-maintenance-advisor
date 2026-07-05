"""Tests for LLM client configuration and errors."""

from __future__ import annotations

import pytest

from src.llm import client as llm_client
from src.llm.client import DEFAULT_OPENAI_MODEL, LLMConfig, MissingOpenAIKeyError


def test_missing_openai_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing API keys fail with a clear, non-secret error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: False)

    with pytest.raises(MissingOpenAIKeyError, match="OPENAI_API_KEY"):
        LLMConfig.from_env()


def test_openai_model_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default model configuration."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: False)

    config = LLMConfig.from_env()

    assert config.api_key == "test-key"
    assert config.model == DEFAULT_OPENAI_MODEL
    assert config.base_url is None


def test_openai_model_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify OPENAI_MODEL overrides the documented default."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: False)

    config = LLMConfig.from_env()

    assert config.model == "custom-model"


def test_openai_base_url_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify OPENAI_BASE_URL overrides the default (for OpenRouter etc.)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: False)

    config = LLMConfig.from_env()

    assert config.base_url == "https://openrouter.ai/api/v1"