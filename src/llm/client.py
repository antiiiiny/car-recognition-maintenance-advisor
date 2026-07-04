"""OpenAI client wrapper for maintenance advisory reports."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only in incomplete local envs
    def load_dotenv() -> bool:
        return False


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class MissingOpenAIKeyError(RuntimeError):
    """Raised when ``OPENAI_API_KEY`` is required but not configured."""


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the OpenAI maintenance report client."""

    api_key: str
    model: str = DEFAULT_OPENAI_MODEL

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load OpenAI configuration from environment variables.

        Returns:
            LLM configuration.

        Raises:
            MissingOpenAIKeyError: If ``OPENAI_API_KEY`` is missing.
        """
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise MissingOpenAIKeyError(
                "OPENAI_API_KEY is not configured. Add it to your local .env file or environment."
            )

        model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
        return cls(api_key=api_key, model=model)


class OpenAIMaintenanceClient:
    """Generate maintenance reports through the OpenAI API."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize the client.

        Args:
            config: Optional OpenAI configuration. If omitted, environment
                variables are used.
        """
        self.config = config or LLMConfig.from_env()

    def generate(self, prompt: str) -> str:
        """Generate a report for the given prompt.

        Args:
            prompt: Prompt text built by ``src.llm.prompts``.

        Returns:
            Generated report text.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("The openai package is required for real LLM report generation.") from exc

        client = OpenAI(api_key=self.config.api_key)
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You generate concise, structured automotive maintenance advisory reports.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""