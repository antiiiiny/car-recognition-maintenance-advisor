"""Helpers that wire Streamlit UI inputs to the Stage 4 pipeline.

These functions are kept separate from the Streamlit rendering code so they
can be unit-tested without spinning up a Streamlit ``AppTest`` server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.inference.pipeline import MaintenancePipelineResult, run_maintenance_pipeline
from src.llm.client import LLMConfig, MissingOpenAIKeyError, OpenAIMaintenanceClient
from src.llm.fake_client import FakeMaintenanceClient
from src.llm.report_generator import MaintenanceReportClient


class AppConfigError(RuntimeError):
    """User-friendly configuration error raised by the app helpers."""


def resolve_artifact_path(path_str: str, label: str) -> Path:
    """Resolve and validate a user-supplied artifact path.

    Args:
        path_str: Path string from the sidebar input.
        label: Human-readable name of the artifact (e.g. "Model file").

    Returns:
        Resolved ``Path`` object.

    Raises:
        AppConfigError: If the path is empty or does not exist.
    """
    if not path_str or not path_str.strip():
        raise AppConfigError(f"{label} path is empty. Set it in the sidebar.")

    path = Path(path_str.strip())
    if not path.exists():
        raise AppConfigError(
            f"{label} not found at: {path}\n"
            "Run Stage 2 training first, or fix the path in the sidebar."
        )
    return path


def build_llm_client(
    use_fake: bool,
    api_key: Optional[str] = None,
) -> MaintenanceReportClient:
    """Pick the LLM client based on the sidebar toggle.

    Args:
        use_fake: True if the fake/offline client was selected.
        api_key: Optional explicit API key. If ``None`` and the real client
            is requested, the key is loaded from the environment via
            ``LLMConfig.from_env()``.

    Returns:
        A maintenance report client (fake or real).

    Raises:
        AppConfigError: If the real client was requested but
            ``OPENAI_API_KEY`` is missing. The message never exposes the key
            itself.
    """
    if use_fake:
        return FakeMaintenanceClient()

    try:
        config = LLMConfig(api_key=api_key) if api_key else LLMConfig.from_env()
    except MissingOpenAIKeyError as exc:
        raise AppConfigError(
            "OPENAI_API_KEY is not configured. Add it to your local .env file "
            "or enable the 'Use fake LLM (offline)' toggle in the sidebar."
        ) from exc
    return OpenAIMaintenanceClient(config=config)


def run_pipeline(
    image_path: str | Path,
    model_path: str | Path,
    class_mapping_path: str | Path,
    image_size: int,
    mileage: Optional[int],
    user_concern: Optional[str],
    llm_client: MaintenanceReportClient,
    model=None,
) -> MaintenancePipelineResult:
    """Run the Stage 4 maintenance pipeline with app-side validation.

    Args:
        image_path: Path to the uploaded image.
        model_path: Path to the trained Keras model.
        class_mapping_path: Path to the class mapping JSON.
        image_size: Target square image size.
        mileage: Optional mileage value.
        user_concern: Optional customer concern.
        llm_client: LLM client (fake or real).
        model: Optional preloaded Keras model.

    Returns:
        Pipeline result with prediction and report.
    """
    return run_maintenance_pipeline(
        image_path=image_path,
        model_path=model_path,
        class_mapping_path=class_mapping_path,
        image_size=image_size,
        mileage=mileage,
        user_concern=user_concern,
        llm_client=llm_client,
        model=model,
    )
