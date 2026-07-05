"""Streamlit app for the car-recognition-maintenance-advisor.

Run with::

    streamlit run src/app/main.py

The app lets a user upload a car photo, runs the Stage 4
``run_maintenance_pipeline()`` end-to-end, and renders the predicted vehicle
plus a structured maintenance report. A sidebar toggle selects the fake LLM
client (default ON) so the app works fully offline without an OpenAI API key.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Streamlit runs this file directly, so the project root is not on sys.path.
# Add it before importing anything from the `src` package.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.app.logging_utils import configure_logging
from src.app.pipeline_runner import AppConfigError, build_llm_client, resolve_artifact_path, run_pipeline
from src.inference.predictor import (
    DEFAULT_CLASS_MAPPING_PATH,
    DEFAULT_INFERENCE_IMAGE_SIZE,
    DEFAULT_MODEL_PATH,
    load_prediction_model,
)


@st.cache_resource(show_spinner="Loading model...")
def _load_cached_model(model_path: str, image_size: int):  # noqa: ARG001 - image_size keys the cache
    """Load the Keras model once per session.

    Args:
        model_path: Path to the trained Keras model.
        image_size: Target image size (used only to key the cache).

    Returns:
        Loaded Keras model.
    """
    return load_prediction_model(model_path)


def _render_sidebar() -> dict:
    """Render the sidebar controls and return the selected options.

    Returns:
        Dictionary of sidebar options.
    """
    st.sidebar.title("Settings")

    options = {
        "model_path": st.sidebar.text_input(
            "Model file",
            value=str(DEFAULT_MODEL_PATH),
            help="Path to the trained Keras model (.keras).",
        ),
        "class_mapping_path": st.sidebar.text_input(
            "Class mapping",
            value=str(DEFAULT_CLASS_MAPPING_PATH),
            help="Path to the class mapping JSON.",
        ),
        "image_size": st.sidebar.number_input(
            "Image size",
            min_value=32,
            max_value=512,
            value=DEFAULT_INFERENCE_IMAGE_SIZE,
            step=32,
            help="Square image size used for inference (must match training).",
        ),
        "mileage": st.sidebar.number_input(
            "Mileage (optional)",
            min_value=0,
            max_value=1_000_000,
            value=0,
            step=1000,
            help="Vehicle mileage. Set to 0 to skip.",
        ),
        "user_concern": st.sidebar.text_area(
            "Customer concern (optional)",
            value="",
            help="Free-text customer concern to pass to the LLM.",
        ),
        "use_fake_llm": st.sidebar.checkbox(
            "Use fake LLM (offline)",
            value=True,
            help="Use the deterministic fake client. Uncheck to call the OpenAI API.",
        ),
    }

    return options


def _render_prediction(prediction) -> None:
    """Render the prediction block.

    Args:
        prediction: ``VehiclePrediction`` from the pipeline.
    """
    st.subheader("Prediction")
    st.write(f"**Predicted vehicle:** {prediction.label}")
    st.write(f"**Confidence:** {prediction.confidence:.4f}")

    if prediction.top_predictions:
        st.write("**Top-5 predictions:**")
        rows = [
            {"Rank": i + 1, "Label": p.label, "Probability": p.probability}
            for i, p in enumerate(prediction.top_predictions)
        ]
        st.dataframe(rows, use_container_width=True)

        try:
            chart_data = {p.label: p.probability for p in prediction.top_predictions}
            st.bar_chart(chart_data, use_container_width=True)
        except Exception:
            # Bar chart is a nice-to-have; never let it crash the app.
            pass


def main() -> None:
    """Render the Streamlit app."""
    configure_logging()
    st.set_page_config(
        page_title="Car Recognition & Maintenance Advisor",
        page_icon="🚗",
        layout="wide",
    )
    st.title("🚗 Car Recognition & Maintenance Advisor")
    st.write(
        "Upload a photo of a car. The CNN predicts the make/model/year, then a "
        "generative model produces a maintenance advisory report."
    )

    options = _render_sidebar()

    uploaded_file = st.file_uploader(
        "Upload a car image",
        type=["png", "jpg", "jpeg"],
        help="PNG or JPEG image of a car.",
    )

    if uploaded_file is None:
        st.info("Awaiting image upload.")
        return

    # Display the uploaded image.
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    # Validate artifact paths before doing any heavy work.
    try:
        model_path = resolve_artifact_path(options["model_path"], "Model file")
        mapping_path = resolve_artifact_path(options["class_mapping_path"], "Class mapping")
    except AppConfigError as exc:
        st.error(str(exc))
        return

    # Build the LLM client (fake or real).
    try:
        llm_client = build_llm_client(use_fake=options["use_fake_llm"])
    except AppConfigError as exc:
        st.error(str(exc))
        return

    # Load the model (cached).
    try:
        model = _load_cached_model(str(model_path), int(options["image_size"]))
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    # Save the uploaded file to a temp path for the pipeline.
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    mileage = int(options["mileage"]) if int(options["mileage"]) > 0 else None
    concern = options["user_concern"].strip() or None

    try:
        with st.spinner("Running pipeline..."):
            result = run_pipeline(
                image_path=tmp_path,
                model_path=model_path,
                class_mapping_path=mapping_path,
                image_size=int(options["image_size"]),
                mileage=mileage,
                user_concern=concern,
                llm_client=llm_client,
                model=model,
            )
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return
    finally:
        # Clean up the temp file; we never persist user uploads.
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    _render_prediction(result.prediction)

    st.subheader("Maintenance Report")
    st.markdown(result.report)

    # Optional: show the raw prompt for transparency (helps grading).
    with st.expander("View prompt sent to LLM", expanded=False):
        from src.llm.prompts import build_maintenance_prompt

        prompt = build_maintenance_prompt(
            predicted_label=result.prediction.label,
            mileage=mileage,
            user_concern=concern,
        )
        st.text(prompt)

    # Optional: download the report as Markdown.
    st.download_button(
        label="Download report as Markdown",
        data=result.report,
        file_name=f"maintenance_report_{result.prediction.label.replace(' ', '_')}.md",
        mime="text/markdown",
    )


if __name__ == "__main__":
    main()
