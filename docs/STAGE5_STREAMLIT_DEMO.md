# Stage 5 — Streamlit Demo

The Streamlit app wires together the Stage 4 `run_maintenance_pipeline()` with a simple upload-and-view UI. It lets a user upload a car photo, runs the CNN prediction, and renders the structured maintenance report.

## Run

```bash
# From the project root, with the venv active
streamlit run src/app/main.py
```

The app opens at `http://localhost:8501`.

## Expected Flow

1. The sidebar shows default paths to the Stage 2 EfficientNetB0 model and class mapping.
2. Upload a PNG/JPEG car image via the file uploader.
3. The app loads the Keras model (cached for the session — first load takes ~5-10s).
4. The CNN predicts the make/model/year and shows:
   - Predicted vehicle label + confidence
   - Top-5 predictions table
   - Top-5 confidence bar chart
5. The LLM generates a structured maintenance report with the required sections:
   - Vehicle
   - Common Issues
   - Preventive Maintenance Checklist
   - Mileage-Based Notes
   - Customer-Friendly Summary
   - Safety Disclaimer
6. An expander shows the raw prompt sent to the LLM (for transparency during grading).
7. A "Download report as Markdown" button saves the report locally.

## Sidebar Options

| Option | Default | Description |
| --- | --- | --- |
| Model file | `artifacts/stage2_full/efficientnetb0/best_model.keras` | Path to the trained Keras model. |
| Class mapping | `artifacts/stage2_full/efficientnetb0/class_mapping.json` | Path to the class mapping JSON. |
| Image size | `160` | Square image size for inference (must match training). |
| Mileage | `0` | Optional vehicle mileage. Set to 0 to skip. |
| Customer concern | (empty) | Optional free-text concern passed to the LLM. |
| Use fake LLM (offline) | ON | Use the deterministic fake client. Uncheck to call the OpenAI API. |

## Offline Mode (Default)

The "Use fake LLM (offline)" toggle is **ON by default**. This uses `FakeMaintenanceClient`, which produces a deterministic, structured report without any network or API calls. This is ideal for:

- Grading demos without burning API credits
- Offline testing
- Verifying the pipeline end-to-end without an OpenAI key

## Real LLM Mode

To use the real OpenAI API:

1. Ensure `OPENAI_API_KEY` is set in your `.env` file (see `.env.example`).
2. Optionally set `OPENAI_MODEL` (default: `gpt-4o-mini`).
3. Uncheck "Use fake LLM (offline)" in the sidebar.

If the key is missing, the app shows a friendly `st.error` with setup instructions — it never crashes or exposes the key.

## Graceful Failure Handling

| Scenario | Behavior |
| --- | --- |
| Model file missing | `st.error("Model file not found at <path>. Run Stage 2 training first.")` |
| Class mapping missing | `st.error("Class mapping not found at <path>...")` |
| `OPENAI_API_KEY` missing (real LLM) | `st.error("OPENAI_API_KEY is not configured...")` |
| Pipeline exception | `st.error("Pipeline failed: <message>")` |
| Temp file cleanup | Always cleaned up after each run; no user uploads are persisted |

## Security

- The app never logs, prints, or displays `OPENAI_API_KEY` or full API payloads.
- Uploaded images are saved to a temp file only for the duration of one pipeline run, then deleted.
- No user data (images, mileage, concerns) is persisted beyond the current session.

## Troubleshooting

- **TensorFlow first import is slow**: TF 2.21.0 can take 5-10s on first import. The model load spinner shows "Loading model..." during this time.
- **Model not found**: Run Stage 2 training first, or fix the path in the sidebar.
- **OpenAI API key missing**: Either set `OPENAI_API_KEY` in `.env` or keep the fake LLM toggle on.
- **Streamlit not installed**: `pip install streamlit>=1.37.0`.

## Test Commands

```bash
# App helper tests only
python -m pytest tests/test_app.py -q

# Full test suite
python -m pytest -q
```

## Files

- `src/app/main.py` — Streamlit app entry point
- `src/app/pipeline_runner.py` — Helper functions (path validation, LLM client selection, pipeline wrapper)
- `src/app/logging_utils.py` — Logging configuration
- `tests/test_app.py` — 7 tests for the app helpers
