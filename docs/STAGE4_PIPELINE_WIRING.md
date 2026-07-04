# Stage 4 - Pipeline Wiring

Stage 4 connects the selected CNN model to the LLM report generator:

```text
image -> CNN prediction -> predicted label -> maintenance report
```

The pipeline can run with the deterministic fake LLM client, so no OpenAI API call is required for verification.

## Default Local Artifacts

The Stage 4 CLI defaults to the project-local selected EfficientNetB0 artifacts:

```text
artifacts/stage2_full/efficientnetb0/best_model.keras
artifacts/stage2_full/efficientnetb0/class_mapping.json
```

These artifacts are gitignored and must exist locally.

## Manual Fake Pipeline Check

```powershell
python -m src.inference.pipeline_cli --fake --image-path "data/car_data/car_data/test/Acura TL Sedan 2012/00001.jpg" --mileage 85000 --concern "engine noise"
```

The output should include a prediction, confidence, top predictions, and a maintenance report.

## Tests

Focused Stage 4 tests:

```powershell
python -m pytest tests/test_inference_predictor.py tests/test_pipeline.py
```

Full suite:

```powershell
python -m pytest
```