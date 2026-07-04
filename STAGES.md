# Project Stages

Hard rule: No stage begins until the previous stage's verification criteria are explicitly confirmed and reported. This applies even under time pressure. If a verification step has not been run, the next stage does not start. Before moving on, state clearly that verification passed and why.

## Stage Summary

| Stage | Name | Scope | Status |
| --- | --- | --- | --- |
| 1 | Foundation & Data Readiness | Repo scaffold, dataset extraction, path handling, EDA, sanity checks | Completed |
| 2 | CNN Pipeline | Data loading, augmentation, EfficientNetB0 and ResNet50 training, comparison, model selection | Completed |
| 3 | LLM Component | Prompt template, OpenAI API wrapper, report generation | Completed |
| 4 | Pipeline Wiring | Image in -> CNN prediction -> prompt -> LLM report, as one callable | Completed |
| 5 | Streamlit Demo | Upload photo, show prediction, show generated report | Not started |
| 6 | Evaluation & Write-up | CNN metrics, confusion matrix, qualitative LLM review, capstone documentation | Not started |

## Stage 1 - Foundation & Data Readiness

Repo scaffold, dataset extraction, path handling, EDA, sanity checks.

Verification criteria:

- [x] All 196 class folders present, none empty
- [x] Image counts match expected train/test split
- [x] Zero corrupt/unreadable files (run an explicit integrity check, e.g. attempt to open every image)
- [x] At least 3 sample images per a few random classes render correctly when previewed
- [x] Dataset path helpers, CLI defaults, and tests all agree on the active dataset layout
- [x] `README.md` is valid UTF-8 Markdown and documents setup/download steps
- [x] Stage 1 unit tests pass after cleanup changes

Verified on 2026-07-01: the extracted dataset at `data/car_data/car_data` contains 196 non-empty class folders in both `train` and `test`, the image counts match the expected split counts (`train=8144`, `test=8041`), zero corrupt files were found by opening every image, and sample images from random classes loaded successfully with no preview errors.

Cleanup verified on 2026-07-03: dataset path helpers and `src.data.eda_cli` both resolve the active dataset to `data/car_data/car_data`; `python -m src.data.eda_cli` reported `train=8144`, `test=8041`, 196 classes, 196 names loaded, and zero corrupt files; `python -m pytest` passed with 2 tests; `README.md` was rewritten as UTF-8 Markdown with setup and dataset instructions.

## Stage 2 - CNN Pipeline

Data loading, augmentation, EfficientNetB0 and ResNet50 training, comparison, model selection.

Verification criteria:

- [x] Both models complete training without errors
- [x] TensorFlow data loaders produce train/validation/test datasets with 196 classes
- [x] Class index mapping is saved and reproducible
- [x] A small smoke-training run completes on a subset before full training
- [x] Metrics are logged for both models, including top-1 accuracy, top-5 accuracy given 196 classes, macro precision/recall, and loss curves
- [x] Model checkpoints and training curves are saved
- [x] A winner is selected using validation metrics, such as higher validation top-1 accuracy with ties broken by validation top-5 accuracy
- [x] The final selected model is evaluated on the held-out test set only after model selection

Progress on 2026-07-03: added reusable Stage 2 modules for CNN dataset loading (`src/data/cnn_dataset.py`), transfer model construction (`src/model/architectures.py`), training utilities and winner selection (`src/model/training.py`), CLI training entry point (`src/model/train_cnn.py`), and evaluation metrics (`src/model/evaluation.py`). Unit tests passed with 10 tests. The real dataset class mapping was generated at `artifacts/models/class_mapping.json` with 196 classes.

Smoke verification on 2026-07-03: TensorFlow 2.21.0 imported successfully. `python -m pytest` passed with 10 tests. EfficientNetB0 and ResNet50 each completed a one-epoch, one-batch smoke training run using `--image-size 64 --batch-size 2 --smoke-batches 1 --weights none --output-dir artifacts/smoke_models`. Both runs loaded train/validation/test datasets with 196 classes (`train=8144`, split to 6516 training and 1628 validation examples; `test=8041`) and saved `best_model.keras`, `class_mapping.json`, `config.json`, and `training_log.csv` under their respective smoke artifact directories. Full training, full metrics logging, model comparison, and final test evaluation are still pending.

WSL GPU verification on 2026-07-04: WSL2 Ubuntu 24.04 sees the GTX 1650 via `nvidia-smi`, but TensorFlow initially skipped GPU registration because CUDA/cuDNN runtime libraries were not discoverable from the project-local `.venv-wsl` under `/mnt/c`. A Linux-filesystem venv was created at `$HOME/.venvs/car-cnn-wsl`; `tensorflow[and-cuda]==2.21.0` installed the NVIDIA CUDA 12 pip wheels; `scripts/activate_wsl_gpu.sh` now activates that venv and prepends the NVIDIA wheel `lib` directories to `LD_LIBRARY_PATH`. `python scripts/check_tf_gpu.py` confirmed TensorFlow 2.21.0 lists `/physical_device:GPU:0` with device name `NVIDIA GeForce GTX 1650`. Short WSL GPU calibration training completed for EfficientNetB0 (`--image-size 128 --batch-size 4 --epochs 1 --smoke-batches 2 --weights none --output-dir $HOME/artifacts/wsl_gpu_calibration`) and ResNet50 (`--image-size 128 --batch-size 2 --epochs 1 --smoke-batches 1 --weights none --output-dir $HOME/artifacts/wsl_gpu_calibration`), both saving `best_model.keras`. `python -m pytest` passed with 10 tests after the WSL GPU helper and diagnostic updates. Full training, full metrics logging, model comparison, and final test evaluation remain pending.

Stage 2 completed on 2026-07-04: added saved-model evaluation, training-curve plotting, and run-comparison CLIs (`src.model.evaluate_cnn`, `src.model.plot_training_curves`, and `src.model.compare_runs`). Replaced non-serializable preprocessing Lambda usage so new Keras checkpoints reload for evaluation. `python -m pytest tests/test_evaluation.py tests/test_model_training.py -q` passed with 7 focused tests after the serialization change; a previous full suite run passed with 12 tests. Medium WSL GPU calibration with ImageNet weights completed using `--image-size 160`, batch size 8 for EfficientNetB0 and batch size 4 for ResNet50. Full-dataset training then completed for both models for 5 epochs under `$HOME/artifacts/stage2_full`: EfficientNetB0 reached validation top-1 accuracy `0.6468058968058968` and validation top-5 accuracy `0.8372235872235873`; ResNet50 reached validation top-1 accuracy `0.5608108108108109` and validation top-5 accuracy `0.7874692874692875`. Training logs, `best_model.keras`, `config.json`, `class_mapping.json`, and curve PNGs were saved for both models. Validation macro metrics were generated for both runs. EfficientNetB0 was selected using the documented validation top-1 rule. Only after selection, EfficientNetB0 was evaluated on the held-out test split, producing test accuracy `0.3528168138291257`, test top-5 accuracy `0.6512871533391369`, macro precision `0.45459300697012256`, macro recall `0.35508250328337215`, and macro F1 `0.3466540373708492`. Reproduction commands are documented in `docs/STAGE2_CNN_TRAINING.md`.

Stage 2 verification passed because both candidate CNNs trained successfully on the full dataset split, the required artifacts and metrics were generated for both models, model selection used validation metrics only, and the held-out test split was used only once for the selected EfficientNetB0 model. Stage 3 may now begin.

## Stage 3 - LLM Component

Prompt template, OpenAI API wrapper, report generation.

Verification criteria:

- [x] Test against at least 5 different predicted labels
- [x] Each output is structurally consistent with the same sections and format every time
- [x] No hallucinated make/model info contradicts the input label
- [x] LLM client supports a mock/fake mode for tests to avoid unnecessary API calls
- [x] OpenAI model name is configurable through environment/config
- [x] Missing API key fails with a clear user-friendly error

Stage 3 completed on 2026-07-04: added prompt construction, OpenAI configuration/client wrapper, deterministic fake client, report generation facade, CLI, tests, and documentation (`src/llm/prompts.py`, `src/llm/client.py`, `src/llm/fake_client.py`, `src/llm/report_generator.py`, `src/llm/report_cli.py`, and `docs/STAGE3_LLM_COMPONENT.md`). The fake client was tested against five Stanford Cars labels (`Acura TL Sedan 2012`, `BMW M3 Coupe 2012`, `Audi R8 Coupe 2012`, `Dodge Charger Sedan 2012`, and `smart fortwo Convertible 2012`) and produced the same required sections in order: Vehicle, Common Issues, Preventive Maintenance Checklist, Mileage-Based Notes, Customer-Friendly Summary, and Safety Disclaimer. Missing `OPENAI_API_KEY` raises a clear `MissingOpenAIKeyError`, and `OPENAI_MODEL` is configurable with a documented default. Manual fake CLI verification succeeded with `python -m src.llm.report_cli --fake --label "Acura TL Sedan 2012" --mileage 85000 --concern "engine noise"`. Focused Stage 3 tests passed with `8 passed`; the full test suite passed with `20 passed`. Stage 4 may now begin.

## Stage 4 - Pipeline Wiring

Image in -> CNN prediction -> prompt -> LLM report, as one callable.

Verification criteria:

- [x] Runs end-to-end on a sample image with zero manual intervention
- [x] Tested on at least 3 different images without errors

Stage 4 completed on 2026-07-04: added single-image prediction, end-to-end pipeline wiring, CLI, tests, and documentation (`src/inference/predictor.py`, `src/inference/pipeline.py`, `src/inference/pipeline_cli.py`, and `docs/STAGE4_PIPELINE_WIRING.md`). The pipeline uses the project-local selected EfficientNetB0 artifacts at `artifacts/stage2_full/efficientnetb0/best_model.keras` and `artifacts/stage2_full/efficientnetb0/class_mapping.json`, then sends the predicted label to the Stage 3 report generator. Focused Stage 4 tests passed with `3 passed`; the full test suite passed with `23 passed`. Manual fake-LLM pipeline verification succeeded on three images with zero manual intervention after command start: `data/car_data/car_data/test/Acura TL Sedan 2012/00043.jpg`, `data/car_data/car_data/test/Audi R8 Coupe 2012/00309.jpg`, and `data/car_data/car_data/test/BMW M3 Coupe 2012/00103.jpg`. Each run produced a predicted class, confidence/top-5 predictions, and a structured maintenance report using the fake LLM client, so no OpenAI API call was required. Stage 5 may now begin.

## Stage 5 - Streamlit Demo

Upload photo, show prediction, show generated report.

Verification criteria:

- [ ] App launches cleanly
- [ ] At least 2 to 3 real image uploads complete the full flow with both prediction and report rendered
- [ ] No unhandled exceptions appear in the console during those runs
- [ ] App handles missing model files and missing OpenAI API keys gracefully
- [ ] App does not expose secrets or raw API payloads

## Stage 6 - Evaluation & Write-up

CNN metrics, confusion matrix, qualitative LLM review, capstone documentation.

Verification criteria:

- [ ] Confusion matrix is generated and reviewed for the most-confused classes
- [ ] Evaluation artifacts are reproducible, meaning re-running the eval script gives consistent results
- [ ] Write-up is complete and matches what was actually built, with no mention of unimplemented features
