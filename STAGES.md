# Project Stages

Hard rule: No stage begins until the previous stage's verification criteria are explicitly confirmed and reported. This applies even under time pressure. If a verification step has not been run, the next stage does not start. Before moving on, state clearly that verification passed and why.

## Stage Summary

| Stage | Name | Scope | Status |
| --- | --- | --- | --- |
| 1 | Foundation & Data Readiness | Repo scaffold, dataset extraction, path handling, EDA, sanity checks | Completed |
| 2 | CNN Pipeline | Data loading, augmentation, EfficientNetB0 and ResNet50 training, comparison, model selection | In progress |
| 3 | LLM Component | Prompt template, OpenAI API wrapper, report generation | Not started |
| 4 | Pipeline Wiring | Image in -> CNN prediction -> prompt -> LLM report, as one callable | Not started |
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

- [ ] Both models complete training without errors
- [x] TensorFlow data loaders produce train/validation/test datasets with 196 classes
- [x] Class index mapping is saved and reproducible
- [x] A small smoke-training run completes on a subset before full training
- [ ] Metrics are logged for both models, including top-1 accuracy, top-5 accuracy given 196 classes, macro precision/recall, and loss curves
- [ ] Model checkpoints and training curves are saved
- [ ] A winner is selected using validation metrics, such as higher validation top-1 accuracy with ties broken by validation top-5 accuracy
- [ ] The final selected model is evaluated on the held-out test set only after model selection

Progress on 2026-07-03: added reusable Stage 2 modules for CNN dataset loading (`src/data/cnn_dataset.py`), transfer model construction (`src/model/architectures.py`), training utilities and winner selection (`src/model/training.py`), CLI training entry point (`src/model/train_cnn.py`), and evaluation metrics (`src/model/evaluation.py`). Unit tests passed with 10 tests. The real dataset class mapping was generated at `artifacts/models/class_mapping.json` with 196 classes.

Smoke verification on 2026-07-03: TensorFlow 2.21.0 imported successfully. `python -m pytest` passed with 10 tests. EfficientNetB0 and ResNet50 each completed a one-epoch, one-batch smoke training run using `--image-size 64 --batch-size 2 --smoke-batches 1 --weights none --output-dir artifacts/smoke_models`. Both runs loaded train/validation/test datasets with 196 classes (`train=8144`, split to 6516 training and 1628 validation examples; `test=8041`) and saved `best_model.keras`, `class_mapping.json`, `config.json`, and `training_log.csv` under their respective smoke artifact directories. Full training, full metrics logging, model comparison, and final test evaluation are still pending.

WSL GPU verification on 2026-07-04: WSL2 Ubuntu 24.04 sees the GTX 1650 via `nvidia-smi`, but TensorFlow initially skipped GPU registration because CUDA/cuDNN runtime libraries were not discoverable from the project-local `.venv-wsl` under `/mnt/c`. A Linux-filesystem venv was created at `$HOME/.venvs/car-cnn-wsl`; `tensorflow[and-cuda]==2.21.0` installed the NVIDIA CUDA 12 pip wheels; `scripts/activate_wsl_gpu.sh` now activates that venv and prepends the NVIDIA wheel `lib` directories to `LD_LIBRARY_PATH`. `python scripts/check_tf_gpu.py` confirmed TensorFlow 2.21.0 lists `/physical_device:GPU:0` with device name `NVIDIA GeForce GTX 1650`. Short WSL GPU calibration training completed for EfficientNetB0 (`--image-size 128 --batch-size 4 --epochs 1 --smoke-batches 2 --weights none --output-dir $HOME/artifacts/wsl_gpu_calibration`) and ResNet50 (`--image-size 128 --batch-size 2 --epochs 1 --smoke-batches 1 --weights none --output-dir $HOME/artifacts/wsl_gpu_calibration`), both saving `best_model.keras`. `python -m pytest` passed with 10 tests after the WSL GPU helper and diagnostic updates. Full training, full metrics logging, model comparison, and final test evaluation remain pending.

## Stage 3 - LLM Component

Prompt template, OpenAI API wrapper, report generation.

Verification criteria:

- [ ] Test against at least 5 different predicted labels
- [ ] Each output is structurally consistent with the same sections and format every time
- [ ] No hallucinated make/model info contradicts the input label
- [ ] LLM client supports a mock/fake mode for tests to avoid unnecessary API calls
- [ ] OpenAI model name is configurable through environment/config
- [ ] Missing API key fails with a clear user-friendly error

## Stage 4 - Pipeline Wiring

Image in -> CNN prediction -> prompt -> LLM report, as one callable.

Verification criteria:

- [ ] Runs end-to-end on a sample image with zero manual intervention
- [ ] Tested on at least 3 different images without errors

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
