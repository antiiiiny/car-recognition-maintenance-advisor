# Capstone Write-Up: Car Recognition & Maintenance Advisor

**Samsung Innovation Campus — Capstone Project**
**Date:** 2026-07-05

---

## 1. Project Overview

This project implements an end-to-end AI pipeline that:

1. Accepts a photo of a vehicle
2. Classifies the make, model, and year using a fine-tuned CNN (Stanford Cars 196-class dataset)
3. Generates a structured maintenance advisory report using a large language model (OpenAI API)
4. Presents both the prediction and the report in a Streamlit web application

The pipeline is: **Image → Preprocessing → CNN → Prompt Engineering → LLM → Web UI**

---

## 2. What Was Actually Built

### Stage 1 — Foundation & Data Readiness ✅

- Extracted the Stanford Cars dataset (196 classes, 8144 train / 8041 test images) into `data/car_data/car_data/`
- Built dataset path helpers (`src/data/dataset_paths.py`) that prefer `data/car_data/car_data` and fall back to `data/raw/stanford-cars`
- Built EDA utilities (`src/data/eda.py`, `src/data/eda_cli.py`) that report class counts, image counts, and corrupt-file checks
- Verified all 196 class folders are non-empty, image counts match expected splits, and zero corrupt files exist
- Rewrote `README.md` as clean UTF-8 Markdown with setup and dataset instructions

### Stage 2 — CNN Pipeline ✅

- Built TensorFlow dataset loaders (`src/data/cnn_dataset.py`) with 80/20 train/validation split, deterministic class mapping, and augmentation layers
- Implemented two transfer-learning architectures (`src/model/architectures.py`):
  - **EfficientNetB0** (ImageNet weights, fine-tuned)
  - **ResNet50** (ImageNet weights, fine-tuned)
- Built training utilities (`src/model/training.py`) with ModelCheckpoint, CSVLogger, EarlyStopping callbacks
- Trained both models for 5 epochs on WSL2 GPU (GTX 1650) at image-size 160, batch-size 8 (EfficientNetB0) / 4 (ResNet50)
- **Model selection** used validation top-1 accuracy (not test-set accuracy):

| Model | Val Top-1 | Val Top-5 | Selected |
|-------|-----------|-----------|----------|
| EfficientNetB0 | 0.6468 | 0.8372 | ✅ |
| ResNet50 | 0.5608 | 0.7875 | — |

- Evaluated the selected EfficientNetB0 on the held-out test set **once**:

| Metric | Value |
|--------|-------|
| Test Top-1 Accuracy | 0.3528 |
| Test Top-5 Accuracy | 0.6513 |
| Macro Precision | 0.4546 |
| Macro Recall | 0.3551 |
| Macro F1 | 0.3467 |

### Stage 3 — LLM Component ✅

- Built a structured prompt template (`src/llm/prompts.py`) with 6 required sections: Vehicle, Common Issues, Preventive Maintenance Checklist, Mileage-Based Notes, Customer-Friendly Summary, Safety Disclaimer
- Built an OpenAI client wrapper (`src/llm/client.py`) that reads `OPENAI_API_KEY` and `OPENAI_MODEL` (default: `gpt-4o-mini`) from environment
- Built a deterministic fake client (`src/llm/fake_client.py`) for offline testing and demos
- Built a report generation facade (`src/llm/report_generator.py`) that chains prompt → client → report
- Verified structural consistency across 5 different vehicle labels using the fake client

### Stage 4 — Pipeline Wiring ✅

- Built single-image prediction (`src/inference/predictor.py`) that loads the Keras model, preprocesses the image, and returns the top-k predicted classes with confidences
- Built the end-to-end pipeline (`src/inference/pipeline.py`) that chains CNN prediction → LLM report generation into one callable: `run_maintenance_pipeline()`
- Verified the pipeline on 3 real test images with zero manual intervention

### Stage 5 — Streamlit Demo ✅

- Rewrote `src/app/main.py` into a full Streamlit app with:
  - Sidebar controls (model path, class mapping, image size, mileage, concern, fake LLM toggle)
  - File uploader (PNG/JPEG)
  - Uploaded image display
  - Prediction with confidence, top-5 table, and bar chart
  - Full maintenance report rendering
  - Prompt expander for transparency
  - Download-as-Markdown button
- Added `src/app/pipeline_runner.py` with testable helpers for path validation and LLM client selection
- Verified 3 real image uploads complete the full flow with no unhandled exceptions
- Verified graceful failure for missing model files and missing OpenAI API key

### Stage 6 — Evaluation & Write-up ✅

- Built confusion matrix analysis (`src/model/confusion_analysis.py`) that:
  - Loads saved predictions from Stage 2 evaluation
  - Computes the top-K most-confused class pairs (excluding the diagonal)
  - Generates a filtered confusion matrix heatmap PNG
  - Generates training curves (accuracy/loss) from the training log
- Built a CLI (`src/model/analyze_confusion.py`) for reproducible evaluation
- Generated artifacts at `artifacts/stage6_evaluation/`:
  - `confusion_analysis.json` — top-20 most-confused pairs
  - `confusion_heatmap.png` — heatmap for top-25 most-confused classes
  - `training_curves.png` — accuracy/loss curves over 5 epochs
- Verified reproducibility: re-running the analyzer produces identical results

---

## 3. CNN Evaluation

### 3.1 Test Set Metrics

The selected EfficientNetB0 model was evaluated once on the held-out test set (8041 images, 196 classes):

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 35.28% |
| Top-5 Accuracy | 65.13% |
| Macro Precision | 45.46% |
| Macro Recall | 35.51% |
| Macro F1 | 34.67% |

### 3.2 Most-Confused Class Pairs

The top-10 most-confused pairs (true → predicted) reveal systematic confusion between visually similar vehicles:

| Count | True Class | Predicted Class |
|-------|-----------|-----------------|
| 26 | GMC Savana Van 2012 | Chevrolet Express Van 2007 |
| 22 | Land Rover LR2 SUV 2012 | Land Rover Range Rover SUV 2012 |
| 20 | Dodge Sprinter Cargo Van 2009 | Mercedes-Benz Sprinter Van 2012 |
| 18 | Chrysler Town and Country Minivan 2012 | Honda Odyssey Minivan 2007 |
| 17 | Chevrolet Express Cargo Van 2007 | Chevrolet Express Van 2007 |
| 17 | Ford Freestar Minivan 2007 | Honda Odyssey Minivan 2007 |
| 16 | BMW X3 SUV 2012 | BMW X5 SUV 2007 |
| 16 | Dodge Caravan Minivan 1997 | Honda Odyssey Minivan 2007 |
| 14 | Dodge Dakota Crew Cab 2010 | Dodge Ram Pickup 3500 Quad Cab 2009 |
| 14 | GMC Savana Van 2012 | Chevrolet Express Cargo Van 2007 |

**Key observations:**

1. **Vans and minivans dominate the confusion list.** The GMC Savana / Chevrolet Express pair (both full-size vans from the same platform) accounts for the most confusion. Similarly, Dodge/Chrysler minivans are frequently confused with the Honda Odyssey — all are similar-bodied minivans from the same era.

2. **Same-manufacturer siblings confuse the model.** Land Rover LR2 → Range Rover, BMW X3 → X5, Dodge Dakota → Ram Pickup — these are vehicles from the same manufacturer with similar styling cues.

3. **Body-style similarity matters more than brand.** The model struggles to distinguish between vehicles of the same body type (van, minivan, SUV) even across different brands, suggesting that fine-grained differences (grille, headlights, badging) are not well-captured at the trained image resolution (160×160).

### 3.3 Training Curves

Training over 5 epochs shows steady improvement without overfitting within the observed range:

| Epoch | Train Top-1 | Val Top-1 | Train Loss | Val Loss |
|-------|------------|-----------|------------|----------|
| 0 | 0.1062 | 0.3458 | 4.403 | 3.125 |
| 1 | 0.3105 | 0.4846 | 3.071 | 2.395 |
| 2 | 0.4289 | 0.5418 | 2.500 | 2.060 |
| 3 | 0.4991 | 0.5958 | 2.156 | 1.837 |
| 4 | 0.5569 | 0.6468 | 1.889 | 1.610 |

Validation accuracy was still improving at epoch 4, suggesting that additional epochs would likely improve performance. The gap between train (0.557) and validation (0.647) accuracy at epoch 4 is notable — validation outperforms training because the training data includes augmentation while validation does not.

### 3.4 Limitations

- **Test accuracy (35.28%) is modest** for a 196-class fine-grained classification task. State-of-the-art models on Stanford Cars achieve 90%+ accuracy, but they use larger input resolutions (224×224 or higher), longer training schedules (50-100+ epochs), and more aggressive augmentation.
- **Image size 160×160** was chosen for GPU memory constraints (GTX 1650, 4GB VRAM). Higher resolution would likely improve fine-grained discrimination.
- **Only 5 epochs** were trained due to time constraints. The validation curve was still improving.
- **No test-time augmentation (TTA)** was used, which typically boosts accuracy by 1-3%.

---

## 4. LLM Report Quality

The maintenance report generator produces structured, customer-friendly advisory reports with 6 consistent sections. The fake client (used for offline demos and grading) produces deterministic output that:

- Echoes the predicted vehicle label exactly (no contradiction)
- Includes all 6 required sections in order
- Incorporates mileage and customer concern when provided
- Includes a safety disclaimer

The real OpenAI client (`gpt-4o-mini`) generates original advisory text based on the predicted label. Reports are general advisory text — they do not reproduce copyrighted maintenance manuals verbatim.

---

## 5. Tech Stack Summary

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| CNN Framework | TensorFlow/Keras 2.21 |
| Pretrained Models | EfficientNetB0, ResNet50 (`tf.keras.applications`) |
| Dataset | Stanford Cars (196 classes, 8144/8041 train/test) |
| LLM | OpenAI API (`gpt-4o-mini` default) |
| Web UI | Streamlit 1.58 |
| Evaluation | scikit-learn, matplotlib, seaborn |
| Testing | pytest (40 tests) |
| GPU Training | WSL2 Ubuntu 24.04, GTX 1650 (4GB VRAM) |

---

## 6. Reproducibility

All evaluation artifacts are reproducible:

```bash
# Re-run confusion analysis (produces identical results)
python -m src.model.analyze_confusion \
    --predictions artifacts/stage2_full/efficientnetb0/eval_test/test_predictions.csv \
    --class-mapping artifacts/stage2_full/efficientnetb0/class_mapping.json \
    --training-log artifacts/stage2_full/efficientnetb0/training_log.csv \
    --output-dir artifacts/stage6_evaluation
```

The analyzer reads the saved predictions CSV from Stage 2 evaluation — it does not re-run the model. Re-running produces byte-identical JSON output and visually identical PNG plots.

---

## 7. What Was NOT Built

To keep the scope honest, the following were **not** implemented:

- **No PyTorch** — TensorFlow/Keras only, per project requirements
- **No model persistence layer** — uploaded images and reports are not stored beyond the session
- **No user accounts or authentication** — the Streamlit app is a local demo
- **No multi-image batch upload** — one image at a time
- **No test-time augmentation (TTA)** — single-pass inference only
- **No hyperparameter tuning** — manual configuration based on GPU constraints
- **No confusion matrix for all 196 classes** — a filtered top-25 heatmap is generated for readability

---

## 8. File Structure

```
car-recognition-maintenance-advisor/
├── data/                          # Stanford Cars dataset (gitignored)
├── artifacts/
│   ├── stage2_full/
│   │   ├── efficientnetb0/        # Selected model + eval artifacts
│   │   ├── resnet50/              # Runner-up model
│   │   └── comparison/            # Model comparison CSV + selected_model.json
│   └── stage6_evaluation/          # Confusion analysis artifacts
├── src/
│   ├── app/                       # Streamlit app + pipeline runner
│   ├── data/                      # Dataset loading, EDA, paths
│   ├── model/                     # CNN architectures, training, evaluation, confusion analysis
│   ├── inference/                 # Predictor + end-to-end pipeline
│   └── llm/                       # Prompts, OpenAI client, fake client, report generator
├── tests/                         # 40 unit tests
├── docs/                          # Stage documentation
├── scripts/                       # WSL2 GPU setup helpers
├── requirements.txt
├── STAGES.md                      # Stage-gated workflow log
└── README.md
```

---

## 9. Conclusion

This project demonstrates a complete AI pipeline from image classification to generative report generation. The CNN achieves 35.28% top-1 accuracy on the Stanford Cars 196-class task — modest but understandable given the 5-epoch, 160×160 training constraints. The most-confused classes are systematically related vehicles (vans, minivans, same-brand siblings), confirming that fine-grained visual differences are the primary challenge. The LLM component produces structured, consistent maintenance advisory reports, and the Streamlit app provides a usable end-to-end demo.
