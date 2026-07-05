# car-recognition-maintenance-advisor

Stanford Cars classification plus AI-generated maintenance advisory reports.

This capstone project identifies a vehicle make/model/year from an uploaded image, then generates a customer-friendly maintenance advisory report for the predicted class.

## Project Goals

- Train and compare TensorFlow/Keras transfer-learning CNNs on the Stanford Cars 196-class dataset.
- Use the selected CNN model to predict the class of an input vehicle image.
- Generate a structured maintenance advisory report with the OpenAI API.
- Provide a simple Streamlit demo for image upload, prediction, and report display.

## Architecture

```text
Image Input -> Preprocessing -> CNN Classification -> Prompt Template -> LLM Report -> Streamlit UI
```

## Tech Stack

- Python 3.10+
- TensorFlow/Keras
- EfficientNetB0 and ResNet50 from `tf.keras.applications`
- Stanford Cars dataset from Kaggle: `jutrera/stanford-car-dataset-by-classes-folder`
- OpenAI Python SDK
- Streamlit
- pandas, NumPy, scikit-learn, matplotlib, seaborn, Pillow

## Repository Structure

```text
car-recognition-maintenance-advisor/
├── data/                  # Local dataset files; gitignored except placeholders
├── notebooks/             # EDA and training experiments
├── src/
│   ├── app/               # Streamlit application
│   ├── data/              # Dataset paths, download helpers, EDA utilities
│   ├── inference/         # Image-to-class prediction pipeline
│   ├── llm/               # Prompt templates and LLM client
│   └── model/             # CNN training/evaluation code
├── tests/                 # Unit tests
├── requirements.txt
├── STAGES.md              # Stage-gated implementation plan
└── README.md
```

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create a local `.env` file in the project root. Do not commit it.

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=your_preferred_openai_model_here
```

The `.env` file is ignored by Git.

## Dataset Setup

This project uses the Stanford Cars dataset from Kaggle:

```text
jutrera/stanford-car-dataset-by-classes-folder
```

Expected extracted layout for the verified Stage 1 setup:

```text
data/
├── anno_test.csv
├── anno_train.csv
├── names.csv
└── car_data/
    └── car_data/
        ├── train/
        └── test/
```

The active dataset helper prefers `data/car_data/car_data` when present and falls back to `data/raw/stanford-cars` otherwise.

To build the Kaggle download command from Python:

```python
from src.data.download_stanford_cars import build_kaggle_command

print(build_kaggle_command("data"))
```

## Stage-Gated Workflow

Before starting new work, read `STAGES.md`.

Current rule: no stage begins until every verification item from the previous stage has been explicitly confirmed and reported.

## Useful Commands

Run tests:

```powershell
python -m pytest
```

WSL2 TensorFlow GPU setup and calibration:

```bash
source scripts/activate_wsl_gpu.sh
python scripts/check_tf_gpu.py
```

See `docs/WSL2_TF_GPU_SETUP.md` for the verified WSL2 + GTX 1650 setup and smoke-training commands.

Run and reproduce Stage 2 CNN training/evaluation:

```bash
source scripts/activate_wsl_gpu.sh
python -m src.model.train_cnn --architecture efficientnetb0 --data-dir data/car_data/car_data --image-size 160 --batch-size 8 --epochs 5 --weights imagenet --output-dir "$HOME/artifacts/stage2_full"
```

See `docs/STAGE2_CNN_TRAINING.md` for the full EfficientNetB0/ResNet50 training, comparison, curve plotting, and held-out test evaluation commands.

Run the dataset EDA summary:

```powershell
python -m src.data.eda_cli
```

Run the Streamlit app shell:

```powershell
streamlit run src/app/main.py
```

Run the Stage 6 confusion matrix analysis:

```powershell
python -m src.model.analyze_confusion `
    --predictions artifacts/stage2_full/efficientnetb0/eval_test/test_predictions.csv `
    --class-mapping artifacts/stage2_full/efficientnetb0/class_mapping.json `
    --training-log artifacts/stage2_full/efficientnetb0/training_log.csv `
    --output-dir artifacts/stage6_evaluation
```

See `docs/CAPSTONE_WRITEUP.md` for the full capstone write-up, including CNN metrics, most-confused class pairs, training curves, and an explicit "What Was NOT Built" section.

## Notes

- Do not commit datasets, trained models, logs, reports, or secrets.
- Do not hardcode API keys.
- Keep CNN inference and LLM report generation independently testable.
- Preserve the full 196-class Stanford Cars label set.
