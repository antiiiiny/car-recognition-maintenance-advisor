# GitHub Copilot Instructions — car-recognition-maintenance-advisor

## Project Overview
This repo implements an end-to-end AI pipeline that identifies a vehicle's make, model, and year from a photo, then uses a generative AI model to produce a maintenance advisory report (common issues, service checklist, plain-English explanation for the customer).

This is a capstone project for the Samsung Innovation Campus program. Code should stay readable and well-commented, since it will be reviewed and graded — prioritize clarity over cleverness.

## Architecture
```
Image Input → Preprocessing → CNN (classification) → Prompt Engineering → LLM (report generation) → Web UI
```

1. **Image Preprocessing**: resize, normalize, augment (train time only)
2. **CNN Model**: fine-tuned EfficientNet-B0 or ResNet50 (transfer learning), 196-class output (make/model/year from Stanford Cars dataset)
3. **Prompt Engineering**: inject predicted class + optional user input (mileage) into a structured prompt template
4. **LLM / GenAI**: generates maintenance checklist, common known issues, and a customer-facing summary
5. **Deployment**: Streamlit app — upload image, view prediction, view generated report

## Tech Stack
- **Language**: Python 3.10+
- **CNN framework**: TensorFlow/Keras only — do not introduce PyTorch
- **Pretrained models**: `tf.keras.applications` — evaluate both EfficientNetB0 and ResNet50 (ImageNet weights, fine-tuned on Stanford Cars) and keep whichever performs better; log both results for the writeup
- **LLM**: OpenAI API — use a configurable model name loaded from `OPENAI_MODEL` with a documented default; abstract the LLM call behind a single function/class so the model could be swapped later if needed
- **NLP libraries**: `openai` Python SDK for API calls; Hugging Face Transformers not needed unless a local model is added later
- **Deployment**: Streamlit
- **Dataset**: Stanford Cars Dataset (Kaggle: `jutrera/stanford-car-dataset-by-classes-folder`) — full 196-class labels (make + model + year), no label collapsing

## Repo Structure Conventions
```
car-recognition-maintenance-advisor/
├── data/                  # raw/processed dataset (gitignored, not committed)
├── notebooks/             # exploratory analysis, training experiments
├── src/
│   ├── data/              # dataset loading, preprocessing, augmentation
│   ├── model/              # CNN architecture, training, evaluation
│   ├── inference/          # prediction pipeline (image -> predicted class)
│   ├── llm/                 # prompt templates, LLM client, report generation
│   └── app/                # Streamlit app
├── tests/                 # unit tests
├── requirements.txt
└── README.md
```

## Coding Guidelines
- Use type hints on all function signatures.
- Docstrings (Google style) on all public functions and classes.
- Never hardcode API keys or secrets — always load from environment variables (`.env` file, gitignored) via `python-dotenv` or `os.environ`. The OpenAI key should be read as `OPENAI_API_KEY`.
- Never read, print, commit, or expose `.env` contents. Only reference required variable names such as `OPENAI_API_KEY` and `OPENAI_MODEL`.
- Keep CNN inference and LLM report generation as separate, independently testable functions — don't couple them into one giant function.
- Prompt templates for the LLM should live in `src/llm/prompts.py` as named constants or template strings, not inline in application code.
- Log predictions and generated reports during development for debugging (but don't log secrets or full API payloads in production code).
- Prefer explicit, readable code over one-liners — this is a learning project, not a golf competition.

## Model & Data Notes
- Class labels follow the Stanford Cars folder/name format, usually `<make> <model/body style> <year>` (e.g. "Acura TL Sedan 2012"). Do not assume the year comes first. Full 196-class label set — do not collapse or simplify.
- Do not commit the dataset itself to the repo; document download/setup steps in `README.md` instead.
- Evaluate the CNN with accuracy, precision/recall (macro-averaged given many classes), and a confusion matrix on the most-confused classes.

## Stage-Gated Workflow Policy
- Always check [STAGES.md](../STAGES.md) before starting any new work in this repo.
- No stage begins until the previous stage's verification criteria are explicitly confirmed and reported.
- This applies even under time pressure: if a verification step has not been run, the next stage does not start.
- Before moving on, state clearly that verification passed and why it passed.
- Do not treat a stage as complete based on partial signals, visual inspection alone, or assumptions.
- When a stage has multiple verification items, every item must pass before advancing.

### Required Stage Order
1. Stage 1 - Foundation & Data Readiness
2. Stage 2 - CNN Pipeline
3. Stage 3 - LLM Component
4. Stage 4 - Pipeline Wiring
5. Stage 5 - Streamlit Demo
6. Stage 6 - Evaluation & Write-up

## What Copilot Should Avoid
- Don't suggest reproducing full copyrighted maintenance manuals or manufacturer service documents verbatim in generated reports — the LLM should generate original explanatory text, optionally citing sources by name.
- Don't suggest storing user-uploaded images or personal data (mileage, etc.) beyond the current session unless a persistence layer is explicitly requested and designed for it.
- Don't invent car makes/models not present in the training label set when suggesting code for the prediction pipeline.