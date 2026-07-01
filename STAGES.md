# Project Stages

Hard rule: No stage begins until the previous stage's verification criteria are explicitly confirmed and reported. This applies even under time pressure. If a verification step has not been run, the next stage does not start. Before moving on, state clearly that verification passed and why.

## Stage Summary

| Stage | Name | Scope | Status |
| --- | --- | --- | --- |
| 1 | Foundation & Data Readiness | Repo scaffold, dataset extraction, path handling, EDA, sanity checks | In progress |
| 2 | CNN Pipeline | Data loading, augmentation, EfficientNetB0 and ResNet50 training, comparison, model selection | Not started |
| 3 | LLM Component | Prompt template, OpenAI API wrapper, report generation | Not started |
| 4 | Pipeline Wiring | Image in -> CNN prediction -> prompt -> LLM report, as one callable | Not started |
| 5 | Streamlit Demo | Upload photo, show prediction, show generated report | Not started |
| 6 | Evaluation & Write-up | CNN metrics, confusion matrix, qualitative LLM review, capstone documentation | Not started |

## Stage 1 - Foundation & Data Readiness

Repo scaffold, dataset extraction, path handling, EDA, sanity checks.

Verification criteria:

- [ ] All 196 class folders present, none empty
- [ ] Image counts match expected train/test split
- [ ] Zero corrupt/unreadable files (run an explicit integrity check, e.g. attempt to open every image)
- [ ] At least 3 sample images per a few random classes render correctly when previewed

## Stage 2 - CNN Pipeline

Data loading, augmentation, EfficientNetB0 and ResNet50 training, comparison, model selection.

Verification criteria:

- [ ] Both models complete training without errors
- [ ] Metrics are logged, including accuracy, top-5 accuracy given 196 classes, and loss curves
- [ ] A winner is selected using a stated rule, such as higher top-1 accuracy on the held-out test set, with ties broken by top-5 accuracy

## Stage 3 - LLM Component

Prompt template, OpenAI API wrapper, report generation.

Verification criteria:

- [ ] Test against at least 5 different predicted labels
- [ ] Each output is structurally consistent with the same sections and format every time
- [ ] No hallucinated make/model info contradicts the input label

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

## Stage 6 - Evaluation & Write-up

CNN metrics, confusion matrix, qualitative LLM review, capstone documentation.

Verification criteria:

- [ ] Confusion matrix is generated and reviewed for the most-confused classes
- [ ] Evaluation artifacts are reproducible, meaning re-running the eval script gives consistent results
- [ ] Write-up is complete and matches what was actually built, with no mention of unimplemented features
