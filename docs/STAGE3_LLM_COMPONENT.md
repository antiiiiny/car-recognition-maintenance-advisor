# Stage 3 - LLM Component

Stage 3 adds the maintenance advisory report generator. It is intentionally separate from CNN inference and Streamlit wiring, which belong to later stages.

## Environment Variables

Required for real OpenAI calls:

```text
OPENAI_API_KEY=your_project_key
```

Optional model override:

```text
OPENAI_MODEL=gpt-4o-mini
```

The implementation never prints or reads `.env` contents directly. It reads configuration through environment variables.

## Fake Mode

Fake mode is deterministic and does not call OpenAI:

```powershell
python -m src.llm.report_cli --fake --label "Acura TL Sedan 2012" --mileage 85000
```

Use fake mode for tests, demos without API credits, and structural verification.

## Real OpenAI Mode

Real mode requires `OPENAI_API_KEY` to be configured locally:

```powershell
python -m src.llm.report_cli --label "Acura TL Sedan 2012" --mileage 85000 --concern "engine noise"
```

If the key is missing, the client raises a clear `MissingOpenAIKeyError` naming `OPENAI_API_KEY`.

## Report Structure

Every report uses these sections:

1. Vehicle
2. Common Issues
3. Preventive Maintenance Checklist
4. Mileage-Based Notes
5. Customer-Friendly Summary
6. Safety Disclaimer

The prompt instructs the LLM to keep the predicted vehicle label unchanged and avoid inventing a different make, model, body style, or year.

## Verification Commands

Focused Stage 3 tests:

```powershell
python -m pytest tests/test_llm_prompts.py tests/test_llm_report_generator.py tests/test_llm_client.py
```

Full suite:

```powershell
python -m pytest
```

Manual fake check:

```powershell
python -m src.llm.report_cli --fake --label "Acura TL Sedan 2012" --mileage 85000
```