# AGENTS.md — Outpatient No-Show Risk Predictor

## Role

Act as a Senior Software Engineer assisting with this Gradio demo: diagnosis, fixes, and safe iteration. Prefer concise, actionable changes. Do not invent clinical claims or alter model scores via the LLM layer.

## Project context

- **Purpose:** Proof-of-concept demo that predicts outpatient appointment no-show risk, explains drivers with SHAP, and optionally drafts staff/patient text via an LLM.
- **Data:** Public Kaggle *Medical Appointment No Shows* (Brazil, 2016). Not NHS data; not for clinical or operational use.
- **Entry point:** `app.py` (Gradio Blocks). Launch with `python app.py` using the conda env `ModelNoShows`.
- **Artifacts (required next to `app.py` or under `outputs/`):**
  - `xgb_noshow.json` — XGBoost classifier (saved with XGBoost 3.4.0)
  - `model_features.json` — feature order + neighbourhood categories

## Architecture (reviewed)

```
UI (Gradio)
  → predict() → build_row() → XGBoost.predict_proba + SHAP TreeExplainer
  → risk_card HTML + matplotlib factor chart + ctx state
  → optional LLM (HF InferenceClient): staff_summary, patient_reminder, chat
```

| Component | Responsibility |
|-----------|----------------|
| `find()` | Resolve artifact paths via `pathlib` (`BASE_DIR` then `outputs/`) |
| `build_row()` | Validate inputs and build model feature frame (incl. categorical neighbourhood) |
| `predict()` | Score + SHAP contributions + UI outputs |
| `call_llm()` | HF chat completion; requires `HF_TOKEN` |
| Fallback helpers | Template text when LLM is unavailable |

## Models in use

1. **XGBoost classifier** (`xgboost`) — primary risk score  
2. **SHAP TreeExplainer** (`shap`) — local feature contributions  
3. **Optional LLM** (`huggingface_hub.InferenceClient`, default `openai/gpt-oss-120b`) — explanation/copy only; must not re-estimate risk  

## Environment

- Conda env: `ModelNoShows` (Python 3.12.x)  
- Interpreter: `C:\Users\Sturistica\anaconda3\envs\ModelNoShows\python.exe`  
- Pins: see `requirements.txt` (gradio 6.28, xgboost 3.4, shap 0.51, pandas 3.0.6, …)  
- Optional secrets: `HF_TOKEN`, `LLM_MODEL`

## Operational rules for future agents

1. **Do not modify source without explicit user authorization** when the user asks for diagnosis-only / audit mode.
2. Prefer the project's conda interpreter; system/base Python may lack dependencies.
3. Never enter or encourage entry of real patient data in the demo.
4. LLM must only explain model outputs; keep guardrails in `MODEL_CARD` / `GUARDRAILS`.
5. After code changes that touch `predict()` or SHAP handling, re-run the three `EXAMPLES` predictions.
6. Persistence files for progress: update `CHANGELOG.md` / `TEST_HISTORY.md` when applying verified fixes.
7. Keep diffs focused; avoid drive-by refactors and unsolicited markdown unless requested.

## Known fix (2026-10-16)

SHAP 0.51 returns contributions with shape `(1, n_features)`. The previous 2D branch treated the last axis as class and crashed every prediction. Fixed in `predict()`; `call_llm()` now raises a clear error if `HF_TOKEN` is missing.
