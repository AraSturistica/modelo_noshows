# Test history

## 2026-10-16 — Post-fix verification

**Environment:** conda `ModelNoShows` · Python 3.12.15  
**Interpreter:** `C:\Users\Sturistica\anaconda3\envs\ModelNoShows\python.exe`  
**Installed packages (matched `requirements.txt`):** gradio 6.28.0, xgboost 3.4.0, shap 0.51.0, pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, matplotlib 3.11.2, huggingface_hub 1.33.0

### Pre-fix failure (reproduced)

```
ValueError: Unexpected SHAP output shape (1, 20); expected one contribution per 20 feature
```

Cause: SHAP `TreeExplainer` returned `(1, 20)`; code treated axis `-1` as class dimension.

### Tests run after fix

| # | Test | Result |
|---|------|--------|
| 1 | Import `app` + load model / explainer | PASS |
| 2 | `call_llm(...)` without `HF_TOKEN` raises `RuntimeError` | PASS |
| 3 | `predict(*EXAMPLES[0])` — same-day reliable | PASS · risk **4%** · **Low** |
| 4 | `predict(*EXAMPLES[1])` — long lead + prior no-shows | PASS · risk **51%** · **High** |
| 5 | `predict(*EXAMPLES[2])` — older regular attender | PASS · risk **22%** · **Medium** |
| 6 | Risk card HTML + matplotlib figure returned | PASS |
| 7 | `ctx` has 6 SHAP factors; AI outputs cleared (`""`, `[]`) | PASS |
| 8 | `staff_summary` / `patient_reminder` fallbacks without token | PASS |
| 9 | PyCharm `get_file_problems` errors-only on `app.py` | PASS · 0 errors |

### Audit findings (reference)

| Severity | Issue | Status |
|----------|-------|--------|
| Critical | SHAP 2D reshape in `predict()` | **Fixed** |
| Medium | `call_llm` used `llm_client` without None check | **Fixed** |
| Low | IDE format/shadowing warnings | Deferred |
| Low | `gitattributes` vs `.gitattributes` | Deferred |
| Info | No automated test suite in repo | Noted |
| Info | LLM disabled locally without `HF_TOKEN` | Expected |

### Command used for verification

```powershell
& "C:\Users\Sturistica\anaconda3\envs\ModelNoShows\python.exe" -c "<import app; run EXAMPLES + call_llm guard>"
```

Outcome: `ALL_TESTS_PASSED`
