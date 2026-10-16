# Changelog

## 2026-10-16 — Audit fix: SHAP shape handling + LLM guard

### Summary

Full diagnostic audit of `modelo_noshows`. Critical bug blocked all predictions; medium robustness gap in `call_llm`. Both fixed and verified. Persistence files `AGENTS.md`, `CHANGELOG.md`, and `TEST_HISTORY.md` added.

### Files modified

| File | Change |
|------|--------|
| `app.py` | Fix SHAP 2D contribution reshape in `predict()`; guard `call_llm` when `llm_client` is `None` |
| `AGENTS.md` | **Added** — agent role, architecture, operational rules |
| `CHANGELOG.md` | **Added** — this file |
| `TEST_HISTORY.md` | **Added** — audit findings and test results |

### Code locations changed (`app.py`)

1. **`call_llm` (approx. lines 101–104)**  
   - Added: raise `RuntimeError` if `llm_client is None` before calling `chat_completion`.

2. **`predict` SHAP branch (approx. lines 312–326)**  
   - Replaced incorrect `values[:, 1] if values.shape[-1] > 1 else values[0]`  
   - With logic that distinguishes:
     - `(n_samples, n_features)` → `values[0]`
     - `(n_features, n_classes)` → class column
     - `(n_samples, n_features, n_classes)` → unchanged 3D path

### Not changed (by design)

- Model artifacts, feature metadata, Gradio UI layout, requirements pins  
- Low-priority items (e.g. `gitattributes` naming, IDE style warnings)
