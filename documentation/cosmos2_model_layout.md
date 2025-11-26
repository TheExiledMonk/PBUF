# cosmos2 per-model layout

The cosmos2 refactor keeps each model fully self-contained so LCDM and PBUF can evolve (or be removed) independently.

## Layout at a glance
- `cosmos2/models/lcdm/`: LCDM model surface, parameters, distances, CMB wrappers; depends on the shared `cosmos2/kernels/*`.
- `cosmos2/models/pbuf/`: PBUF-only helpers (elastic background/growth, distances, sanity gates, per-model fit registry `PBUF_FIT_REGISTRY`).
- `cosmos2/pbuf/`: Microphysics bootstrap for PBUF (thermal table LUT export + helpers consumed by the PBUF model).
- `cosmos2/fits/`: Model-agnostic fit evaluators; PBUF-specific ones live alongside the model under `cosmos2/models/pbuf/fits.py`.

LCDM never imports from the PBUF packages; PBUF is self-contained aside from public entrypoints (`cosmos2/fits`, `cosmos2/api/engine`) and the shared thermal microphysics under `cosmos2/pbuf/`.

## Removing PBUF without touching LCDM
Follow this checklist if you need a LCDM-only build:

1. Delete the PBUF payload: `cosmos2/models/pbuf/` and `cosmos2/pbuf/` (plus any PBUF configs you no longer ship).
2. Trim the public surfaces:
   - Drop the PBUF export from `cosmos2/models/__init__.py`.
   - Remove the PBUF branch from `cosmos2/models/model_factory.py` (leave LCDM + legacy-error guard).
   - Collapse `_make_joint_evaluator`/`_evaluate_fit_breakdown` in `cosmos2/api/engine.py` to the LCDM path and delete the `PBUF_FIT_REGISTRY`/`build_pbuf_joint_chi2` imports and `build_pbuf_model_config`.
   - Remove the `PBUF_FIT_REGISTRY` allowance from `cosmos2/science_runner/config.py` and the LUT note in `cosmos2/science_runner/run_reports.py`; update `cosmos2/utils/cpu_affinity.py` if you keep the LCDM-only runner.
3. Clean configs: ensure `config/science_runs/*` and CLI invocations request `lcdm` only; drop any PBUF LUT plumbing that was passed into `create_model`.

Because the two models are isolated, these edits leave the LCDM pipeline intact (fit registry, kernels, runners, and tests remain unchanged).
