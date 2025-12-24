# Results Table Source Map (PBUF V11 draft)

Table: Best-fit parameters (reference run)
- Data: best-fit parameter dictionaries as recorded by the pipeline for each model.
- Reference run identifier: `2025-12-19T101316_unified_joint-4` (jackknife seed 342).
- Artifacts: model best-fit payloads plus the recorded model parameter snapshot (includes derived quantities for PBUF).

Table: chi2 breakdown by dataset (reference run)
- Data: per-dataset chi2 breakdown recorded at the model run directory level.
- Reference run identifier: `2025-12-19T101316_unified_joint-4` (jackknife seed 342).
- Artifacts: per-model chi2 breakdown payloads for the five datasets.

Jackknife summary claims (150 folds)
- Data: jackknife draw lists recorded for each joint run, one run per jackknife seed.
- Run identifiers (one per seed):
  - seed 42: `2025-12-18T051057_unified_joint-1`
  - seed 142: `2025-12-18T103924_unified_joint-2`
  - seed 242: `2025-12-18T160504_unified_joint-3`
  - seed 342: `2025-12-19T101316_unified_joint-4`
  - seed 442: `2025-12-19T044657_unified_joint-5`
- Artifacts: jackknife draw records (chi2_LambdaCDM and chi2_PBUF per fold) and the associated jackknife seed stored in the run config snapshot.
