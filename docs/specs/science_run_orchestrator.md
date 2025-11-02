# PBUF v10 Science Run Orchestrator

Single entry-point script that automates the PBUF v10 science workflow: parameter tuning with the coordinate basin walker, staged scenario evaluations, robust resume after interruption, strict compute-budget parity across models, complete provenance capture, and minimal artifacts centered on the authoritative best-fit parameters.

---

## 1. Scope & Goals

- Automate the sequence **tuning → scenario runs → joint comparisons** for both ΛCDM and PBUF using the existing CLI (`cli.py fit coord`).
- Support **resume** after crashes or interruption, continuing from the precise last step.
- Enforce identical **compute budgets** across models (workers, grid density, evaluation caps).
- Persist exactly one authoritative **best-fit parameter vector** per `(model × scenario)` plus essential diagnostics.
- Record full **provenance**: dataset versions/hashes, seeds, git commit, CLI command, wall time, CPU-hours, environment versions.
- Produce **joint comparison** artifacts for each scenario (AIC/BIC, χ² partitions, fairness flags).
- Out of scope: implementing new fitters or physics; this orchestrator only orchestrates the existing CLI.

---

## 2. Inputs & Configuration

**Config file:** `configs/science_run.json`

```jsonc
{
  "run_id": "v10_baseline_A",
  "models": ["lcdm", "pbuf"],
  "scenarios": [
    {"id": "geom", "datasets": ["cmb", "bao_iso", "bao_aniso", "cc"]},
    {"id": "sn_rel", "datasets": ["sn_pantheon"], "options": {"sn_mode": "relative"}},
    {"id": "sn_abs", "datasets": ["sn_pantheon"], "options": {"sn_mode": "absolute"}},
    {"id": "geom_plus_sn_rel", "datasets": ["cmb", "bao_iso", "bao_aniso", "cc", "sn_pantheon"], "options": {"sn_mode": "relative"}},
    {"id": "geom_plus_sn_abs", "datasets": ["cmb", "bao_iso", "bao_aniso", "cc", "sn_pantheon"], "options": {"sn_mode": "absolute"}}
  ],
  "budgets": {
    "island_samples": 200,
    "island_delta": 20,
    "workers": 12,
    "eval_cap_per_model": 5000
  },
  "phase6a": {"enabled_for_pbuf": true, "ksat_bounds": [0.5, 1.0]},
  "seeds": {"island_seed": 42, "global_random_seed": 1337},
  "output_root": "data/science_runs/",
  "env_meta": {"machine": "desktop-20c", "notes": "baseline strict"}
}
```

**Parsing rules**

- All fields required except `env_meta`.
- `scenarios` execute in order; every scenario evaluates **both models**.
- `options.sn_mode` may be `"relative"` (default) or `"absolute"`; the orchestrator swaps the underlying dataset (`sn_pantheon` vs `sn_pantheon_abs`) accordingly.

---

## 3. Entry-Point & CLI Integration

- Script: `scripts/run_science.py`
- Invocation:

  ```bash
  python scripts/run_science.py --config configs/science_run.json
  ```

- Internal CLI calls:

  ```bash
  python cli.py fit coord \
    --model {model} \
    --datasets {csv_datasets} \
    --workers {workers} \
    --island-samples {island_samples} \
    --island-delta {island_delta} \
    --island-seed {island_seed} \
    --output {raw_output_path}
  ```

- Pass Phase-6a and `k_sat` bounds when CLI exposes them. Otherwise, validate after the run and mark results in artifacts.

---

## 4. Output Directory Layout

Root directory: `data/science_runs/{timestamp}_{run_id}/`

```
{root}/
  state.json                 # single source of truth for resume
  meta.json                  # environment + dataset hashes + budgets
  logs/                      # per-step logs
  raw/                       # raw CLI outputs (--output)
  artifacts/                 # parsed JSON artifacts (best-fit centric)
  plots/                     # optional QC plots (if generated elsewhere)
```

**Naming conventions**

- Step markers: `{stageOrder}-{scenarioId}-{model}-{status}.json` where `status ∈ {started, done, failed}`.
- Logs: `{stageOrder}-{scenarioId}-{model}.log`.
- Raw outputs mirror `--output` filename under `raw/`.

---

## 5. Stages & Execution Flow

### Stage 0 — Environment Snapshot

- Capture git commit hash, Python + package versions, CPU count, RAM, hostname.
- Record dataset manifests and hashes (reuse dataset registry when available).
- Save to `meta.json`.

### Stage 1 — Optional Single-Dataset Scouts (Gate)

- For each dataset in `["cmb","bao_iso","bao_aniso","cc","sn_pantheon","sn_pantheon_abs"]` and each model, run `cli.py fit coord`.
- Persist best-fit artifact for each run.
- Allow skip via `--skip-scouts` flag.

### Stage 2 — Scenario Runs (Core)

- For each scenario in config order:
  - Run both models with the scenario dataset list.
  - Enforce identical budgets (`workers`, `island_samples`, `island_delta`, `eval_cap_per_model`).
  - Persist best-fit artifact and runtime metrics.
- After both models finish, emit joint comparison artifact.

---

## 6. Resume Logic & `state.json`

`state.json` is the authoritative resume ledger. Example:

```json
{
  "run_id": "v10_baseline_A",
  "timestamp": "2025-10-31T01:23:45Z",
  "scenarios": [
    {
      "id": "geom",
      "models": {
        "lcdm": {
          "status": "done",
          "raw_path": "raw/01-geom-lcdm.json",
          "artifact_path": "artifacts/01-geom-lcdm-done.json",
          "started_at": "...",
          "ended_at": "...",
          "wall_seconds": 1234,
          "cpu_hours": 4.11
        },
        "pbuf": {
          "status": "started",
          "raw_path": "raw/02-geom-pbuf.json",
          "artifact_path": "artifacts/02-geom-pbuf-started.json",
          "started_at": "..."
        }
      },
      "joint": {"status": "pending"}
    }
  ],
  "last_step": "geom:pbuf"
}
```

**Rules**

- On launch, read `state.json`.
- Skip models with `status = done`.
- If `status = started` and missing `ended_at`, rerun and overwrite.
- If `status = failed`, retry once; if the second attempt fails, keep `failed` and skip joint artifact.
- Write `started_at` before spawning CLI.
- After successful parse, write `ended_at`, `wall_seconds`, `cpu_hours`, set `status = done`.
- `last_step` reflects most recent unit of work.

---

## 7. Artifacts & Schemas

### 7.1 Best-Fit Artifact (authoritative)

Path: `artifacts/{order}-{scenarioId}-{model}-done.json`

```jsonc
{
  "run_id": "v10_baseline_A",
  "scenario": "geom_plus_sn_abs",
  "model": "pbuf",
  "datasets": ["cmb","bao_iso","bao_aniso","cc","sn_pantheon_abs"],
  "best_fit": {
    "params": {
      "H0": 67.4,
      "Om0": 0.287,
      "Ok0": 0.0,
      "alpha": 0.043,
      "Rmax": 5000000.0,
      "k_sat": 0.9649,
      "eps0": 0.909768,
      "n_alpha": 0.8,
      "n_eps": -0.5,
      "n_R": 0
    },
    "derived": {
      "rd": 147.1,
      "theta_star": 1.0439,
      "DA_zstar": 13.9
    }
  },
  "fit_stats": {
    "chi2_total": 91.933,
    "chi2_per_dataset": {
      "cmb": 0.792,
      "bao_iso": 5.472,
      "bao_aniso": 62.79,
      "cc": 17.191,
      "sn_pantheon_abs": 746.936
    },
    "dof": 1695,
    "chi2_reduced": 0.54,
    "aic": 1710.3,
    "bic": 1750.8
  },
  "physics_flags": {
    "phase6a_applied": true,
    "ksat_within_bounds": true,
    "sanity_margins": {
      "min_Omega_sigma": 0.0001,
      "max_rho_el_over_H2": 0.07,
      "max_abs_Hpp_over_Hp": 1.02,
      "H_monotonic": true
    }
  },
  "predictives": {
    "sn_abs": {"z": "...", "mu_obs": "...", "mu_model": "...", "residuals": "..."},
    "bao_iso": {"z": "...", "DV_over_rd_obs": "...", "model": "...", "residuals": "..."},
    "bao_aniso": {"z": "...", "DM_over_rd_obs": "...", "DH_over_rd_obs": "...", "model": "..."},
    "cc": {"z": "...", "H_obs": "...", "H_model": "..."},
    "rsd": {"z": "...", "fs8_obs": "...", "fs8_model": "..."}
  },
  "growth": {"D_of_a": "...", "f_of_a": "...", "fs8_of_z": "...", "sigma8_today": 0.78},
  "elastic": {"Omega_sigma_of_z": "...", "rho_el_over_H2_of_z": "...", "S_of_z": "...", "z_turn": 0.5},
  "optimizer": {"trace": "optional", "top_candidates": "optional"},
  "runtime": {"workers": 12, "wall_seconds": 1234.5, "cpu_hours": 4.11},
  "provenance": {
    "git_commit": "abc1234",
    "dataset_hashes": {"pantheon": "...", "bao_dr16": "..."},
    "seeds": {"island_seed": 42, "global_random_seed": 1337},
    "cli": "python cli.py fit coord --model pbuf --datasets ... --workers 12 --island-samples 200 --island-delta 20 --island-seed 42 --output raw/05-geom_plus_sn_abs-pbuf.json"
  }
}
```

- Arrays may be stored as lists; large arrays can be optionally gzip compressed (e.g., `.json.gz`).
- The `best_fit` block is authoritative.

### 7.2 Joint Comparison Artifact

Path: `artifacts/{order}-{scenarioId}-joint-comparison.json`

```json
{
  "run_id": "v10_baseline_A",
  "scenario": "geom_plus_sn_abs",
  "models": {
    "lcdm": {
      "chi2_total": 91.233,
      "aic": 1710.1,
      "bic": 1750.4,
      "dof": 1695,
      "params": {"H0": 68.9, "Om0": 0.274, "Ok0": 0.0},
      "chi2_per_dataset": {"cmb": 0.781, "bao_iso": 5.472, "bao_aniso": 62.79, "cc": 17.191, "sn_pantheon_abs": 746.936},
      "runtime": {"workers": 12, "cpu_hours": 3.85}
    },
    "pbuf": {
      "chi2_total": 91.933,
      "aic": 1710.3,
      "bic": 1750.8,
      "dof": 1693,
      "params": {"H0": 67.4, "Om0": 0.287, "alpha": 0.043, "Rmax": 5000000.0, "k_sat": 0.9649},
      "chi2_per_dataset": {"cmb": 0.792, "bao_iso": 5.472, "bao_aniso": 62.79, "cc": 17.191, "sn_pantheon_abs": 746.936},
      "runtime": {"workers": 12, "cpu_hours": 4.11},
      "phase6a": {"applied": true, "passed": true}
    }
  },
  "parity": {
    "compute_budget_equal": true,
    "dataset_masks_equal": true,
    "priors_equal": true
  },
  "deltas": {"delta_chi2": 0.7, "delta_aic": 0.2, "delta_bic": 0.4},
  "provenance": {"git_commit": "abc1234", "dataset_hashes": {"...": "..."}}
}
```

---

## 8. Compute-Budget Parity & Fairness

- Apply identical `workers`, `island_samples`, `island_delta`, and `eval_cap_per_model` to both models for each scenario.
- Record parity flags in joint artifact: `compute_budget_equal`, `dataset_masks_equal`, `priors_equal`.
- For PBUF, capture Phase-6a status plus margins; for ΛCDM, capture any corresponding sanity checks.

---

## 9. Metrics & Minimalism

- Mandatory artifact fields: `best_fit.params`, `fit_stats` (χ² total & per dataset, DOF, reduced χ², AIC/BIC), `runtime`, `provenance`.
- Recommended: per-dataset `predictives`, `growth` (D, f, fσ₈) traces, PBUF `elastic` diagnostics, `physics_flags`.
- Optional diagnostics: `optimizer.trace`, `top_candidates` (e.g., top 10 within Δχ² ≤ 5–10).
- Keep artifacts compact; focus on best-fit centric data.

---

## 10. Error Handling & Logging

- Capture CLI exit code and stdout/stderr; write to `logs/{order}-{scenario}-{model}.log`.
- On parse failure, save raw CLI output under `raw/`, mark `status = failed` in `state.json`, record short `parse_error`.
- Do not create joint artifact unless both model runs finish with `status = done`.

---

## 11. Time Accounting

- Record `started_at`, `ended_at`, `wall_seconds` for every step.
- Compute `cpu_hours = wall_seconds × workers / 3600` and persist in both state and artifact files.

---

## 12. Acceptance Criteria (Functional Tests)

1. Interrupt mid-run (e.g., during `geom:pbuf`), then relaunch; script resumes at `geom:pbuf` and completes; joint comparison emitted only after both models finish.
2. Rerun without changing config; orchestrator skips completed steps recognized via `state.json`.
3. Each model artifact contains exactly one best-fit parameter vector and required metrics.
4. Joint artifact reports Δχ²/ΔAIC/ΔBIC, parity flags, seeds, commit, dataset hashes.
5. CPU-hours and wall-time fields exist and are plausible for all steps.

---

## 13. Implementation Notes

- Use atomic writes (temp file + rename) for `state.json` and artifacts.
- Keep arrays JSON-serializable; support optional `.json.gz` compression for large arrays.
- Always pass `--output` to CLI and store raw file under `raw/`.
- If Phase-6a or bounds are not CLI options, validate the best-fit post hoc and update `physics_flags`.

---

## 14. Deliverables

- `scripts/run_science.py` — orchestrator entry point.
- `docs/specs/science_run_orchestrator.md` — this specification.
- `configs/science_run.json` — example configuration file.
- `README.md` quickstart updates:
  - Add `python scripts/run_science.py --config configs/science_run.json`.
  - Document resume behavior, output tree, artifact/log locations.

---

## 15. Summary

Run the PBUF v10 science workflow with one command. The orchestrator enforces parity, captures provenance, resumes safely, and emits compact best-fit artifacts plus joint comparisons ready for review and publication.

---

## 16. `joint()` Run Data Capture Checklist

Authoritative checklist for what every joint comparison run must record. Enables reproducible plots, tables, diagnostics, and fair model comparisons.

### 1) Repro & Run Metadata

- Timestamps, wall/runtime per stage and total, CPU-hours, workers.
- Git commit hash; environment hash (Python/NumPy/SciPy versions).
- Dataset manifest hashes and versions (CMB prior type, BAO tables, Pantheon STAT+SYS vs STAT, SH0ES prior, CC compilation, RSD set).
- Joint config: model, parameter bounds/priors, Phase-6a strictness, optimizer settings, random seeds.
- Compute-budget parity flags (equal evaluations per model); early-stop indicators.

### 2) Best-Fit Summary (per model)

- Best-fit parameter vector (with units/meaning).
  - ΛCDM: H₀, Ωₘ₀, Ω_b h² (if present), Ω_k₀, n_s, derived Ω_Λ.
  - PBUF: H₀, Ωₘ₀, α, Rmax, k_sat, (ε, etc.), Ω_k₀.
- Derived anchors at best fit: r_d, D_A(z_*), D_M(z), D_H(z), E(z); Ω_σ(z) for PBUF.
- Physics flags (PBUF Phase-6a): Ω_σ ≥ 0; monotonic H(z); bounded ρ_el/H²; |H″/H′| bounds; growth smoothness.
- Minimum χ²_total, DOF, reduced χ²; per-dataset χ² partitions and percentage contributions.

### 3) Information Criteria & Model Comparison

- AIC, ΔAIC (PBUF – ΛCDM).
- BIC, ΔBIC.
- Optional WAIC / PSIS-LOO estimates with standard errors.
- Likelihood-ratio metrics; Bayes-factor proxy (if priors defined).
- Optional KL divergence between posterior predictives.

### 4) Dataset-Level Predictives & Residuals

- SN (Pantheon): z, μ_obs, μ_model, residuals Δμ, normalized residuals, cov-weighted residuals (C⁻¹Δ), nuisance offsets, mask, binned residual RMS vs z.
- SH0ES: prior term, χ²/−2lnL contribution, posterior mean/σ for H₀.
- BAO iso: z, (D_V/r_d)_obs, model, residuals, per-point χ², AP scaling factors.
- BAO aniso: z, (D_M/r_d)_obs, (D_H/r_d)_obs, model predictions, residuals, per-point χ².
- CMB priors: R, ℓ_A, 100θ*, Ω_b h² (if included), model values, residuals, prior type (Planck18, etc.).
- Cosmic chronometers: z, H_obs, H_model, residuals, per-point χ².
- RSD: z, (fσ₈)_obs, (fσ₈)_model, residuals, per-point χ²; note AP corrections and σ₈ convention (PBUF: σ₈ ≡ D(1)).

### 5) Geometry & Growth Traces

- H(z) vs z, E(z) vs z.
- D_C/D_M, D_A, D_L vs z.
- Growth: D(a), f(a) = d ln D / d ln a, fσ₈(z), σ₈(a); report D(1) (PBUF σ₈).
- Sensitivity: ± small parameter steps with Δχ² profile slices.

### 6) Elastic-Sector Diagnostics (PBUF)

- Ω_σ(z) curve; ρ_el/H² vs z.
- Saturation envelope S(z; k_sat) and effective k_eff.
- Turn-on indicators (e.g., z_turn from Rmax) and late-time asymptote.
- Phase-6a margins: min(Ω_σ), max(ρ_el/H²), max(|H″/H′|), monotonicity checks.

### 7) Optimizer / Search Trace

- Parameter path of the coordinate/basin walker (tested points + χ²).
- Finalist list (top-N) with parameters and per-dataset χ².
- Boundary and edge hits per parameter; Phase-6a rejections vs acceptances.
- Convergence flags; tolerance reached; any re-tests after parameter changes.

### 8) Uncertainty, Sensitivity & Influence

- Approximate covariance/Fisher matrix near best fit (numerical Hessian).
- 1D/2D profile likelihoods for key pairs (H₀–Ωₘ₀, α–k_sat, Rmax–k_sat).
- Pointwise influence (jackknife/LOO Δχ²_i).
- Survey-level jackknife (per-survey χ² and parameter shifts).
- Outlier list: top residuals per dataset with IDs.

### 9) Fairness & Calibration Notes

- Identify datasets calibrated to ΛCDM (BAO compressions, RSD); record AP corrections per model.
- Priors (Gaussian/flat) per parameter; prior contributions to −2lnL.
- Parity notes: PBUF Phase-6a status, ΛCDM sanity checks.

### 10) Performance Metrics (beyond χ²)

- Goodness: reduced χ²; per-dataset χ² density; posterior predictive p-values.
- Parsimony: AIC/BIC penalty terms; k_eff (from WAIC/LOO if available).
- Stability: sensitivity of best fit to dataset removal (Δparams, Δχ² LOO).
- Robustness: variability across seeds/worker partitions; dispersion of top-N.
- Computational: evaluations per second; cost per accepted model; Phase-6a rejection rate.

### 11) Plot-Ready Bundles

- SN Hubble diagram and residuals for both models.
- CMB prior residual bars (R, ℓ_A, 100θ*).
- BAO iso/aniso overlays with ratio plots.
- Cosmic chronometer H(z) overlays and residuals.
- RSD fσ₈ overlays with calibration notes.
- Corner plots for top-N (or profile slices).
- Δχ² bar stacks by dataset (both models).
- Elastic diagnostics: Ω_σ(z), ρ_el/H², S(z).
- Runtime/profile bars for Methods documentation.

### 12) Minimal JSON Keys (self-contained runs)

- `run_meta` (hashes, versions, budgets, seeds).
- `best_fit` (parameters, derived values, χ² partitions, flags).
- `predictives` (arrays per dataset).
- `diagnostics` (Phase-6a margins, influence, profiles).
- `optimizer_trace` (tested points and results).
- `fairness_notes` (calibration/AP details).
- `performance` (AIC/BIC/WAIC/LOO, stability/robustness, runtime).

> Always run `cli.py fit coord` with the same datasets used in later joint evaluations; verify that joint fits use parameters from `parameter_defaults.py`.
