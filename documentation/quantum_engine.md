# Quantum Engine

The quantum engine integrates the legacy spacetime rigidity solver (ε₀ scan) with the α_QM post-processing codepath. It is implemented under `configs/quantum/` and exposed via `config.quantum.engine.run_quantum_engine`. This document explains how the engine is organized, how data flows through the system, and what configuration/operational hooks exist.

## Code Map

| Area | Purpose | Key Modules |
| --- | --- | --- |
| Entry point + orchestration | Public API that stitches the ε₀ and α subsystems and returns a structured record. | `engine.py` |
| Configuration + paths | Loads YAML/JSON configs, resolves repository-relative paths, and exposes typed views of each config section. | `config.py`, `paths.py` |
| Data discovery | Locates the concrete event file(s) or directory and enumerates matching payloads based on configured patterns. | `data_access.py` |
| ε₀ scanner | Loads & validates events, runs the rigidity scan, and reports credible intervals + metadata. | `e0_runner.py`, `e0/` package |
| α_QM derivation | Generates regulator/field/mixing samples, enforces reproducibility rules, and summarizes α/derived parameters. | `alpha_runner.py`, `core/` |
| Data pipeline utilities | Tools for building normalized multi-messenger datasets before they reach the engine. | `pipeline/` submodules |

## End-to-End Flow

`run_quantum_engine` (`engine.py`) implements the canonical flow:

1. Load configuration via `load_config`, merging `configs/quantum/config/defaults.yaml` with optional user overrides.
2. Resolve the event source with `discover_event_source`. A single file is used directly; otherwise every file matching `data.events_patterns` inside the discovered directory is enumerated.
3. Execute `run_e0_pipeline` to produce ε₀, uncertainty intervals, event counts, and warnings. Event inputs go through `_load_events`/`validate_event` to ensure schema compliance.
4. Feed the resulting ε₀ into `run_alpha_pipeline`. This step enumerates every regulator × field set × mixing sample combination, tests whether each sample lies inside the configured α band, and computes the aggregate α value/error + derived attributes.
5. Collate runtime metadata (timings, configuration used, file paths, per-stage statistics) and return the final `quantum_state` dictionary with `eps0`, `alpha_QM`, their errors, derived parameters, and provenance.

The returned dictionary is self-contained and can be serialized directly for downstream science reports.

## Event Lifecycle

1. **Discovery** – `data_access.discover_event_source` honors explicit `data.events_dir` overrides, otherwise it scans `data.events_search_roots` and honors a best-guess heuristic (preferring `<root>/events` or the first matching event-like file).
2. **Loading** – `e0_runner._load_events` dispatches to the legacy `e0.events.load_event/load_events_from_directory` helpers. JSON, CSV, and NPZ (via `event_storage`) are supported. Errors are downgraded to warnings to maximize usable events.
3. **Validation** – `e0.events.validate_event` enforces the multi-channel schema: positive distance `L_Mpc`, ≥2 channels, per-channel `t_obs/sigma_t/mass_eV` ranges, `E_eV` for massive messengers, and an intrinsic lag model. Invalid records are skipped with warnings, and the engine aborts only if no valid events remain.
4. **Normalization (pre-engine)** – The `pipeline/` tools automate moving from messenger-specific formats into the canonical schema. Highlights:
   - `pipeline.multimessenger_ingest` ingests GWOSC catalogs, GCN/Fermi parses, Super-K tables, etc., normalizes units, and emits consolidated CSVs plus layout reports.
   - `pipeline.event_builder` + `pipeline.event_storage` programmatically assemble and persist event dictionaries, while `pipeline.time_conversion` handles GPS/MET coordination.
   - Loader utilities (`fermi_loader.py`, `gwosc_loader.py`, `messenger_loader.py`) encapsulate source-specific quirks, and `event_matcher.py`/`gcn_indexer.py` help tie alerts back to GW triggers.

Most day-to-day engine runs start with already-normalized JSON/NPZ artifacts produced by the pipeline scripts.

## ε₀ Pipeline Details

The ε₀ stage (`e0_runner.py`) wraps the historical rigidity solver found in `e0/run_fit.py`:

- `_load_events` merges file-by-file payloads into a single list, tracks which filenames were inspected, and reports missing data as warnings to keep the scan resilient.
- `run_e0_pipeline` validates every raw event, dropping failures while appending reasons to the warnings list. It then calls `compute_rigidity`, passing:
  - `eps_range`: from `QuantumEngineConfig.eps_min/eps_max`
  - `steps`: default `500000` from `defaults.yaml`
  - `k_eps`: coupling strength tuning knob
  - `threads`: number of workers to use when scanning (multiprocessing enabled when `threads > 1`).
- `compute_rigidity` builds a linspace over the specified ε range, evaluates `total_loglike` for each point (optionally in parallel), and produces:
  - `best_eps0` from the max loglike sample
  - loglike traces and credible intervals (`lower_68/upper_68/lower_95/upper_95`) computed via normalized trapezoidal integration
  - `n_events` plus `loglikes` arrays for diagnostics.
- `E0Result.stats()` exposes the subset of metadata used by the top-level engine: ranges, step count, event tallies, and best log-likelihood.

If no events survive validation, the runner raises `RuntimeError` so callers can surface “no data” errors early.

## α Pipeline Details

`alpha_runner.py` parameterizes the α_QM search space using `AlphaConfig`:

- `_generate_mixing_samples` creates a log-spaced list between `alpha.mixing_range[0]` and `[1]` with `mixing_samples` points.
- `_build_samples` iterates across every regulator (loop coefficient) and field set (effective degrees of freedom), pairing them with each mixing sample to build `AlphaSample` records. Each sample computes:
  - `f_cut`: √(1 / (loop_coeff × N_eff))
  - `f_coup`: g² / (1 + g²) using the mixing strength `g`
  - `alpha_value`: `f_coup * f_cut⁴ / eps0`
  - A band flag based on `alpha.alpha_band`.
- `_compute_alpha_summary` prefers the geometric mean of in-band samples; falling back to the global distribution triggers a warning. Dispersion becomes the quoted `alpha_error`.
- `_validate_reproducibility` compares in-band mean α values for each regulator under a chosen `reference_field`. Deviations beyond `warnings_threshold` emit warnings rather than failing the run, allowing exploratory scans to proceed.
- `run_alpha_pipeline` packages the aggregate α estimate, preserves per-stage metadata (sample counts, mixing range, regulator/field counts), and surfaces the most representative “reference sample” (fields/regulator/f_cut/f_coup/mixing strength) inside `derived_parameters` for downstream thermodynamic modeling.

## Configuration Surface

`config.py` defines three dataclasses exposed via `QuantumEngineConfig`:

- `DataConfig`: data roots, optional explicit `events_dir`, search roots/patterns, reports/log destinations.
- `AlphaConfig`: α band bounds, mixing range/sample count, regulator coefficients, field sets, reproducibility requirements, target regulator/field for reference samples, and warning thresholds.
- `QuantumEngineConfig`: top-level source tag, ε scan bounds, number of steps, `k_eps`, thread count, and a nested `AlphaConfig`.

`load_config(path)` merges the default YAML with user overrides (YAML or JSON). Relative paths are resolved relative to the repo root via `paths.resolve_path`, keeping configs portable.

Thermal handoff: `regulator`/`field_content`/`thermal_mode` are fixed to `hard_cutoff`/`SM_full`/`exp`. The exponential coefficients (β, T*, power) are derived at runtime from the quantum microphysics via `quantum.thermal.fitter.derive_thermal_params` (a refactor of the `tools/test_quantum_thermal_bridge.py` logic). The config only controls grid/fit bounds: `fit_samples`, `fit_min`, `fit_max`, `fit_points`, `t_min`, and `t_max`.

## Outputs and Reporting

`run_quantum_engine` returns a dictionary shaped as:

```json
{
  "eps0": <float>,
  "eps0_error": <float>,
  "alpha_QM": <float>,
  "alpha_error": <float>,
  "derived_parameters": {
    "regulator": "...",
    "field_set": "...",
    "f_cut": <float>,
    "f_coup": <float>,
    "mixing_strength": <float>
  },
  "source": <config.source>,
  "run_metadata": {
    "config_used": {...},
    "stats": {
      "runtime_seconds": <float>,
      "events": E0Result.stats(),
      "alpha": AlphaResult.metadata,
      ...
    },
    "paths": {
      "event_files": [...],
      "reports_dir": "...",
      ...
    },
    "warnings": [...]
  }
}
```

Warnings from both subsystems are concatenated to ease monitoring. Downstream tooling can persist the full blob as JSON (see `quantum/tools/export_thermal_table.py` for an example consumer).

## Running the Engine

Typical usage from the repo root:

```bash
python - <<'PY'
from config.quantum import run_quantum_engine
result = run_quantum_engine("/path/to/overrides.yaml")  # optional override path
print(result["eps0"], result["alpha_QM"])
PY
```

Notes:
- Custom configs can tweak data locations, ε scan resolution (`e0.steps`), or α sampling knobs. Invalid configs raise `ValueError` early, ensuring runs fail fast.
- The engine only touches files under the resolved `data` roots plus `logs/` & `outputs/`; sandboxing those locations keeps the run reproducible.
- To debug data issues, inspect the `run_metadata["warnings"]` array first—missing files, validation failures, and reproducibility deltas all surface there.

With this overview, contributors can confidently extend the quantum engine, add new messenger loaders, or adjust the ε/α tuning parameters without reverse-engineering the codebase each time.
