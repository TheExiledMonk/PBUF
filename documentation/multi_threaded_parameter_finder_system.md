# Multi-Threaded Parameter Finder System

## Goal
Replace the legacy basin walker with a modular search brain that keeps LCDM and PBUF models strictly isolated. A single `MainFinder` drives the search while a pool of dataset-specific worker threads evaluates proposed points in parallel. Every path stays model-local, quantum execution happens exactly once for PBUF, and the CLI orchestrates the full run.

## Deployment layout
- Configs live under `configs/basin_walker/`:
  - `<model>_bounds.json` defines tight priors per physical parameter (see `configs/basin_walker/lcdm_bounds.json` or `pbuf_bounds.json`).
  - `optimise_config.json` holds execution knobs (`max_iterations`, sampling scale, `threads`, and `dataset_weights`).
- Model-specific basin code stays inside each model tree:
  - `cosmos/models/pbuf/basin/` contains `main_finder.py`, `worker.py`, and `manager.py`.
  - `cosmos/models/lcdm/basin/` mirrors the same structure.
- Results are written to `cosmos/models/<model>/optimisation_result.json`.

## Execution workflow
1. CLI entry: `python -m cli.main optimise --model pbuf --datasets cmb,bao,cc --engine basin --threads 8 --save-result`.
2. `cli.main` loads the appropriate `ModelLoader`, which instantiates either the LCDM or PBUF model and hands it to the `BasinManager`.
3. For PBUF only, the `QuantumEngine` executes once before the main loop and caches whatever thermal/quantum artefacts are required.
4. `BasinManager` loads bounds, config, and bootstraps the `MainFinder` and `WorkerPool`.
5. Workers execute dataset-specific `evaluate_point()` logic inside threads. They instantiate their own model copy and report a `{dataset, chi2, status}` dict; this keeps the models thread-safe.
6. MainFinder governs proposal, validation, aggregation, and the global best-state update loop while `WorkerPool` returns asynchronous results.

## MainFinder strategy
- Sampling phase: the first `initial_samples` iterations use low-discrepancy (Sobol) or Latin-hypercube sampling inside the bounds file so the space is covered systematically.
- Hill-climbing phase: once the seed samples finish, the finder switches to Gaussian steps whose `main_step_scale` decays over time. Every proposal is checked for bound violations before submission.
- Per-iteration flow:
  1. `p = propose_next_point()`.
  2. Reject if `bounds_ok(p)` fails.
  3. `future = pool.submit(evaluate_point, p)`.
  4. Wait for `result = future.result()` (or similar synchronization).
  5. Update `best` if `result.chi2` improves, protecting the shared best record with a lock.
- `best` includes chi², parameters, timestamp, datasets in play, `quantum_hash`, `thermal_table`, and whether `phase6a_passed`.

## Worker design
    - Each dataset gets its own worker class (`WorkerCMB`, `WorkerBAO`, `WorkerSN`, `WorkerSH0ES`, `WorkerCC`, `WorkerJoint`), all inheriting a common base that knows how to load the model and run checks.
- Workers perform:
  - Model-specific sanity checks (`phase6a_check` for PBUF; LCDM flatness, Ω ranges, and `H0` limits).
  - Dataset chi² computation only when sanity passes.
  - A standard return payload: `{"dataset": "...", "chi2": value, "status": "ok"}` (or `status` indicates rejection).
- Dataset weights from `optimise_config.json` apply when computing `chi2_total = Σ weight[d] * chi2[d]`.
- `WorkerPool` uses `concurrent.futures.ThreadPoolExecutor` (or `multiprocessing` when CPU-bound) and is configured via the CLI `--threads` flag and config defaults.

## Validation and sanity layers
- **PBUF phase-6a:** ensure `Ω_σ(a) ≥ 0`, `H'(a) > 0`, `|H''/H'| < 1.2`, `ρ_el/H² < C_max`, and closure `Ω_m0 + Ω_r0 + Ω_k0 + Ωσ(a=1) = 1`.
- **LCDM sanity:** enforce `Ω_k0 = 0`, total density ≈ 1, `H0 ∈ [60,75]`, `Ω_m0 ∈ [0.20,0.40]`, `Ω_b0 ∈ [0.02,0.08]`. Violations skip χ² work and propagate a large penalty.
- Any parameter rejection (bounds violation or physics guardrail) prevents dispatching to workers and the candidate is discarded.

## Configuration sources
- Bounds: `configs/basin_walker/<model>_bounds.json` (see `lcdm_bounds.json` for the structure).
- Optimiser: `configs/basin_walker/optimise_config.json` includes:
  - `max_iterations`
  - `initial_samples`
  - `main_step_scale`
  - `threads`
- `dataset_weights` map per dataset (e.g., `cmb`, `bao_iso`, `bao_aniso`, `sn`, `sh0es`, `cc`, `rsd`, `wl_s8`, `lensing_cross`, `galaxy_pk`). CLI runs can override individual entries via `--dataset-weight <dataset>=<weight>` when calling `cli.py optimise`.
- Combine CLI args (`--threads`, `--datasets`) with the config defaults when instantiating `BasinManager`.

## Output artifact
- `cosmos/models/<model>/optimisation_result.json` contains:
  ```
  {
    "best_chi2": ...,
    "best_parameters": { ... },
    "datasets": ["cmb", ...],
    "quantum_hash": "...",
    "thermal_table": "...",
    "boundaries_used": { ... },
    "phase6a_passed": true/false,
    "timestamp": "..."
  }
  ```
- Every worker run and the `MainFinder` log whether phase-6a/sanity passed so the JSON clearly shows if the result was physically valid.

## Future-proofing
- The architecture already wires dataset-specific workers to the same `MainFinder`, so adding lensing, RSD, WL, joint datasets, or PBUF early-universe tests requires implementing `WorkerX` and plugging it into `WorkerPool`.
- Quantum execution remains a once-per-model step, so future engines can hook into the same `QuantumEngine` stub without restarting expensive calculations.
