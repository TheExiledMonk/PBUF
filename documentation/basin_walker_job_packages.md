# Basin Walker Job Packages

## Goal
Enable the basin walker (unified_joint/engine logic) to emit controller-aware **job packages** instead of doing all work in local threads. Each package becomes a standalone controller job/slice that any worker can execute.

## Motivation

- Maximizes throughput by distributing work across every connected worker slot instead of only the local 20-thread budget.
- Makes progress visible, restartable, and recoverable via the controller dashboard.
- Prevents ultra-long slices that appear stale while calculation is still running.

## Scope

1. **Define package boundaries**
   - Identify natural units such as `seed` ranges, jackknife subsets, or prediction modules.
   - Keep packages short enough to finish in minutes; each becomes a controller slice.

2. **Emit packages instead of threads**
   - Basin walker turns each unit into a config payload (`config/science_runs/...`) or inline override.
   - Submit packages via `POST /controller/jobs` (or `/controller/jobs` from CLI) with `slice_count` matching the degree of parallelism desired per package.

3. **Leverage controller features**
   - Each package is tracked via `JobRecord`, so we get worker alerts, requeue logic, reporting, etc.
   - Controller will assign slices to available slots, respecting core/slot ratios.

4. **Completion guarantees**
   - Jobs only finish when the controller confirms all slices completed.
   - Failed slices requeue and worker alerts surface; dashboard logs show history.

5. **Operational considerations**
   - Packages need metadata to allow the dashboard to trace them back to the basin walker origin.
   - Maintain existing configs for reporting/predictions; job templates should reuse them.

## Outcomes

- A single unified_joint run becomes a set of controller jobs covering fitting, jackknife, and prediction modules.
- Workers across the cluster collaborate on each package without manual orchestration.
- Dashboard shows real-time activity and completion for every package.

## Controller integration notes

- `/controller/jobs` is implemented in `cosmos_control/api.py` and ultimately calls `Controller.submit_job` (`cosmos_control/controller.py`). Each submission persists the config under `data/science_runs/<execution_id>/config.json`, auto-creates the slices via `_create_slices`, and tracks everything through the `JobRecord`/`SliceRecord` dataclasses in `cosmos_control/models.py`.
- Submissions accept either a `config_path` or `config_payload` that can be rehydrated with `ScienceRunConfig.from_path` (`cosmos2/science_runner/config.py`). Job records therefore capture the exact payload that drove the job and can include arbitrary controller-side `metadata` fields.
- Since the basin walker already builds the `ScienceRunConfig` payload for each unified run, it can also emit per-package overrides (seed ranges, jackknife ids, prediction modules) before handing them to the controller. Keeping the payload structure consistent means the existing recorder/recorder helpers continue to work once slices finish.
- Each package should tag the payload with a small metadata block (`metadata["origin"] = "basin_walker"`, `metadata["package_id"]`, `metadata["package_type"]`, `metadata["seed_range"]`, `metadata["jackknife_index"]`, etc.) so the dashboard can correlate controller jobs back to the original unified run.

## Implementation outline

1. Replace the local `run_optimisation` call inside `cosmos2/science_runner/modes/joint.py` with a loop over package generators that produce config payloads per seed range / jackknife / prediction module. These generators can reuse the existing `_build_model_configs` helper along with the `ScienceRunConfig` defaults and inline overrides.
2. For each package, construct a minimal payload that includes `run_name`, `models`, `engine_settings`, `parameter_bounds`, and any custom metadata; then POST it to `/controller/jobs` (or reuse the CLI `cosmos_cli.py control` plumbing) with a `slice_count` set to the desired per-package parallelism (e.g., the `workers` value that was previously driving the multi-threaded sampler).
3. Leverage `cosmos_control/controller.py` so every package becomes a tracked `JobRecord`. The slice descriptors created there can reuse the existing range/parameters convention (see `_create_slices` and `SliceDescriptor.parameters`) to identify the package index, model name, and jackknife slice that the worker is about to evaluate.
4. Update the controller dashboard metadata (plugins, logs, run history) to surface the `metadata` block saved at job submission time; this allows operators to see the basin walker namespace, the intended model, and other contextual hints for each job.
5. Keep the legacy `ScienceRunRecorder`/`RunContext` outputs unchanged by letting each worker still produce the same `history_entries`, `chi2_history`, reports, and predictions, but route them through the job slice results that the controller aggregates.

## Package generator responsibilities

Each generator is responsible for translating a logical unit of work into three things: (a) a minimal `config` payload for the controller, (b) `engine_settings`/`parameters` overrides that limit the work to the chosen seed range/jackknife draw/prediction module, and (c) the small metadata block operators need to trace the package back to the basin walker pipeline.

### Seed batches

- Continue to split `engine_settings.n_batches` (`n_seeds`/`batch_size`) into packages that finish in a few minutes, tagging each payload with the `seed_start`/`seed_end` and the original `workers` budget so downstream slices can scale their parallelism.
- Override `engine_settings.n_batches`, `rng_seed`, and any other sampler knobs so each package only evaluates its allotted subset of seeds. Persist the overridden values in metadata so the dashboard knows what a package actually ran.
- Include `metadata["package_type"] = "seed_batch"` plus `package_index`, `seed_start`, `seed_end`, and `total_batches` so slices can identify where they sit in the full baseline run.

### Jackknife draws

- Use the `JackknifeConfig` to enumerate draws and emit one package per draw. Each payload should still point at the same `runs_name`/`fits` definition but override the jackknife information so the worker only runs the single draw (and not the entire sequence).
- Recommend storing `metadata["jackknife_draw"] = {"index": i, "seed": jackknife_seed, "datasets": datasets, "fraction_removed": config.jackknife.fraction_removed}` plus `package_type = "jackknife_draw"` so the worker, dashboard, and logs can highlight the draw seed and datasets being masked.
- The overrides should contain the numeric seed and a copy of `jackknife.datasets_to_test` so worker-side helpers can rehydrate the draw mask via `JackknifeResampler` before optimization and then clear it afterwards.

### Prediction slices

- Predictions are deterministic post-processing that should run once per model/module combination after the fit finishes. Emit metadata that lists the prediction modules (`metadata["prediction_modules"]`) and mark the package as `package_type = "prediction_batch"`.
- Override any prediction-specific configs (module configs, output directories, etc.) so the worker knows which tables/plots to write. If a prediction module should only run once after all fits, gate submission on the baseline job completing successfully.
- The controller job can reuse the same slices/slice_count pattern because predictions are also embarrassingly parallel; include `metadata["prediction_phase"] = "post_fit"` to keep this visible.

## Metadata expectations and tracing

- Persist a `metadata` block in every payload before the controller receives it (see `_prepare_package_payload`). At minimum include:
  - `origin: "basin_walker"` so the controller UI can filter packages that came from the unified runner.
  - `package_id`, `package_type`, and any package-scoped indexes (seed range, draw index, module name).
  - `run_name`/`models`/`engine` context from the science config so operators can correlate a JobRecord with the basin walker run.
  - `jackknife_draw`/`prediction_modules` sub-objects when relevant to provide rich context to logs/alerts.
- Controller dashboards should display this metadata alongside job progress. The worker compute logs should also prefix new slices with the metadata payload so failure messages can be traced back to a specific package or draw.
- If the controller requeues slices or workers surface alerts, the metadata is the link back to the original basin walker sequence (seed block, jackknife draw, or prediction module).

## Implementation status

- Seed packages now reset the jackknife/prediction flags, override `engine_settings.n_batches`, and carry metadata that records `seed_start`, `seed_end`, `total_batches`, and the `workers` budget so the dashboard knows how each slice scoped its work.
- Jackknife packages iterate `JackknifeConfig.n_draws`, tag each job with `jackknife_draw`/`jackknife_config` metadata, and rely on the controller worker to regenerate the draw mask via `JackknifeResampler` before every slice (the worker clears the mask afterwards so future slices stay clean).
- Prediction packages are emitted per module with a tight `predictions.modules` override and metadata that advertises the module name, config overrides, and `prediction_phase: "post_fit"` so the controller UI can surface it alongside engine jobs. The worker runner reads that metadata to limit the predictions block to the requested modules.
- The controller dashboard and `/system/status` plugin already summarize the metadata we emit, so live jobs now surface their package ids, seed ranges, draw indexes, and prediction modules without extra developer work.
- Both the `cosmos2_science_runner` and legacy `scripts/science_runner.py` CLIs now accept `--controller-endpoint` (or inherit `COSMOS_CONTROLLER_ENDPOINT`/`engine_settings.controller_endpoint`) so running the science runner submits packages to the controller instead of executing them locally.

## Running via controller

1. Start the controller daemon (`cli control start-controller`) and at least one worker (`cli control start-worker`).
2. Execute `cosmos2_science_runner --controller-endpoint http://controller:8080 config.yaml` (or set `COSMOS_CONTROLLER_ENDPOINT`) so the unified runner emits packages instead of running locally.
3. Visit `/system/status` or `/dashboard` to watch each job’s metadata, worker alerts, and slice progress; the metadata we now store is surfaced directly in that view.

## Suggested next steps
1. Define the canonical package types (batch seeds, jackknife groups, predictions) and the maximum runtime/`slice_count` for each so packages stay short and visible on the dashboard.
2. Build the submission helper (e.g., `cosmos2.science_runner.controller_jobs.schedule_package`) that serializes overrides, attaches metadata, and POSTS to `/controller/jobs`.
3. Ensure the unified runner waits for controller confirmation before marking the basin walker run as complete, reusing the job status tracking in `cosmos_control/controller.py`.
