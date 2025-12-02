# Unified Science Runner

## Motivation

The current `Cosmos2ScienceRunner` directly orchestrates the baseline joint run while jackknife handling has grown organically through separate helpers (`cosmos2/science_runner/jackknife.py`, ad-hoc reporting patches, and todo items). We now need to:

- replace the legacy jackknife implementation with a new design that is rebuilt from scratch and sits inside the unified runner architecture;
- make the science runner emit structured events so that a lightweight monitoring API can query execution progress (per-model steps, draw status, engine progress) across threads;
- keep instrumentation, recording, reporting, and configuration overrides consistent while allowing new modes (regular joint run, jackknife, fit ensembles, synthetic scans) to share the same pipeline.


## Design principles

- **Single orchestration path** – one runner entry point prepares shared state (directories, metadata, config hashes, output registration) and then delegates to a short-lived mode instance so that features such as `RunRecorder`, reporting hooks, CLI switches, and configuration overrides work identically for all modes.
- **Pluggable mode contract** – modes express only the bits that differ (e.g., dataset masking, sequential jackknife draws, synthetic setups) while reusing shared helpers for predictions, recorder updates, and final report artifacts.
- **Explicit context sharing** – a `RunContext` object captures common data (`config`, `run_dir`, `recorder`, `joint_payload`, dataset manifest, baseline model summaries, etc.) and is mutated by both shared runner phases and mode-specific steps.
- **Extendable pipeline stages** – every mode has well-defined `prepare`, `execute`, and `finalize` hooks so we can insert instrumentation, progress reporting, or additional summaries without scattering logic (especially useful for jackknifing which needs baseline + draw-specific phases).

## Architecture sketch

### Event-driven instrumentation

- The unified runner should produce typed events (`RunStarted`, `ModelPrepared`, `EngineProgress`, `JackknifeDraw`, `RunFinished`, etc.) that are enqueued on a thread-safe `EventBus`.
- A supervisory API can consume the event queue to display progress, integrate with dashboards, or expose a streaming endpoint without touching the computational core.
- Each `RunModePlugin` emits its own events through context helpers so that mode-specific steps (baseline training, draw masking, result aggregation) provide transparent visibility to observers.

### Core orchestrator (`UnifiedScienceRunner`)

1. **Entry** – `scripts/science_runner.py` (CLI) still parses configs/flags but now instantiates `UnifiedScienceRunner`.
2. **Shared preparation** – `UnifiedScienceRunner.prepare_run()` builds the run directory, records config+hashes, writes dataset manifests, and resolves the right `RunModePlugin` from a registry keyed by `config.auto_mode` (default “joint”) or CLI overrides.
3. **Mode execution** – the selected plugin receives the shared `RunContext` and runs its own flow, returning structured outputs that the core runner records (model summaries, chi² history, jackknife metrics, etc.).
4. **Finalization** – after a mode completes, `UnifiedScienceRunner.finalize_run()` flushes metadata/history, triggers reports (existing helpers), and exposes results to higher-level callers (e.g., `science_runner.py` or a future API that needs to inspect `analysis_summary`).

### Shared types

- `RunContext`: dataclass with `config: ScienceRunConfig`, `recorder: RunRecorder`, `run_dir: Path`, `timestamp: str`, `joint_payload`, dataset manifest, computed hashes (config, joint, datasets, bounds), `model_summaries`, `history_entries`, optional `jackknife_summary`, and mode-specific metadata bag.
- `ModeResult`: captures final summaries (success flag, history entries, chi² history, optional jackknife section, derived predictions) so the runner can persist them in one place.
- `RunModePlugin` (abstract base class): requires `name`, `prepare(context)`, `execute(context, progress_callback) -> ModeResult`, and `finalize(context, result)` hooks.

## Mode contract

1. `prepare(context)` – called once after shared runner preps directories; allows a mode to fetch datasets, validate jackknife settings, or generate derived config fragments (e.g., `masked_datasets` metadata for jackknife draws).
2. `execute(context, progress_callback)` – runs the heavy lifting for each mode while emitting structured events (`ModelPrepared`, `EngineProgress`, `JackknifeDraw`) through the shared context so simple API consumers can poll run status across threads. Plugins reuse the same helpers for model configuration, optimization execution, and recording of derived artifacts, but they can inject customized dataset preparation, draws, or scanning behavior.
3. `finalize(context, result)` – after execution the plugin can add mode-specific files (e.g., `jackknife_summary.json`, `jackknife_level*_results.json`), update `context.model_summaries`, or emit extra history entries for cross-draw analyses.

## Mode examples

### Joint mode (existing behavior)

- Normalized model config generation and call to `run_optimisation`.
- After optimization the plugin produces predictions, chi² breakdowns, writes per-model `parameters.json`, and records engine traces via the shared recorder.
- Resulting `ModeResult` contains `history_entries`, `chi2_history`, `engine_trace`, `run_meta` values, which the orchestrator persists identically no matter which mode is used.

### Jackknife mode

- Rebuilds jackknife behavior from the ground up (removing the legacy `jackknife.py` helpers) while staying within the shared runner lifecycle.
- Parses a dedicated jackknife section in `ScienceRunConfig`, generates randomized dataset masks/selection strategies, and sequentially runs masked fits through the common optimisation helpers. Each draw writes per-model summaries via the recorder so reporting stacks stay unchanged.
- The mode emits `JackknifeDrawStarted`, `JackknifeDrawFinished`, and `JackknifeAnalysisReady` events to the shared `EventBus`, letting a monitoring API track draw count, success rates, and parameter shifts across the run in real time.
- A final aggregated summary (`jackknife_summary.json`, `jackknife_combined_results.json`) is written by the mode and attached to `ModeResult` so the orchestrator can persist history/metadata identically to other modes.

### Future modes (e.g., data/fit ensembles, synthetic scans)

- Register new plugins that share the same contract. They can opt into registering additional CLI flags through a plugin-specific parser extension or by reading `config.extra`.
- Because shared metadata + recorder usage remain in the core runner, adding a new mode only requires implementing the three hooks plus hooking it into the registry.

## Implementation next steps

1. Create `cosmos2/science_runner/unified_runner.py` housing `UnifiedScienceRunner`, `RunContext`, `ModeResult`, and the new `EventBus` + `RunEvents` helpers so every plugin can emit progress updates safely across threads.
2. Introduce `cosmos2/science_runner/modes/__init__.py` with `BaseModePlugin`, registry logic, and the `JointMode` implementation that reuses existing optimizers.
3. Remove or archive the legacy `cosmos2/science_runner/jackknife.py` and replace it with a fresh `JackknifeMode` under `modes/` that manages dataset masking, draw orchestration, event emission, and aggregated summaries through the shared recorder.
4. Update `scripts/science_runner.py` to select a mode via config/CLI (defaulting to `joint`), exert progress events through the new API, and invoke `UnifiedScienceRunner.execute()` instead of the legacy runner class.
5. Ensure existing report generators, `RunRecorder`, and CLI helpers read from the shared artifacts (`jackknife_summary.json`, `history.json`, event logs) without directly coupling to the new mode internals.

With this design the science runner becomes a single orchestration pipeline that can be extended to additional analyses while preserving recording/reporting consistency.

## Unified configuration example

- The new manifest `config/science_runs/unified_joint.json` serves as a reference for every field the unified runner understands:
  * Top-level keys such as `run_name`, `description`, `models`, `mode`, and `joint_config` pair with `parameter_bounds`, `priors`, and `engine_settings` to describe the shared optimisation context.
  * `output`, `reporting`, and `jackknife` remain available just like before, letting reporters and future jackknife modes read the same metadata.
  * Any unspecified keys are preserved in `ScienceRunConfig.raw` so custom plugins can inspect them if desired.
- Multiple plugins can be declared in the same file under the `plugins` array. Each entry names the `plugin` to execute (e.g., `"joint"` or `"jackknife"`) and may override mode-specific knobs (engine settings, descriptions, custom `jackknife` sections, etc.). The orchestrator iterates through that array—by default it runs the first `plugin` with `auto_mode` set accordingly but future extensions can schedule the entire sequence, reusing the base config plus per-plugin overrides.
- Because all plugins share the same `ScienceRunConfig`, data that is common (fits list, parameter bounds, output paths) only needs to be defined once while plugin-specific behavior stays isolated inside the `plugins` entries. When the runner iterates over the `plugins` array it copies the base config and merges per-plugin overrides before instantiating the corresponding `RunModePlugin`, ensuring a single JSON can describe a complex workflow such as “full joint” followed by a “jackknife validation” pass.
## Additional suggestions

- **Persist event batches** so that retries or dashboard reconnections can replay the exact progress stream and expose what happened before a failure (`RunStarted`, `EngineProgress`, `JackknifeDraw` events).
- **Provide a monitoring endpoint** (CLI status command or lightweight HTTP server) that subscribes to the `EventBus` and surfaces per-model status, current draw, and ETA without touching the compute threads.
- **Add mode-level health checks** in `RunModePlugin.prepare()` → ensure datasets exist, jackknife config is sane, and engine settings are coherent before the heavy lifting starts.
- **Standardize reporting hooks** so each mode can declare optional reporters (JSON, HTML, metrics) whose outputs automatically join the unified metadata, avoiding duplicated serializer logic.
- **Create mode-focused regression tests** that assert both the core runner (history entries, hashes) and event emission behavior, keeping future pluɡins safe while the shared runner evolves.
- **Document plugin onboarding** with a concise “How to add a mode” checklist (registration, CLI flag exposure, event-stream hooks) so contributors don’t reverse-engineer the system.
- **Define observable KPIs** (current draw id, median chi², pending engine steps, event lag) to keep the monitoring API consistent across dashboards or alerts.
- **Consider mode isolation** (per‑mode processes or worker pools) so heavy jackknifing keeps logs/events deterministic even when overlapping with joint or ensemble runs.
