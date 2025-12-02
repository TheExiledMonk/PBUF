# Migrating to `cosmos2`

This repository bundles a drop-in `cosmos2` engine that mirrors the public `cosmos` surface for LCDM/PBUF science runs. Use the notes below to switch imports and flags without touching on-disk datasets or configs.

## Imports and factories
- Replace `from cosmos.factory.model_factory import create_model` (or `cosmos.models.get_model`) with `from cosmos2.models import create_model`. Supported models: `lcdm`, `pbuf` only.
- Swap legacy `from cosmos.datasets import get_dataset` with `from cosmos2.data.registry import get_dataset` when wiring fits to standardized `.npz` inputs.
- For joint χ² evaluators, replace `from cosmos.fits.joint import build_joint_chi2_evaluator` with `from cosmos2.fits.joint import build_joint_chi2_evaluator` and pass the `cosmos2` model factory.
- When instantiating PBUF, pass the thermal table LUT mapping (see `cosmos2.api.engine._load_pbuf_lut` or reuse `cosmos2.pbuf.microphysics.ensure_thermal_table()` and feed arrays into `create_model("pbuf", lut=table_dict)`).

## Science runner and CLI
- Config sheets (JSON/YAML) remain compatible; reuse `run_name`, `models`, `fits_config`/`joint_config`, `engine_settings`, `output`, and weights. Unsupported models (`ede_lcdm`, `running_lambda`, `dgp`, `mg_lcdm`, `desi_mod`) stay legacy-only.
- Use `python cosmos_cli.py science --config <sheet>` as the default gateway (it now dispatches to the threaded `cosmos2` engine). `python scripts/cosmos2_science_runner.py --config <sheet>` remains available directly. Flags:
  - `--override-fits`, `--override-models`, `--mode`, `--engine` (defaults to `cosmos2_basin`), `--monitor <ansi|plugin|textual>` (console updates), `--resume` (read/write `checkpoint.json`).
  - `--config-dir` and `--interactive` mirror the legacy `science_runner.py` UX.
- The legacy `scripts/science_runner.py` wrapper now targets the cosmos2 engine as well, so existing invocations automatically use the threaded runner.
- Programmatic API: `from cosmos2.science_runner import run_science_run` (or `Cosmos2ScienceRunner`) with optional `progress_callback` and `dry_run`.

## Engine selection and monitoring
- `engine_settings["monitor"]` now mirrors `--monitor` and accepts canonical monitor names (`ansi`, `plugin`, `textual`). Set it to a string to enable the better dashboard directly from configuration (e.g., `"monitor": "textual"`).
- For advanced setups, register your own monitor name before the CLI parses arguments using `cosmos2.threads.monitor_types.register_monitor_mode`. Example:
```python
from cosmos2.threads.monitor_types import register_monitor_mode

register_monitor_mode(
    "html",
    "plugin",
    aliases=["web"],
    description="Stream plugin panels to an HTML dashboard.",
)
```
`available_monitor_modes()` drives the CLI choices, so new names show up in `--monitor`/`engine_settings`. Existing monitors (ansi, plugin, textual) live under `cosmos2/threads/monitor_types.py`.
- Checkpoint/resume: set `"resume": true` in `engine_settings` or pass `--resume`; checkpoints are written to `checkpoint.json` alongside outputs and include best params and χ² history.

## Fits and datasets
- Fit registry lives under `cosmos2.fits`; names match legacy (`cmb`, `sn`, `bao_iso`, `bao_aniso`, `cc`, `rsd`, `wl_s8`, `lensing_cross`, `galaxy_pk`, `sh0es`).
- Standardized datasets are loaded from `data/standardized/*.npz` via `cosmos2.data.registry.get_dataset`; no changes to file formats are required.

## Output compatibility
- Outputs mirror `cosmos/science_runner`: `best_fit.json`, `chi2_breakdown.json`, per-fit JSON under `fits/`, `engine_trace.json`, `chi2_history.json`, `run_meta.json`, and `history_entry.json`.
- Reports now come from the standalone `reporting_system` package (`reporting_system.ReportGenerator` / `reporting_system.report_cli`), so there is no reporting surface inside `cosmos2`.
