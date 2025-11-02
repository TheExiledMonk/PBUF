# CLI Switch Reference

This note aggregates every argument exposed by `cli.py`, grouped by top-level command. Use it as a quick lookup when scripting repeated runs or wiring the CLI into automation. All paths are workspace-relative unless noted.

## Top-level commands

```
python cli.py <command> [subcommand] [options]
```

| Command | Purpose |
|---------|---------|
| `dataset` | Download or convert raw datasets. |
| `run` | Execute a single-dataset fit (LCDM, PBUF, or both). |
| `joint` | Run the legacy joint fit across all datasets. |
| `fit` | Launch one of the optimization pipelines (`coord`, `grid`, `run`, `joint-comprehensive`). |
| `report` | Generate analysis reports. |
| `test` | Invoke validation test bundles. |

---

## dataset

### `python cli.py dataset download`

| Flag | Required | Description |
|------|----------|-------------|
| `--name {planck2018_distance_priors, pantheon_sn, bao_boss_dr12, bao_eBOSS, cc_cosmic_chronometers_compilation, rsd_fsigma8_compilation}` | ✓ | Identifies which raw dataset bundle to fetch. |

### `python cli.py dataset convert`

| Flag | Required | Description |
|------|----------|-------------|
| `--source <dir>` | ✓ | Name of the directory under `data/raw/` holding the raw files. |
| `--output <path>` | ✓ | Destination `.npz` file produced by the converter. |
| `--type {sn_pantheon, sn_sh0es, sn, bao_iso, bao_aniso, bao, cc, rsd, cmb, sh0es, auto}` | – | Optional explicit dataset type (auto-detection kicks in when omitted). |

---

## run

`python cli.py run <fit>`

| Argument / Flag | Required | Description |
|-----------------|----------|-------------|
| `fit` positional | ✓ | Dataset selector (`cmb`, `sn`, `sn_pantheon`, `sn_sh0es`, `bao_iso`, `bao_aniso`, `cc`, `rsd`). |
| `--model {lcdm,pbuf,both}` | – | Choose which model(s) to fit (default: both). |
| `--parameters <json>` | – | Parameter overrides applied to the single model specified via `--model`. |
| `--lcdm <json>` | – | LCDM-specific overrides (legacy syntax, kept for backwards compatibility). |
| `--pbuf <json>` | – | PBUF-specific overrides (legacy syntax). |

---

## joint

`python cli.py joint`

| Flag | Required | Description |
|------|----------|-------------|
| `--lcdm <json>` | – | Override LCDM parameters for the joint evaluation. |
| `--pbuf <json>` | – | Override PBUF parameters. |

---

## fit

Top-level entry: `python cli.py fit <subcommand> [options]`

### `fit joint-comprehensive`

| Flag | Default | Description |
|------|---------|-------------|
| `--model {lcdm,pbuf}` | `pbuf` | Model family to optimize jointly. |
| `--datasets <csv or all>` | *(unset)* | Comma-separated dataset list, or `all` to include every registered dataset bundle. |

### `fit run`

| Flag | Required | Description |
|------|----------|-------------|
| `--model {lcdm,pbuf}` | ✓ | Model to fit across every dataset compatible with it. |
| `--lcdm <json>` | – | LCDM parameter overrides (applied only when fitting LCDM). |
| `--pbuf <json>` | – | PBUF parameter overrides. |

### `fit grid`

| Flag | Default | Description |
|------|---------|-------------|
| `--model {lcdm,pbuf,both}` | `both` | Evaluates one or both model families. |
| `--datasets <csv or all>` | *(auto)* | Dataset list (defaults to a base bundle including supernovae). |
| `--include-bao` | off | Adds `bao_iso` and `bao_aniso` to the selection. |
| `--grid-config <path>` | *(unset)* | Path to a JSON grid specification. |
| `--output-dir <path>` | `data/results` | Directory that receives score tables. |
| `--workers <int>` | `1` | Parallel worker count (1 = serial execution). |
| `--tag <string>` | *(unset)* | Optional label inserted into metadata/output filenames. |
| `--refine-top <int>` | `0` | Number of top-ranked cosmologies to refine locally. |
| `--refine-fraction <float>` | `0.05` | ± fractional window per parameter during refinement. |
| `--refine-points <int>` | `3` | Samples per axis inside each refinement cube. |

### `fit coord`

Full documentation lives in [`docs/fit_coord.md`](fit_coord.md). In short, the subcommand exposes:

- dataset selection (`--datasets`, `--include-bao`),
- physics/scan controls (`--phase6a`, `--delta-chi2`, `--skip-second-pass`, `--max-workers`, etc.),
- convergence loop toggles (`--converge`, `--max-cycles`, `--improvement-tol`),
- optional island sampling (`--island-samples`, `--island-delta`, `--island-seed`),
- output configuration (`--output`, `--seed-json`, `--eps0`, `--quiet`, `--no-progress`).

---

## report

`python cli.py report generate`

| Flag | Required | Description |
|------|----------|-------------|
| `--formats <list or all>` | – | Comma-separated formats (`html`, `md`, `pdf`, `json`) or `all` to emit every supported artefact. |

---

## test

`python cli.py test all`

Runs the complete validation suite (currently no additional switches).

---

## Notes

- All JSON override flags expect valid JSON strings (wrap them in single quotes when invoking from the shell).
- When a command family offers subcommands (e.g. `dataset`, `fit`, `report`, `test`), you must specify a subcommand; otherwise the CLI prints contextual help.
- Most commands surface errors as human-readable messages prefixed with `❌`.
