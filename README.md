# PBUF v10: The Planck-Bound Unified Framework  
**A Λ-free Elastic Spacetime Model for Cosmology**

| Field | Value |
| --- | --- |
| Author | Fabian Olesen |
| Affiliation | Independent Researcher, Cosmos Engine Project |
| Version | v10 Baseline A (2025) |
| ORCID | 0009-0009-7125-8547 |
| DOI | https://doi.org/10.5281/zenodo.17394412 |
| GitHub | https://github.com/TheExiledMonk/PBUF |

This repository hosts the automation and reporting pipeline used to reproduce the cosmological analyses behind **PBUF v10: The Planck-Bound Unified Framework**. The tooling sits on top of the **Cosmos Engine v1.0 (beta)** runtime and orchestrates coordinated ΛCDM/PBUF comparisons, provenance capture, and publication-ready reporting.

---

## 🚀 One-Command Reproduction

To replicate the PBUF v10 results (fresh virtual environment, science run, and reports) execute:

```bash
./scripts/science_run_oneclick.sh
```

What it does:
1. Creates/refreshes a virtual environment (`.venv_science`).
2. Installs `requirements.txt`.
3. Launches the unified science runner via `python cli.py run science`.
4. Builds the full report suite with `python cli.py report generate`.

Outputs are written under `data/science_runs/` and `reports/output/`.

---

## 📚 Project Overview

- **Cosmos Engine v1.0 beta** supplies the physical models, dataset loaders, and optimisation primitives.
- **PBUF v10 orchestration** (`scripts/run_science.py`, `configs/science_run.json`) enforces a two-stage workflow:  
  _Stage 1_: single-dataset “scout” fits for diagnostics.  
  _Stage 2_: scenario bundles covering relative/absolute SN modes, geometry-only combinations, and joint fits.  
- **Reporting stack** (`reports/report_pipeline.py`) yields interactive HTML, Markdown, and JSON artifacts summarising every run.

The full design is documented in `docs/specs/science_run_orchestrator.md`.

---

## 🧭 Manual Setup (if you prefer step-by-step)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure the standardised datasets referenced by the science configuration exist under `data/standardized/` (run the converters in `data_interface/` if needed).

---

## 🔄 Running the Science Orchestrator

### CLI-first workflow

```bash
python cli.py run science \
  --config configs/science_run.json \
  --fresh                     # optional: force a new timestamped run directory
  --science-root data/science_runs  # optional override
```

Key flags:
- `--fresh` — ignore existing checkpoints and start clean.
- `--resume-dir PATH` — pick up an incomplete run manually.
- `--skip-scouts` — bypass Stage 1 diagnostics.
- `--quiet-cli`, `--no-progress` — forward logging controls to the coordinate walker.

### Direct script entry (legacy)

```bash
python scripts/run_science.py --config configs/science_run.json
```

Both entry points record progress in `state.json`, capture environment metadata in `meta.json`, and generate per-step artifacts + logs under the timestamped run directory.

---

## 📁 Output Layout

```
data/science_runs/{timestamp}_{run_id}/
  meta.json        # environment snapshot (git, packages, dataset hashes)
  state.json       # authoritative resume ledger
  logs/            # stdout/stderr for every CLI call
  raw/             # coordinate walker outputs
  artifacts/       # best-fit and joint comparison summaries
reports/output/
  report.html      # interactive dashboard
  summary.md       # publication-ready tables
  results.json     # machine-readable bundle
  plots/           # generated figures (H(z), μ(z), BAO, RSD, etc.)
```

Joint comparison artifacts (`*-joint-comparison.json`) provide Δχ², ΔAIC, and ΔBIC between ΛCDM and PBUF with parity checks, mirroring the results cited in the v10 manuscript.

---

## 🧪 Inspecting and Extending Runs

1. Check `state.json` to confirm each step reports `"status": "done"` with wall time and CPU-hours.
2. Dive into `artifacts/{order}-{scenario}-{model}-done.json` for per-dataset χ², parameter vectors, and provenance records.
3. Joint artifacts provide consolidated statistics once both models finish a scenario.
4. To rerun or widen parameter bounds, edit `configs/science_run.json` (e.g., `budgets`, `targets`, custom scenario options) and re-launch via the CLI.

---

## 📰 Reporting Pipeline

The reporting stage is reusable on its own:

```bash
python cli.py report generate --output reports/output
```

Internally this calls `reports/report_pipeline.py::build_full_report`, which:
1. Collects all science runs under `data/science_runs/`.
2. Aggregates χ²/AIC/BIC statistics per dataset and model.
3. Generates plots via `reports/plotter.py`.
4. Emits HTML/Markdown/JSON/PDF (PDF optional) summaries.

---

## 🤝 Citation

Please cite the Zenodo record when referencing this workflow or the accompanying parameters:

```
Fabian Olesen. "PBUF v10: The Planck-Bound Unified Framework —
A Λ-free Elastic Spacetime Model for Cosmology." Cosmos Engine Project, 2025.
Version v10 Baseline A. DOI: 10.5281/zenodo.17394412.
```

Direct BibTeX is provided on the DOI landing page.

---

## 🔗 Additional Resources

- Cosmos Engine v1.0 beta release notes — see the main Cosmos Engine repository for APIs and validation suites.
- `README_PBUF.md` — extended physics discussion and prior guardrails.
- `docs/fit_coord.md` — coordinate basin walker options.

For questions or collaboration enquiries, open an issue or reach out through the Cosmos Engine Project channels listed in the GitHub profile.
