# Planck-Bound Unified Framework (PBUF)

The Planck-Bound Unified Framework (PBUF) is a research pipeline for testing an elastic-spacetime cosmology against key late-time datasets: Pantheon+ supernovae, BAO (isotropic and anisotropic), and Planck 2018 CMB distance priors. This repository bundles data acquisition helpers, fitting scripts, and report generators so the full analysis can be reproduced end-to-end.

## Prerequisites
- Python ≥ 3.9 (tested with CPython on Linux/macOS)
- Git, Make, and a C/C++ compiler toolchain (for scientific Python wheels if binary wheels are unavailable)
- ~5 GB free disk space for downloaded datasets and fit products
- Optional: `virtualenv`/`venv` for an isolated environment

## Installation
```bash
git clone https://github.com/TheExiledMonk/PBUF.git
cd PBUF
python3 -m venv .venv          
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All project modules assume the repository root on `PYTHONPATH`. The `Makefile` and helper scripts set this automatically; if running modules manually, prefix commands with `PYTHONPATH=.`.

## Replicating The Main Findings

### 1. Prepare Input Data
Fetch Pantheon+ and BAO releases, then build the derived data products used in the paper:
```bash
make prepare-data
```
This downloads raw catalogs into `data/supernovae/raw` and `data/bao/raw`, then creates processed versions under `data/supernovae/derived` and `data/bao/derived`. Default releases and options can be changed by overriding the `Makefile` variables (e.g. `make prepare-data FETCH_RELEASE=my-branch`).

### 2. Run Individual Fits (optional sanity check)
To reproduce each experiment separately:
```bash
make fit-sn        # Pantheon+ SN fits for LCDM and PBUF
make fit-bao       # BAO isotropic fits
make fit-bao-ani   # BAO anisotropic fits
make fit-cmb       # Planck 2018 CMB distance-prior fits
make fit-joint     # Combined SN + BAO + CMB calibration
```
Outputs land in `proofs/results/<experiment>/fit_results.json` along with diagnostic plots and tables.

### 3. Full Pipeline In One Step
To replicate the full analysis exactly as in the manuscript (data fetch → all fits → unified report):
```bash
make run-all
```
Expected total runtime on a recent laptop is ~1–2 hours, depending on CPU speed and download latency.

### 4. Generate The Unified Report
The previous step already creates the comparison dashboard at `reports/output/unified_report.html`. Regenerate just the report after making local changes with:
```bash
make report
```
Open the HTML file in a browser to inspect parameter tables, confidence contours, and derived quantities.

### 5. Inspecting Fit Products
- Numerical posteriors and metadata: `proofs/results/**/fit_results.json`
- Intermediate BAO/SN datasets: `data/*/derived`
- Logs and console output: standard output from the `make` commands (redirect to files if desired)

### 6. Cleaning Generated Artifacts
```bash
make clean-all
```
This removes derived datasets, fit outputs, HTML reports, and cached arrays so you can rerun from scratch.

## Running Tests & Linters
Basic regression tests are provided under `tests/`. After installing dev dependencies (same as runtime requirements), run:
```bash
pytest
```
Set `PYTHONPATH=.` if your shell session does not already see the project modules.

## Configuration Notes
- Command-line options for each pipeline are documented via `--help`, e.g. `python pipelines/fit_joint.py --help`.
- Model settings (datasets, priors, default cosmology values) live in `config/`. Edit `config/settings.yml` or `config/datasets.yml` to explore alternative calibrations.
- Recombination method defaults to Planck 2018 (`PLANCK18`). Override with `--recomb` flags when invoking the CMB scripts or by setting `recomb_method` in parameter dictionaries.

## Citation
If you use PBUF in academic work, please cite:
Olesen, F. (2025). *Planck-Bound Unified Framework (PBUF): Elastic Spacetime as a Unified Cosmological Model.* Zenodo. DOI: pending review