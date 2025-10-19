# PBUF Cosmology Framework

This repository scaffolds the refactored PBUF cosmology framework with a focus on reproducible fits, modular physics implementation, and automated reporting.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # optional helper if you create one
pip install numpy scipy pandas matplotlib jinja2 pyyaml
```

Or run the helper script (creates `.venv` and installs `requirements.txt`):

```bash
bash scripts/setup_env.sh
source .venv/bin/activate

# Fetch (from PantheonPlusSH0ES/DataRelease) and prepare Pantheon+ data
python pipelines/data/fetch_sn_data.py --source pantheon_plus --out data/supernovae/raw/pantheon_plus --release-tag main
python pipelines/data/prepare_sn_data.py --raw data/supernovae/raw/pantheon_plus --derived data/supernovae/derived --z-prefer z_cmb --compose-cov stat,sys --release-tag PantheonPlusDR1
```

Run the mock supernova fits and build a unified report:

```bash
python pipelines/fit_sn.py --model lcdm
python pipelines/fit_sn.py --model pbuf
python pipelines/generate_unified_report.py --inputs proofs/results/**/fit_results.json --out reports/output/unified_report.html
```

To run against the prepared Pantheon+ dataset instead of the mock sample:

```bash
python pipelines/fit_sn.py --dataset pantheon_plus --model lcdm --out proofs/results
python pipelines/fit_sn.py --dataset pantheon_plus --model pbuf --out proofs/results
```

Outputs live under `proofs/results/<MODEL>/<RUN_ID>/`. Single-fit HTML reports reside in the same run directory. Unified reports are written to `reports/output/unified_report.html` and reference figure assets via absolute file paths so they can be opened locally.

## Repository Highlights

- All physics lives inside `core/`; pipelines are orchestration-only.
- Datasets are declared in `config/datasets.yml` with provenance metadata.
- Diagnostic plots and χ² surfaces are generated through `utils/plotting.py` and `utils/chi2_surface.py`.
- Tests cover the shared model API, pipeline JSON schema, and the SN fitting path.
