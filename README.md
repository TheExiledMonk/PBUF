# PBUF v11 — The Planck-Bound Unified Framework
**A Λ-free Elastic Spacetime Model for Cosmology**

| Field | Value |
| --- | --- |
| Author | Fabian Olesen |
| Version | v11 (2025) |
| ORCID | 0009-0009-7125-8547 |
| DOI | [https://doi.org/10.5281/zenodo.17507682](https://doi.org/10.5281/zenodo.17507682) |
| GitHub | https://github.com/TheExiledMonk/PBUF |

This repository captures the computation, orchestration, and reporting pipelines that produce the published LCDM/PBUF comparisons. The Cosmos Engine v2.0 stack underneath provides the physical models, dataset loaders, and optimisation primitives while the top-level repo bundles the cosmological models, joint optimisation tooling, and reporting helpers needed to reproduce those results end-to-end.

## System requirements

- **Python** 3.10 or later (3.11+ preferred) with `venv` support.
- **Bash** or a POSIX shell to run the provided helpers (`setup.sh`, `cosmos_cli.py`, `scripts/regenerate_report.py`).
- **Git** to clone and keep the repository in sync with `https://github.com/TheExiledMonk/PBUF`.
- Optional: CUDA-aware tooling if you target GPU-enabled optimisers inside `cosmos2`. The standard runs work on CPU-only machines.

## Environment bootstrap

1. Clone the project and change into the workspace:
   ```bash
   git clone https://github.com/TheExiledMonk/PBUF
   cd PBUF
   ```
2. Run the bootstrap script once to create the isolated environment and install dependencies:
   ```bash
   ./setup.sh
   ```
   `setup.sh` creates `.venv`, activates it, and runs `pip install -r requirements.txt`. If requirements change, rerun `pip install -r requirements.txt` inside the environment to stay up to date.
3. Activate the virtual environment for every new shell session:
   ```bash
   source .venv/bin/activate
   ```
   Inside the activated shell you can run all CLI commands without prefixing `python3` or worrying about global packages.

## Running the science workflow

`cosmos_cli.py` is the gateway for fits, science runs, tooling, and reporting. Prefer invoking the CLI directly so you pick up layered configuration, logging, and the threaded runner that modernises `cosmos2`.

```bash
python cosmos_cli.py science --config config/science_runs/minimal.json
```

This executes the scout/joint stages over `cmb`, `cc`, and `rsd` probes using both LCDM and PBUF models and stores results under `data/science_runs/minimal`. You can swap in any other configuration from `config/science_runs/` to explore alternative priors, dataset subsets, or engine settings. If you ever need to bypass the CLI, run `python scripts/cosmos2_science_runner.py --config <sheet>` to talk to `Cosmos2ScienceRunner` directly.

## Data provisioning for reproducibility

The repository already stores normalized datasets, but if you need to rebuild or refresh them you can fetch raw releases through the toolbox. `config/downloader/datasets.yaml` is the authority for dataset keys and URLs.

```bash
python cosmos_cli.py toolbox data-sync --datasets bao_aniso cc_cosmic_chronometers_compilation pantheon_sn planck2018_distance_priors rsd_fsigma8_compilation
```

The KiDS-1000 cosmic shear release can now be fetched with the `weak_lensing_kids1000` key.
Pass `--dataset-components weak_lensing_kids1000=xi,cov,nz` (or any subset of `xi`, `cov`, `nz`) to control which components are materialized during conversion.

Each dataset writes `data/raw/<dataset>` plus metadata (`source.json`). Once the raw inputs are available, rerun the ingestion scripts (see `toolbox/data_sync.py`) or rerun the science command—the loader will prefer freshly standardized caches from `config/data/standardized/`.

### Planck 2018 raw products

Raw Planck 2018 likelihood files (COM_Likelihood_Data-baseline release) are now governed by the `planck_2018_raw` dataset key. The download URL is placeholder for now, so confirm that you have extracted `COM_Likelihood_Data-baseline` into `data/raw/COM_Likelihood_Data-baseline` before running the CLI. Trigger conversion via:

```bash
python cosmos_cli.py toolbox data-sync --datasets planck_2018_raw
```

Use `--planck-components` to limit the conversion set (e.g. `--planck-components cmb_raw,cmb_masks,cmb_lensing`) if you only need a subset of the NPZ bundles.

## Quantum dataset pipeline

Quantum engine inputs live under `configs/quantum/` and share their own tooling.

```bash
python cosmos_cli.py toolbox quantum-download
python cosmos_cli.py toolbox quantum-ingest --summary logs/quantum_ingest_summary.json --output data/quantum/normalized.csv
python cosmos_cli.py toolbox quantum-compact
```

`quantum-download` mirrors raw alerts, `quantum-ingest` normalizes them into the shared schema, and `quantum-compact` packs the results into `.npz` archives in `data/quantum`. Rerun these commands whenever you update the quantum source material.

## Verifying and regenerating results

Every science run under `data/science_runs/<run_name>` contains:

- `model_comparison_report.html`: the HTML page used for dissemination (see `data/science_runs/full_joint/.../model_comparison_report.html` for an example).
- JSON summaries (`history.json`, `model_summaries.json`, `jackknife_*.json`) plus `logs/<run_name>/` with CLI output, chi² traces, and recorder events.

To rebuild the enhanced jackknife report for an existing run:

```bash
python scripts/regenerate_report.py data/science_runs/<run_name>
```

This script loads `model_summaries.json` and jackknife artifacts, writes a human-readable markdown report, and can be committed if you need to archive a new diagnostics page. For automated regression checks, compare the contents of `model_comparison_report.html` or the recorder logs to published outputs in `data/science_runs/<reference>/`.

## Directory layout highlights

- `cosmos2/`: Cosmological models, fits, and the new science runner infrastructure for the v2 release.
- `config/science_runs/`: Ready-made science configurations (e.g., `minimal.json`) plus any overrides you add for bespoke tests.
- `configs/quantum/`: Defaults and overrides for the quantum ingestion pipeline.
- `toolbox/`: Download/ingest utilities for cosmology and quantum datasets; callable via `python cosmos_cli.py toolbox`.
- `data/`: Raw downloads under `data/raw/`, normalized caches in `data/quantum` and `data/science_runs`, and any exported diagnostics.
- `reporting_system/`: HTML/text report generators and helpers used by CLI reporting commands and `scripts/regenerate_report.py`.
- `documentation/`: Markdown docs covering model layout, data interfaces, optimisation engines, and migration notes.

## Documentation & further reading

- `documentation/data_interface.md` – canonical description of the raw-to-standardized pipeline and schema helpers.
- `documentation/model_registry.md` – how models are registered, documented, and versioned inside Cosmos v2.
- `documentation/quantum_engine.md` – structure of the quantum dataset ingestion and its dependencies on `configs/quantum/`.
- `documentation/migration_to_cosmos2.md` – notes for contributors coming from the legacy Cosmos v1 stack.
- `unified_science_runner.md` – design notes for the new runner architecture and mode contract.

## Citation

Please cite the Zenodo record when referencing this workflow or the accompanying parameters:

> Fabian Olesen. "PBUF v11: The Planck-Bound Unified Framework — A Λ-free Elastic Spacetime Model for Cosmology." Cosmos Engine Project v2.0, 2025. Version v11. DOI: 10.5281/zenodo.17394412. Direct BibTeX is provided on the DOI landing page.

## Support

For questions or collaboration enquiries, open an issue or reach out through the Cosmos Engine Project channels listed in the GitHub profile.
