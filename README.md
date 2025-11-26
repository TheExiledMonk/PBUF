# PBUF v11 — The Planck-Bound Unified Framework
**A Λ-free Elastic Spacetime Model for Cosmology**

| Field | Value |
| --- | --- |
| Author | Fabian Olesen |
| Affiliation | Independent Researcher, Cosmos Engine Project |
| Version | v11 (2025) |
| ORCID | 0009-0009-7125-8547 |
| DOI | [https://doi.org/10.5281/zenodo.17507682](https://doi.org/10.5281/zenodo.17507682) |
| GitHub | https://github.com/TheExiledMonk/PBUF |

This repository captures the computation, orchestration, and reporting pipelines behind **PBUF v11**, the Lambda-free elastic spacetime investigation built on top of *Cosmos Engine v2.0*. Cosmos Engine v2.0 supplies the physical models, dataset loaders, and optimisation primitives that make this workflow possible, and the repo bundles the cosmological models, joint optimization tooling, and reporting helpers needed to reproduce published LCDM/PBUF comparisons.

## Reproducing the science run

1. **Clone the repo** (or pull the latest tags/branches from `https://github.com/TheExiledMonk/PBUF`).
2. **Bootstrap the environment**:
   ```bash
   ./setup.sh
   ```
   This creates `.venv`, activates it, and installs every requirement from `requirements.txt`.
3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```
4. **Run the minimal science configuration**:
   ```bash
   python cosmos_cli.py science --config config/science_runs/minimal.json
   ```
   This executes the scout/joint stages over the `cmb`, `cc`, and `rsd` fits using both LCDM and PBUF models, producing results under `data/science_runs/minimal`.
   `cosmos_cli.py` is the gateway for fits, science runs, and toolbox helpers—prefer it over `python -m <module>` invocations.

## Refetching input data

> The repository already compresses the normalized datasets used by the minimization pipeline. If you need to rebuild them from the raw sources (e.g., for reproducibility or updates), follow the steps below.

### Cosmos datasets
Raw cosmology measurements are downloaded via the toolbox downloader module. Supported dataset keys and URLs live in `config/downloader/datasets.yaml`.

```bash
source .venv/bin/activate
python cosmos_cli.py toolbox data-sync --datasets bao_aniso cc_cosmic_chronometers_compilation pantheon_sn planck2018_distance_priors rsd_fsigma8_compilation
```

Each fetch writes a `data/raw/<dataset>` tree plus `source.json` metadata. Once downloaded, rerun any ingest/conversion scripts that depend on those inputs (see `toolbox/data_sync.py` for helpers); the science runner will automatically pick up the fresh files.

### Quantum datasets
Quantum engine inputs are pulled with:

```bash
python cosmos_cli.py toolbox quantum-download
python cosmos_cli.py toolbox quantum-ingest --summary logs/quantum_ingest_summary.json --output data/quantum/normalized.csv
python cosmos_cli.py toolbox quantum-compact
```

`quantum-download` mirrors the raw alerts, `quantum-ingest` normalizes them into the shared structure, and `quantum-compact` repacks the results into the `.npz` files stored under `data/quantum`. Rerun these steps whenever you need to rebuild the quantum inputs from scratch.

## Directory layout highlights

- `cosmos/`: Cosmological models, fits, and runner infrastructure.
- `cosmos2/models/`: Per-model LCDM/PBUF packages (no shared model code); see `documentation/cosmos2_model_layout.md` for layout notes and removal steps.
- `config/science_runs/`: Ready-made science configurations such as `minimal.json`.
- `configs/quantum/`: Quantum-engine knobs (defaults + overrides) now live here.
- `toolbox/`: Download/ingest utilities exposed via `python cosmos_cli.py toolbox <action>`.
- `data/`: Raw downloads under `data/raw/`, normalized outputs under `data/quantum` and `data/science_runs`.

## Model availability
LCDM and PBUF remain the primary production models.
`ede_lcdm` is now available for Early Dark Energy explorations.
It uses the Hilltop phenomenology and plugs into the existing API.
The new variant keeps existing LCDM/PBUF fits untouched in v11.
Use `ede_lcdm` when you need a low-redshift calibration that decays after recombination.

## Next steps

- Inspect the generated `data/science_runs/minimal` tree for diagnostics and joint-fit outputs.
- Tune `config/science_runs/<other>` or add new fits/priors if you want to explore modified scenarios.
- Run `python cosmos_cli.py toolbox quantum-ingest --output <path>` with custom summaries if you need alternative quantum summaries.

## Citation

Please cite the Zenodo record when referencing this workflow or the accompanying parameters:

> Fabian Olesen. "PBUF v11: The Planck-Bound Unified Framework — A Λ-free Elastic Spacetime Model for Cosmology." Cosmos Engine Project v2.0, 2025. Version v11. DOI: 10.5281/zenodo.17394412. Direct BibTeX is provided on the DOI landing page.

## Support

For questions or collaboration enquiries, open an issue or reach out through the Cosmos Engine Project channels listed in the GitHub profile.
