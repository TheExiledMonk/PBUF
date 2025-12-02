# NPZ Metadata Expansion for Cosmos2

## Goal
Make each standardized `.npz` cache a self-describing dataset so Cosmos2 can infer:

- dataset type / physics domain,
- fitting pipeline(s) and preferred likelihood backend(s),
- required model backends and variables,
- redshift coverage, observables, and data-vector layout,
- covariance expectations and χ² recipe (descriptive),
- provenance (source release, converter version, commit hash).

The aim is to shift Cosmos2 from a config-driven registry to a dataset-driven system that scales to arbitrary combinations of releases.


## Current flow
- `toolbox/converter.py` converts raw inputs and writes `npz_dict` payloads with minimal metadata (`dataset_type`, timestamps).
- Cosmos2 relies on explicit dataset keys in science configs and registries (`cosmos2/data/registry.py`, `cosmos2/fits/registry.py`) to match loaders → fits → backends.
- Metadata is currently ignored when routing, so every new dataset requires manual wiring in configs/registries.


## Proposed metadata schema
Add a richer `"meta"` dictionary (or companion catalog) per `.npz` that includes the following required fields:

| Key | Purpose |
| --- | --- |
| `dataset_type` | High-level domain (`cmb`, `rsd`, `bao`, `weak_lensing`, `sn`, etc.). |
| `observable` | Canonical observable(s) (`distance_modulus`, `fsig8`, `xi_plus_minus`, `C_ell`, etc.). |
| `redshift_range` | `{ "z_min": float, "z_max": float }`. |
| `fit_modules` | List of fit pipelines the dataset should trigger (`["bao_iso"]`, `["shear_xi"]`, `["cmb_raw"]`). |
| `likelihood_engine` | Identifier for the chosen likelihood backend (Cosmos2 maps the string to code). |
| `backend_calls` | Required model helpers (e.g., `["H(a)", "D(a)", "r_d", "C_l"]`). |
| `data_vector_layout` | Description of `obs` ordering, vector shape, and indexing scheme. |
| `covariance_structure` | `"dense"`, `"diagonal"`, `"block"`, `"none"`, or instructions. |
| `chi2_recipe` | Human-readable χ² composition steps (no code yet). |
| `citation` | DOI/arXiv/survey reference for reporting. |
| `source_release` | e.g., `Planck 2018 (R3.01)`, `KiDS-1000 DR4`. |
| `provenance` | Converter+commit info (`converted_by=cosmos2, commit=abcd1234, date=2025-12-02`).

Optional/extended keys can include:

| Key | Notes |
| --- | --- |
| `tomography_bins` | WL binning metadata. |
| `frequency_channels` | CMB bandpass info. |
| `units` | Explicit units per observable. |
| `parameter_transformations` | Descriptions like `"fσ₈ = f * σ₈"` or scaling hints. |
| `default_masks` | Indices/masks used in the data vector. |
| `model_variables` | Relevant parameters/derived values for the dataset. |


## Behavioral shift
1. Converter (`toolbox/converter.py`) enriches `npz_dict["meta"]` with the schema before saving.
2. Metadata helpers (`cosmos2/data/metadata.py`) load these metas and can answer:
   - dataset type / observable / redshift span,
   - fit modules and likelihood engine,
   - backend calls, data vector layout, covariance structure,
   - χ² recipe hints, citation, and provenance.
3. Fit registry (`cosmos2/fits/registry.py`) is keyed by the identifiers listed under `fit_modules`/`likelihood_engine` instead of dataset name strings.
4. Dataset registry becomes optional: if metadata exists it drives loader/fit selection, reducing manual wiring.
5. Controller can scan metadata, aggregate requested fits/likelihoods, and auto-build composite jobs; compute nodes simply execute the described backends.


## Example metadata
**KiDS-1000 WL**
```jsonc
{
  "dataset_type": "weak_lensing",
  "observable": "xi_plus_minus",
  "redshift_range": { "z_min": 0.1, "z_max": 1.2 },
  "fit_modules": ["shear_xi"],
  "likelihood_engine": "chi2_cov",
  "backend_calls": ["P_kappa", "D(a)", "chi(z)", "growth_factor"],
  "data_vector_layout": "xi+ and xi- stacked by (i,j,theta)",
  "covariance_structure": "dense",
  "chi2_recipe": "chi2 = (d - m)^T C^-1 (d - m)",
  "citation": "KiDS-1000 Collaboration, A&A 645, A104 (2021).",
  "DOI": "doi",
  "source_release": "KiDS-1000 (DR4)",
  "provenance": "converted_by=cosmos2, commit=abcd1234, date=2025-12-02",
  "source": "http link where we got it from"
}
```

**Planck raw CMB**
```jsonc
{
  "dataset_type": "cmb",
  "observable": ["C_ell_TT", "C_ell_TE", "C_ell_EE"],
  "redshift_range": { "z_min": 999, "z_max": 1100 },
  "fit_modules": ["cmb_raw"],
  "likelihood_engine": "custom_cmb_raw",
  "backend_calls": ["C_l", "H(a)", "tau_recombination", "sound_speed"],
  "data_vector_layout": "multipole-indexed arrays for TT/TE/EE",
  "covariance_structure": "block",
  "chi2_recipe": "Apply beam + window, then chi2 = (d - m)^T Cov^-1 (d-m)",
  "citation": "Planck Collaboration 2018 (R3.01).",
  "source_release": "Planck Legacy Archive R3.01",
  "provenance": "cosmos2-converter, commit=efgh5678, 2025-12-02"
}
```


## CLI / user-facing additions
- Add a `cosmos_cli.py dataset-info dataset=<name>` command that prints metadata summaries:
  - dataset type / observable / redshift range,
  - required fit modules / likelihood engine,
  - data vector / covariance hints,
  - citation and provenance.
- This helps users understand large workflows quickly.
 - Each `.npz` represents one full observational dataset; joints simply combine multiple metadata-rich files so the controller can merge their fit modules/likelihoods rather than chaining inside a single archive.


## Future ideas
- Dataset-driven GPU/parallel routing (e.g., WL tomography sharded automatically).
- Metadata validation (required keys, schema version) and propagation into published papers.
- A catalog (JSON/YAML) keyed by dataset name for quick CLI queries without loading binaries.
