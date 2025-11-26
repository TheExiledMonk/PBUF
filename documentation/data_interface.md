# Data Interface

The `config/data_interface/` package is the canonical gateway from raw cosmology products to the schema that every PBUF fit consumes. It hides survey-specific layouts, persists standardized caches under `config/data/standardized/`, and exposes helper APIs so χ² evaluators only ever see a single in-memory format. This document summarizes the package layout, per-dataset toolchain, and how to extend it safely.

## Package Map

| Area | Purpose | Key Modules |
| --- | --- | --- |
| Unified loader | Public API for fetching all probes (synthetic or real) and optionally standardizing upfront. | `__init__.py` (`load_all_datasets`, `_load_synthetic`) |
| Schema + utilities | Defines the PBUF Data Object (v1), validates datasets, prints summaries, and converts legacy payloads. | `standardize.py`, `standard_interface.md` |
| Observational loaders | Dataset-specific readers for already-standardized `.npz` caches or CSV catalogs. | `cmb_loader.py`, `sn_loader.py`, `bao_loader.py`, `cc_loader.py`, `rsd_loader.py` |
| Raw→standard converters | Scriptable pipelines that ingest survey releases, validate with `ensure_standard_dataset`, and emit `.npz` caches for loaders. | `*_raw_to_standardized.py` (SN, BAO, CC, RSD) |
| Reference data | Pre-built or user-provided artifacts used by loaders/converters. | `config/data/standardized/*.npz`, `config/data/raw/**` |

## End-to-End Data Flow

1. **Raw ingestion (optional)** – Each `<probe>_raw_to_standardized.py` script knows how to fetch or locate raw releases (e.g., Pantheon+SH0ES text files, SDSS DR16 BAO CSVs), checksums them, and reshapes measurements/covariances into the unified dictionary structure.
2. **Schema enforcement** – Converters call `standardize.ensure_standard_dataset(dataset, <TYPE>)`, which enforces key presence (`name/type/z/obs/err/cov/meta`), dtype/shape consistency, and WL-specific rules. Any violation raises early so failures happen before caching.
3. **Caching** – Standardized outputs are written into `config/data/standardized/<dataset>.npz`. For example, `bao_aniso_raw_to_standardized` populates `bao_aniso_dr16.npz` with interleaved `D_M/r_d, D_H/r_d` values and block-diagonal covariances.
4. **Loading** – Runtime paths (e.g., `load_bao_data`) read the `.npz` payloads, reconstruct numpy arrays and metadata, and re-run `ensure_standard_dataset` for safety. Loaders that read CSVs directly (SN/CC/RSD) use the same schema.
5. **Standardization helpers** – `load_all_datasets(use_real=True, standardize=True)` calls `standardize_all_datasets` so every returned entry fully matches the PBUF object schema, ready for χ² functions in `cosmos/fits/*`.

## Schema + Utilities

- **PBUF Data Object v1** (`standardize.py`) — Dict keys: `name`, `type`, `z`, `obs`, `err`, `cov`, `meta`. `type` must be one of `CMB`, `SN`, `BAO_ISO`, `BAO_ANISO`, `CC`, `RSD`, or `WL` (weak lensing has extra arrays like `theta_bins`, `tomo_pairs`, `n_of_z`).
- **`ensure_standard_dataset`** — Coerces arrays to `float`, verifies shape compatibility (`obs` vs `err`, square covariances), inserts defaults, and dispatches to `_validate_wl_dataset_schema` for WL-specific requirements. Use it whenever a dataset crosses module boundaries.
- **`describe_dataset`** — Quick diagnostic printer for interactive sessions, listing counts, cov shapes, and metadata.
- **Legacy converters** — Helpers such as `convert_bao_to_standard`, `convert_cmb_to_standard`, etc., live in `standardize.py` so legacy loaders can be gradually refactored without rewriting schema glue.

## Dataset Pipelines

### CMB distance priors

- Source: `config/data/standardized/cmb.npz` (Planck 2018 TT,TE,EE+lowE+lensing distance priors).
- Loader: `cmb_loader.load_cmb_priors` reads `obs` (`[R, l_a, θ_*]`), `cov`, reconstructs diagonal `err`, and returns a schema-compliant dict.
- Converter: Typically built offline; `standardize.convert_cmb_to_standard` exists for legacy payloads.

### Supernovae (Pantheon+ / Pantheon+SH0ES)

- Loader: `sn_loader.load_sn_data` scans several CSV fallbacks (Pantheon+ release or internal derived tables). It detects column layouts (`zHD`, `MU_SH0ES`, etc.) and outputs `type="SN"` dictionaries.
- Converter: `sn_raw_to_standardized.sn_raw_to_standardized` ingests the official Pantheon+SH0ES `.dat` + covariance text files, validates ~1700×1700 matrices, symmetrizes/regularizes them, and caches to `data/standardized/sn_pantheon_shoes.npz`.

### BAO (SDSS DR16)

- Loader: `bao_loader` exposes `load_bao_data` (anisotropic) and `load_bao_iso_data` which read cached `.npz` files, handle structured metadata, and insist on schema validation at load time.
- Converter: `bao_raw_to_standardized` scripts combine SDSS DR16 catalogs with cosmology helper utilities; anisotropic conversions interleave `[D_M/r_d, D_H/r_d]` per redshift and stitch per-bin covariance blocks, while isotropic conversions package `D_V/r_d` vectors. Both modules record MD5 hashes and hint at download URLs for provenance.
- Fit integration: `cosmos/fits/bao_aniso` consumes a more targeted cache under `data/bao_aniso/` (`desi_bao_aniso.npz`) that stores `z_eff`, `DM_over_rd`, `DH_over_rd`, and the block-diagonal covariance. The loader there reorders the observables per bin, computes the inverse covariance, and hands the structured dictionary to the model-agnostic χ² evaluator.

### Cosmic chronometers (CC)

- Loader: `cc_loader.load_cc_data` reads simple `Hz(z)` CSV compilations when present.
- Converter: `cc_raw_to_standardized.cc_raw_to_standardized` validates the CSV, manufactures a diagonal covariance from reported uncertainties, and stores `CC_compilation` caches.

### Redshift-space distortions (RSD)

- Loader: `rsd_loader.load_rsd_data` expects `f_sigma8` CSVs and returns `RSD_compilation` dictionaries.
- Converter: `rsd_raw_to_standardized.rsd_raw_to_standardized` mirrors the CC flow: checksum gate, diagonal covariance build, `.npz` caching, and metadata stamping.

### Weak lensing (WL)

- Not yet backed by dedicated loaders in this repo, but `standardize.convert_wl_to_standard` and `_validate_wl_dataset_schema` show the expected structure: `theta_bins`, `data_vector`, full covariance, tomographic bin pairs, `n_of_z` histograms, etc. Use these helpers when integrating shear catalogs.

## Synthetic Datasets & Testing

`load_all_datasets(use_real=False)` produces deterministic mock data spanning all probes (e.g., 3-point BAO vectors, 5-point RSD arrays). The optional `standardize=True` parameter runs every mock through `standardize_all_datasets`, making it easy to write unit tests that mimic real PBUF inputs without large files.

## Working with Standardized Caches

- Cache locations live under `config/data/standardized/`. Existing files include `cmb.npz`, `sn_pantheon_shoes.npz`, `bao_aniso_dr16.npz`, `bao_iso_dr16.npz`, `cc_compilation.npz`, `rsd_compilation.npz`, etc.
- Loader failures typically mean caches do not exist yet; run the corresponding `<probe>_raw_to_standardized.py` script to regenerate them. Each script prints checksum hints and saves metadata (survey name, observable, reference, version) alongside the arrays.
- Caches store `meta` as 0-D object arrays to preserve dictionaries inside `.npz`. Loaders unwrap them before returning.

## Extending the Interface

1. **Add a converter** – Create `<probe>_raw_to_standardized.py`, parse your raw release, call `ensure_standard_dataset`, and cache the result (`np.savez`/`np.savez_compressed`). Provide CLI logging for checksum/version tracking.
2. **Add a loader** – Read from either the standardized cache or a lightweight CSV. Always pipe the result through `ensure_standard_dataset` with the correct `type`.
3. **Register the dataset** – Update `data_interface.__init__.load_all_datasets` and (if needed) `standardize.standardize_all_datasets` so joint fits discover the new probe automatically.
4. **Document + test** – Extend `config/data_interface/standard_interface.md` with any new schema constraints, and add unit tests that call your loader/converter plus `ensure_standard_dataset`.

Following this structure keeps every cosmological probe interchangeable: χ² code in `cosmos/fits/*` only sees the PBUF schema, while the data interface handles messy release formats, caching, and validation.
