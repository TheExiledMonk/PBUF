# Supernova Data Integration

## Sources

- **Pantheon+** distance modulus vector and covariance components.
- **SH0ES** calibrator catalog (future integration placeholder).

All raw files are stored under `data/supernovae/raw/` exactly as fetched. The
fetcher (`pipelines/data/fetch_sn_data.py`) records URL, timestamp, size, and
SHA256 for each download in `fetched.json`.

## Preparation Pipeline

1. Run `prepare_sn_data.py` against a raw directory. The CLI never edits the
   raw files; it only reads them.
2. The preparer emits canonical, read-only artefacts in
   `data/supernovae/derived/`:
   - `supernova_index.csv` / `.parquet`: one supernova per row with standard
     column names (see schema below).
   - `supernova_index.cov.npy`: covariance matrix composed from requested
     components (e.g., stat+sys).
   - `supernova_index.meta.json`: provenance manifest including hashes for both
     raw inputs and derived outputs, the chosen `z` column, transform version,
     and whether symmetrisation was applied.

## Canonical Schema

Columns in `supernova_index.csv`:

| column        | dtype   | description                                 |
|---------------|---------|---------------------------------------------|
| `sn_id`       | string  | unique supernova identifier                  |
| `z_helio`     | float64 | heliocentric redshift (if provided)          |
| `z_cmb`       | float64 | CMB-frame redshift (preferred when present)  |
| `mu`          | float64 | observed distance modulus (verbatim)         |
| `sigma_mu`    | float64 | 1σ per-object uncertainty                    |
| `sample_flags`| string  | quality flags / sample membership            |
| `host_mstellar`| float64| log₁₀ host stellar mass                      |
| `ra_deg`      | float64 | right ascension in degrees                   |
| `dec_deg`     | float64 | declination in degrees                       |
| `source`      | string  | literal `pantheon+`                          |
| `release_tag` | string  | upstream release tag                         |
| `row_index`   | int32   | stable ordering index aligned with covariance|

No smoothing, clipping, or recalibration is performed: μ, redshifts, and
covariance entries remain exactly as published.

## Provenance Guarantees

- Every raw and derived file is checksummed (SHA256) and recorded in the
  manifest.
- Symmetrisation is logged when near-symmetric covariances are homogenised via
  `(C + Cᵀ)/2`.
- The manifest stores the preferred redshift column (`z_prefer`) and which
  covariance components were composed so downstream fits can reproduce the
  pipeline choices exactly.

## Loader Contract

`dataio.loaders.load_sn_pantheon()` reads the prepared artefacts and returns a
dictionary containing `z`, `mu`, `sigma_mu`, `cov`, `index_map`, `tags`, and a
`meta` block with provenance fields. Pipelines can switch between `z_cmb` and
`z_helio` in their own configuration; the loader merely reflects the choice
recorded in the manifest.

## Usage Cheatsheet

```bash
# Fetch (defaults to the "main" branch of PantheonPlusSH0ES/DataRelease)
python pipelines/data/fetch_sn_data.py --source pantheon_plus --out data/supernovae/raw/pantheon_plus --release-tag main

# Offline/dev mode (copies bundled fixtures instead of downloading ~70 MB of data)
python pipelines/data/fetch_sn_data.py --source pantheon_plus --out data/supernovae/raw/pantheon_plus --use-sample

# Prepare
python pipelines/data/prepare_sn_data.py \
  --raw data/supernovae/raw/pantheon_plus \
  --derived data/supernovae/derived \
  --z-prefer z_cmb \
  --compose-cov stat,sys \
  --release-tag PantheonPlusDR1

# Fit (existing pipeline)
python pipelines/fit_sn.py --dataset pantheon_plus --model lcdm --out proofs/results
```
