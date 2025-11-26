## Galaxy Power Spectrum (Compressed P(k))

Galaxy power spectrum inputs live in the compressed P(k) format described in the V11 roadmap:
each redshift contributes a short data vector (fs8, geometric ratios, etc.) together with a covariance.
The `cosmos.fits.galaxy_pk` helpers keep the fit module cosmology-agnostic and push every cosmology calculation
through the standard model API (`fs8`, `DM`, `Hubble`, `DH`).

### Data layout

```
data/standardized/galaxy_pks.npz
```

The dataset must expose:

- `z` – 1D array of redshifts (one row per compressed point).
- `obs` – (n_points, n_observables) matrix of compressed measurements.
- `cov` – square covariance matrix whose size equals `obs.size`.
- `labels` (optional) – a list of strings identifying each observable column. When omitted, the loader
  falls back to `obs_0`, `obs_1`, ….
- `fiducials` (optional) – dict containing per-row fiducial arrays (e.g. `DM`, `H`, `DH`). The loader
  also scans for keys such as `DM_fid`, `H_fid`, etc. to auto-populate the fiducials map.
- `meta`, `name`, `type` – metadata fields that follow the PBUF standard (the loader defaults `type` to `GALAXY_PK`).

The loader normalizes the inputs, validates shapes, and inverts the covariance matrix for reuse.
Supported observables include:

- `fs8`, `fσ8`, `fsigma8`, etc. → `model.fs8(z)`
- `D_M/D_M^fid`, `DM_over_fid`, … → `model.DM(z) / DM_fid`
- `H/H^fid`, `H_over_fid`, … → `model.Hubble(z) / H_fid`
- `D_H/D_H^fid`, `DH_over_fid`, … → `model.DH(z) / DH_fid`

Unknown labels raise a clear `ValueError`, keeping the fit layer pure χ² evaluation with no embedded cosmology.

### Fit API

| Function | Description |
| --- | --- |
| `load_galaxy_pk_dataset(path=None)` | Reads `galaxy_pks.npz`, standardizes the schema, and returns the normalized dataset dict. |
| `run_galaxy_pk_fit(model, dataset=None)` | Builds the model prediction vector (fs8, DM/H ratios, …), compares it to the observation vector, and returns `(chi2, {"galaxy_pk_model_vector": ...})`. The routine **never** computes growth, σ₈, distances, or H(z) manually. |

The fit always flattens the `(n_points, n_observables)` matrices into a single residual vector before computing χ².
If the dataset lacks `cov`, a flattened `err` array may be supplied instead.

### Integration and Sanity

- The new loader is registered in `cosmos.datasets.get_dataset("galaxy_pk")`, so optimisation helpers, scripts, and CLI
  tools can request the dataset by name.
- `cosmos.optim.sanity.evaluate_candidate(..., datasets=["galaxy_pk"])` now includes the compressed P(k) χ² in the total score and
  respects the existing sanity guardrails: an insane model returns `HUGE_CHI2` before the dataset is evaluated.
- The fit layer never mutates the model, never depends on bias modeling, and never works with fiducials beyond simple division (DM_over_fid, etc.).

### Testing

- `tests/fits/test_galaxy_pk.py` ensures:
  - The dataset loader validates shapes and fiducials.
  - The χ² fit changes when model parameters vary (LCDM and PBUF).
  - Missing fiducials or unknown observable labels raise errors.
  - `evaluate_candidate` returns `HUGE_CHI2` when the LCDM sanity checks fail.
