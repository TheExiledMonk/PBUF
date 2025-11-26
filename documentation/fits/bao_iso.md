# BAO Isotropic Fit

- **Location**: `cosmos/fits/bao_iso/bao_iso.py`
- **Data**: `data/standardized/bao_iso.npz` (cached D_V/r_d measurements, includes `z`, `Dv_over_rd`, `cov`, and metadata sourced from the DESI release)

## Loader
1. `load_bao_iso_dataset()` prefers the `data/standardized/bao_iso.npz` cache via `data_interface.bao_loader.load_bao_iso_data()`, normalizes the arrays, attaches the covariance inverse, and surfaces the metadata carried over from the legacy standardized cache (fallback paths support older `data/bao_iso/` drops).
2. The returned dictionary already conforms to the PBUF data schema (`type="BAO_ISO"`, `obs`, `err`, `cov`, `inv_cov`, etc.), which keeps the fit-neutral plumbing happy.

## Fit evaluation
- `run_bao_iso_fit(model, dataset=None)` requests `model.DV(z)` and `model.sound_horizon()` (no H(z), Ωʼs, or other cosmology helpers are touched inside the fitter).
- The predicted `D_V(z)/r_d` vector is compared to the observations via the inverse covariance when available, otherwise the provided `sigma` array is used.
- Returns `chi2` plus an extras dictionary containing the model-predicted `DV_over_rd_model` vector for later diagnostics.

## Integration
- The loader is wired into `cosmos.datasets` under the `bao_iso` key, making it available to optimisers and basin workers through `get_dataset("bao_iso")`.
- All models now implement `sound_horizon()` and `DV(z)`, so the fitter never computes distances or r_d itself.
- Basin engines and the standardised sanity runner can safely reference `bao_iso` alongside the existing `cmb`, `sn`, `sh0es`, etc.
