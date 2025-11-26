# BAO Anisotropic Fit

- **Location**: `cosmos/fits/bao_aniso/bao_aniso.py`
- **Data**: `data/standardized/bao_aniso.npz` (cached DESI anisotropic BAO release with `z_eff`, `obs`, `cov`, `labels`, and metadata)

## Loader
1. `load_bao_aniso_dataset()` prefers the standardized cache under `data/standardized/bao_aniso.npz` (via `data_interface.bao_loader.load_bao_data()`), emits the flattened observables plus covariance inverse, and keeps the legacy `data/bao_aniso/` loader as a fallback.
2. The loader validates that either `(DM_over_rd, DH_over_rd)` or `(DA_over_rd, H_times_rd)` are present, stacks them in the order defined in the file, and builds a flattened observation vector plus covariance inverse and error vector.
3. The returned dictionary includes `z`, `obs`, `cov`, `inv_cov`, `labels`, `observables`, `values`, and `meta`, keeping everything discoverable for downstream tools.

## Fit evaluation
- `run_bao_aniso_fit(model, dataset=None)` requests `model.sound_horizon()`, the relevant distance/Hubble helpers (`DM`, `DH`, `DA`, `Hubble`), and only those; no curvature or expansion calculations happen inside the fitter.
- The model vector is built bin-by-bin so it matches the flattened `obs`/`cov` order coming out of the loader. For `DM_over_rd` and `DH_over_rd` datasets it predicts `[DM(z)/r_d, DH(z)/r_d, …]`; for `DA_over_rd` plus `H_times_rd` it predicts `[DA(z)/r_d, H(z) r_d / c, …]`.
- The usual χ² formula is used: `χ² = (obs − model)ᵀ C⁻¹ (obs − model)` with fallback to `err` if no covariance matrix is available.
- The extras dictionary returns `{"bao_aniso_model": …}` so callers can inspect the predicted vector for debugging.

## Observables & Model API
- Each dataset declares which observables it carries via the `observables` list. The fitter dispatches to the matching helper rather than hard-coding a particular survey.
- Models must expose the same API used by this fitter:
  - `sound_horizon()` – comoving r_d in Mpc
  - `DM(z)` – transverse comoving distance (with curvature)
  - `DA(z)` – angular diameter distance
  - `DH(z)` – c / H(z) for radial BAO
  - `Hubble(z)` – H(z) in km/s/Mpc (used for `H_times_rd` observables)
- The fitter never integrates distances itself; it simply relies on the model interface and the cached module metadata.

## Integration notes
- The anisotropic loader can be wired into dataset registries the same way `bao_iso` is. The loader’s dictionary follows the PBUF data schema enough that other tools can reuse `z`, `obs`, `cov`, `err`, and `meta`.
- Because the fit only uses the provided interfaces, it works unchanged for both LCDM and PBUF (or any future model that implements the API).
- The new dataset file and loader are easy to extend: drop a new `.npz` under `data/bao_aniso/`, ensure it has the required arrays, and the fitter will adopt the additional bins automatically.
