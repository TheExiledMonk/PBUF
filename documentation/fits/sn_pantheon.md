# SN Pantheon Fit

- **Location**: `fits/sn/sn_pantheon.py`
- **Data**: `data/standardized/sn_pantheon.npz` (Pantheon+SH0ES distance moduli, full covariance, metadata)

## Loader
1. `load_sn_pantheon_dataset()` extracts `z`, `mu`, the covariance matrix, and its inverse. If `cov` is present we keep both it and the diagonal errors; otherwise we fall back to provided uncertainties.
2. The returned dict follows the PBUF dataset schema (`z`, `obs`, `err`, `cov`, `inv_cov`, `meta`).

## Fit logic
- `run_sn_pantheon_fit(model, dataset=None)` loads the dataset (cached by `cosmos.datasets.get_dataset("sn")` when possible), asks the supplied `model` for `distance_modulus(z)`, and builds the residual vector.
- χ² is evaluated as
  ```text
  χ² = (μ_model − μ_obs)^T C^{-1} (μ_model − μ_obs)
  ```
  using the supplied covariance if available, otherwise the diagonal errors.
- The function returns `(chi2, {"mu_model": mu_model})` so callers can inspect the predicted curve without re-running integrals.

## Model requirements
- The fit code only ever touches `model.distance_modulus(z)` and `model.parameters` (`H0`, `Ω_m0`, etc.). All H(z), curvature, and thermal lookups stay inside the model.
- Any sanities (closure, monotonicity, H>0) are already enforced inside the model before `distance_modulus` runs.

## Integration
- Basin workers (`cosmos/models/*/basin/worker.py`) call this fit helper once per dataset and reuse the cached inverse covariance from the loader.
- The CLI `tools/debug_sn_plot.py` uses the same module to visualize residuals for a candidate model.
