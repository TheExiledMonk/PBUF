# SH0ES H₀ Prior

- **Location**: `fits/sh0es/sh0es_prior.py`
- **Data**: `data/standardized/sh0es.npz` (Gaussian prior from the SH0ES ladder, includes `obs`, `cov`, and metadata)

## Loader
1. `load_sh0es_dataset()` unwraps the single H₀ measurement, computes σ from the covariance or provided `err`, and keeps the metadata.
2. The returned dict keeps `obs` as a scalar, `sigma`, and an explicit `type` (`"SH0ES"`).

## Prior evaluation
- `run_sh0es_prior(model, dataset=None)` reads `model.parameters["H0"]`, compares it to the SH0ES mean, and returns
  
  ```text
  χ² = ((H₀_model − H₀_SH0ES) / σ_SH0ES)²
  ```

- No cosmology math happens here: the fitter simply reads the model parameter dictionary and reports the scalar χ² plus a small extras dict containing the current and target H₀.

## Integration
- Basin workers and the sanity layer call this helper when `sh0es` is one of the requested datasets.
- The model’s `parameters` property exposes all required entries (H₀, Ω_m, …) so any future probes needing parameter-level priors can follow the same pattern.
