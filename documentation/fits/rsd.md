# Redshift-Space Distortions (RSD)

RSD surveys measure the growth rate of structure through the combination **f(z)σ₈(z)**. In V11 this probe is handled identically for LCDM and PBUF: the fit code only reads the standardized dataset, calls `model.fs8(z)`, and compares to the published covariance. All growth physics (D(a), f(z), σ₈(z)) lives inside the model layer.

## Data layout

- The combined dataset lives inside `data/standardized/rsd.npz`.
- Required fields:
  - `z`: redshifts for each measurement.
  - `obs`: observed fσ₈ values.
  - `cov`: covariance matrix (diagonal with reported variances in the published file).
  - `meta`: survey metadata (name, reference, units, etc.).
- The loader in `cosmos/fits/rsd/rsd.py` calls `ensure_standard_dataset()` so the schema matches every other PBUF fit. The `inv_cov` entry is also cached so RSD χ² calculations stay cheap.

## Model API

The fit module only ever calls:

```python
fs8_model = model.fs8(z)
```

Both LCDM and PBUF expose the following helpers:

```python
D_z = model.growth_factor(z)
f_z = model.growth_rate(z)
fs8 = model.fs8(z)
```

Growth is solved via the model-specific ODE, `D(1)=1` normalization, and a cached table that can be interrogated at any redshift. LCDM exposes a `sigma8_0` parameter (defaulting to Planck’s 0.811) while PBUF derives its amplitude from the same normalization. Fits never access `Ω`, `H(z)`, or `E(a)` directly—each model handles those internals behind `fs8(z)`.

## χ² definition

Let `Δ = fs8_model − fs8_obs`.

- If a covariance matrix `C` is provided (and `inv_cov` is stored):  
  ```
  χ² = Δᵀ C⁻¹ Δ
  ```
- Otherwise, assume the provided errors `σ` describe uncorrelated uncertainties:
  ```
  χ² = Σ (Δ / σ)²
  ```

The fit returns both the scalar χ² and the model vector under `"fs8_model"` so downstream diagnostics can compare predictions.

## Pipeline integration

- Loader: `cosmos/fits/rsd/load_rsd_dataset()` (cached via `cosmos.datasets.get_dataset("rsd")`).
- Fit helper: `cosmos/fits/rsd/run_rsd_fit(model, dataset)` only calls `model.fs8(z)` and compares to the provided obs vector.
- Basin walker: including `"rsd"` in the dataset list causes the engine to add `chi2_rsd` to the total via `chi2_total += chi2_rsd` while reporting the `"fs8_model"` vector in the dataset summaries.
- Sanity layer: `cosmos.optim.sanity.evaluate_candidate(..., datasets=["rsd"])` now runs the RSD χ² after the regular LCDM/PBUF sanity checks.

By keeping the fit module thin and delegating growth calculations entirely to the model layer, RSD becomes a clean probe of the same physics that governs the rest of V11.
