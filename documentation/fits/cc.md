# Cosmic Chronometers (CC)

Cosmic chronometers deliver direct measurements of the Hubble parameter via differential galaxy ages. This makes CC a clean expansion probe that can be folded into the V11 pipeline without diving into any cosmology beyond asking the model for `H(z)` at the needed redshifts.

## Data layout

- The canonical CC cache lives at `data/standardized/cc.npz` (the compiled `CC_compilation`), and the loader now reads that file first via `cosmos/fits/cc/load_cc_dataset()`.
- If the standardized cache is unavailable, the loader falls back to any `.npz` files placed under `data/cc/` that expose the same columns.
- Each dataset (whether standardized or legacy) must expose:
  - `z`: 1‑D array of redshifts.
  - `H_obs`: observed H(z) values in km/s/Mpc.
  - `H_err`: 1σ uncertainties (used only when no covariance is provided).
  - `cov` (preferably): covariance matrix matching `H_obs`.
  - `meta`: metadata describing the compilation, survey, reference, etc.
- The `cosmos/fits/cc/cc.py` loader concatenates whatever sources are present, builds a block‑diagonal covariance when multiple files supply covariances, or falls back to diagonal covariances when only errors are available. The metadata records which file(s) contributed to the compilation so diagnostics can trace each source.

## Model API

The fit code only ever calls `model.Hubble(z)` to populate the predicted vector. Both LCDM and PBUF must expose this method:

```python
H_model = model.Hubble(z)
```

Internally the models handle their own parameter sets, expansion laws, curvature, and Phase‑6a checks. They must raise invalidity via the normal sanity machinery when any constraint (H>0, H′>0, curvature bounds, thermal lookup limits, etc.) fails. The CC fit code never inspects `H0`, `Ω_m0`, `Ω_k0`, or any other cosmological quantities directly.

## χ² definition

Let `Δ = H_obs − H_model`.

- If a covariance matrix `C` is provided:

  ```
  χ² = Δᵀ C⁻¹ Δ
  ```

- Otherwise, assume uncorrelated data with errors `σ`:

  ```
  χ² = Σ (Δ / σ)²
  ```

The CC fit returns both `χ²` and the model vector under the `"H_model"` key so downstream diagnostics can compare predictions to observations.

## Pipeline integration

- Dataset loader: `cosmos/fits/cc/load_cc_dataset()` (cached by `cosmos.datasets.get_dataset("cc")`).
- Fit helper: `cosmos/fits/cc/run_cc_fit(model, dataset)` (ringing only `model.Hubble`).
- Basin walker: `cosmos.optim.sanity.evaluate_candidate(..., datasets=["cc"])` adds the CC χ² to the total via `chi2_total += chi2_cc` and reports the `"H_model"` vector in the dataset summary.
- Sanity routing treats `"cc"` the same as other late‑time probes; the fit code never applies priors, Phase‑6a, or any cosmology-specific logic.

Per the V11 guideline, all cosmological logic stays inside the model layer, while CC acts purely as a loader + χ² monitor.
