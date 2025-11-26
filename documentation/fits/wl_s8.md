# Weak Lensing S₈

We keep weak lensing in V11 as a compressed constraint on the familiar S₈ combination. The fit module never touches the growth ODEs or H(z) — it just reads a standardized dataset, asks the model for `Ωₘ,₀` and `σ₈`, and evaluates the published χ².

## Data layout

- File: `data/standardized/wl_s8.npz`.
- Required keys:
  - `S8_obs`: observed S₈ values (one per survey).
  - `S8_err`: 1σ uncertainties.
  - `gamma`: exponent(s) that encode the survey’s Ωₘ sensitivity (most entries sit near 0.5).
- Optional keys:
  - `cov`: full covariance matrix (N×N). If present the fit uses `C⁻¹`.
  - `labels`: human-readable survey names useful for debugging metadata.
  - `meta`: any extra metadata (the loader stashes the file path plus the provided contents).

The loader in `cosmos/fits/wl/wl_s8.py` coerces each array to floats, ensures their shapes match, inverts any covariance once, and returns a dict that other layers can reuse.

## Model API

WL S₈ is intentionally model-neutral. The only helpers the fit module calls are:

```python
Om = model.omega_m0()
sigma8 = model.sigma8()
```

Both LCDM and PBUF expose these helpers (LCDM can take a `sigma8_0` parameter while PBUF re-derives σ₈ from its cached D(a) solution). The fit module never builds Ω, H(z), or growth itself.

## χ² definition

Given a model prediction

```
S₈_model = σ₈ × (Ωₘ₀ / 0.3)^γ
```

we define `Δ = S₈_model − S₈_obs`.

- If `cov` is available:
  ```
  χ² = Δᵀ C⁻¹ Δ
  ```
- Otherwise:
  ```
  χ² = Σ (Δ / σₛ₈)²
  ```

The fit returns `{"S8_model": S8_model}` so downstream diagnostics can compare the prediction vector.

## Pipeline integration

- Loader: `cosmos/fits/wl/load_wl_s8_dataset()` caches the dataset via `cosmos.datasets.get_dataset("wl_s8")`.
- Fit helper: `cosmos/fits/wl/run_wl_s8_fit(model, dataset)` only asks the model for `omega_m0()` and `sigma8()` and evaluates the χ² above.
- Basin walker: include `"wl_s8"` in the dataset list so the engine instantiates `LCDMWorkerWL`/`PBUFWorkerWL` and does `chi2_total += chi2_wl_s8`, reporting the `"S8_model"` vector in the per-dataset summary.
- Sanity layer: `cosmos.optim.sanity.evaluate_candidate(..., datasets=["wl_s8"])` runs the WL χ² right alongside the other probes, so any invalid model gets the same `HUGE_CHI2` penalty as the rest of the pipeline.

This approach keeps V11 fast—only two scalar calls per evaluation—and leaves room for a full shear power spectrum upgrade in V12.
