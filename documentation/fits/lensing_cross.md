# Lensing Cross-Correlations

Compressed, model-neutral constraints on lensing cross-amplitudes keep V11 fast while still forcing Ωₘ/σ₈-enabled models to respect the observed strength of CMB lensing × galaxies, WL × galaxies, and similar cross probes. The supplied `lensing_cross.npz` file currently carries four mock anchors (DES, KiDS, HSC, and ACT cross-galaxy) so we can exercise the multi-measurement workflow without waiting for the full V12 spectra.

## Data layout

- File: `data/standardized/lensing_cross.npz`.
- Required keys and shapes (N = number of datasets):
  - `A_obs`: published amplitude(s) (typically around unity).
  - `A_err`: 1σ uncertainty per measurement (used only if no covariance is supplied).
  - `z_eff`: effective redshift for each measurement.
  - `p_exponent` / `q_exponent`: exponents that encode the survey sensitivity to S₈ and fσ₈ respectively.
  - `S8_fid`, `fs8_fid`: the fiducial values used by the survey; they may be scalars and are broadcast internally.
  - `gamma`: WL-style S₈ exponent (default 0.5 when absent).
  - `meta`: optional metadata hash (the loader always records `{"file": ...}` plus whatever was provided).
- Optional keys:
  - `cov`: N×N covariance matrix (`inv_cov` is computed once in the loader). When absent the fit treats every amplitude as independent and applies `weights` before rescaling by `A_err`.
  - `labels`: human-readable dataset names (stored as strings for diagnostics).
  - `weights`: positive scaling factors (default 1.0) that amplify or downweight individual amplitudes when the covariance matrix is missing. They are also exposed in the fit extras so downstream tooling can inspect or rebalance individual LCDM/PBUF contributions.

The loader in `cosmos/fits/lensing_cross/lensing_cross.py` validates the shapes, expands scalars, stores the covariance/inverse, and returns a dict that every downstream layer reuses.

## Model API

This fit is intentionally agnostic to the growth ODEs or Cℓ kernels. The only helpers it calls are:

```python
S8_model = model.S8(gamma[i])
fs8_model = model.fs8(z_eff[i])
```

Both LCDM and PBUF expose these helpers: LCDM uses its cached growth table to compute σ₈, PBUF reuses the R(a) solution. The fit never builds H(z) or P(k) itself.

## χ² definition

Let `A_model` be the amplitude predicted by the survey-specific scalings:

```
A_model = (S8_model / S8_fid)^p_exponent × (fs8_model / fs8_fid)^q_exponent
```

Then:

- With a covariance matrix: `χ² = (A_model − A_obs)ᵀ C⁻¹ (A_model − A_obs)`.
- Otherwise: `χ² = Σ ((A_model − A_obs) / A_err)²`.

The fit returns `{"A_model": A_model}` so diagnostics can compare predicted vs observed amplitudes.

## Pipeline integration

- Loader: `cosmos.fits.lensing_cross.load_lensing_cross_dataset()` is cached via `cosmos.datasets.get_dataset("lensing_cross")`.
- Fit helper: `cosmos.fits.lensing_cross.run_lensing_cross_fit(model, dataset)` evaluates the χ² above using the model’s `S8`/`fs8` and the survey-provided exponents.
- CLI: `cli.py fit lensing_cross --model=<lcdm|pbuf>` now exposes these compressed constraints to quick experiments.
- Basin walker: include `"lensing_cross"` in the dataset list so LCDM/PBUF workers queue `LensingCross` workers; the total χ² simply adds `chi2_lensing_cross`. Extras now include `weights`, `residuals`, and the error-scaled contributions so any diagnostics (or later weighting scheme) can see how each amplitude participates in the total compressed χ².
- Sanity & optimisation: `cosmos.optim.sanity.evaluate_candidate(..., datasets=["lensing_cross"])`, `cosmos.models.*.evaluate_chi2(..., datasets=["lensing_cross"])`, and the basin workers all treat lensing cross exactly like the other compressed fits.

As soon as V12 models support P(k) and Cℓ integrals, this compressed data can serve as a sanity check: any model that fails the full cross-spectrum should naturally also deviate from these amplitude-level constraints.
