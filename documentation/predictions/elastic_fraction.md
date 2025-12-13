# Prediction Module Dev Doc — elastic-fraction

This document describes the `elastic-fraction` prediction module implemented in `cosmos2/predictions/modules/elastic_fraction.py`. The module registers via `@register_prediction`, consumes the shared `PredictionModelAdapter`, and produces a `PredictionResult` payload that works both with the CLI and the science runner.

## Purpose

Track the elastic energy density fraction
\[
f_σ(a) = \frac{Ω_σ(a)}{Ω_{\mathrm{tot}}(a)} = \frac{Ω_σ(a)}{E(a)^2}
\]
as a function of redshift. Recording `fσ(z)` alongside its underlying `Ωσ(z)` curve allows quick comparisons between the elastic (e.g. PBUF) and non-elastic (e.g. ΛCDM) models. ΛCDM naturally produces `Ωσ=0`, so this prediction visually and numerically highlights the elastic contribution to the total budget.

The module:

- Samples a configurable redshift grid (default `z ∈ [0, 5]` with 300 points).
- Queries the model’s elastic surface via `model.elastic.omega_sigma(a)` (fallbacks safely to zeros).
- Uses `model.background.E(a)` to build `Ω_{\mathrm{tot}}(a) = E(a)^2`.
- Computes `fσ(z)`, a validity mask (`Ω_tot>0` and all values finite), and summary statistics such as `Ωσ₀`, `fσ₀`, the raw `fσ` peak and `z_peak`, and the half-peak redshifts.
- Logs a warning if fewer than 10 valid samples survive the mask.
- Exposes `meta["description"]`/`meta["notes"]` strings documenting the diagnostic.
- Emits a canonical `PredictionPlot` that shows `fσ(z)` (and optionally `Ωσ(z)`) on valid points.

## CLI usage

Running the module from the CLI is as simple as:

```
cosmos_cli predict elastic-fraction --model pbuf --z-max 6.0 --points 400
```

Options that are also available in the science-runner configuration (`predictions.module_configs.elastic-fraction`) are:

- `--z-min <float>`: minimum redshift (default `0.0`).
- `--z-max <float>`: maximum redshift (default `5.0`).
- `--points <int>`: number of samples on the redshift grid (default `300`).

The module automatically writes its JSON payload and plot when `--save-json` / `--save-plots` are provided via `cosmos_cli`.

## Model requirements

`PredictionModelAdapter` exposes two surfaces that this module depends on:

1. `model.elastic.omega_sigma(a)` – returns `Ωσ(a)` for each scale factor. If the underlying model does not expose an elastic surface, the adapter falls back to a zero-valued stub so the module still runs.
2. `model.background.E(a)` – returns the normalized expansion history `E(a)=H(a)/H₀`, which is squared to form `Ω_tot(a)`.

No file paths or hard-coded constants are assumed; the module works with whichever model instance is passed.

## Numerical specification

The module builds:

- `z_grid = np.linspace(z_min, z_max, points)`
- `a_grid = 1 / (1 + z_grid)`
- `Ωσ(a)` via `model.elastic.omega_sigma`
- `E(a)` via the background helper
- `Ω_tot(a) = E(a)^2`
- Valid points where `Ω_tot > 0` and all values are finite
- `fσ(z)` using these masked values (NaN elsewhere)

Summary diagnostics are computed via `value_at_z` (picking the closest valid point at `z=0`) and through simple array scans for `fσ_peak`, `z_peak`, `z_half_peak_lo`, and `z_half_peak_hi`. The half-peak routine walks the valid subset to linearly interpolate the redshift where `fσ` crosses 50% of its peak on either side.

## Outputs

The module emits the following structured payload within `PredictionResult.results`:

```json
{
  "name": "elastic-fraction",
  "z": [...],
  "a": [...],
  "Omega_sigma": [...],
  "Omega_tot": [...],
  "f_sigma": [...],
  "mask_valid": [...],
  "summary": {
      "Omega_sigma_0": ...,
      "f_sigma_0": ...,
      "f_sigma_peak": ...,
      "z_peak": ...,
      "z_half_peak_lo": ...,
      "z_half_peak_hi": ...
  },
  "meta": {
      "z_min": ...,
      "z_max": ...,
      "n_points": ...,
      "model_name": "...",
      "created_at": "...",
      "version": "1.0",
      "description": "...",
      "notes": "... (same as description but more detailed)"
  }
}
```

Top-level `PredictionResult.metadata` also mirrors the model name, grid size, valid point count, and the human-readable description so reporting cards and CLI summaries can reuse that string.

## Reporting integration

The science-run report loader now builds a dedicated “Elastic energy fraction” card. It shows a comparison table with `Ωσ₀`, `fσ₀`, `fσ_peak`, `z_peak`, and the half-peak redshifts for every model, and it renders a shared plot overlaying `fσ(z)` curves (plus `Ωσ(z)` on a second axis). If a module run had no valid samples, the card emits a warning and omits the plot while still listing the summary values (which will be `null` or empty).

## Testing

`tests/test_predictions.py` now includes a toy model with a known analytic `Ωσ(a)` and `E(a)`. The tests verify:

- The ratio `fσ(z)` matches `Ωσ/Ω_tot` within tolerance.
- `Ωσ₀`, `fσ₀`, `fσ_peak`, `z_peak`, and the half-peak redshifts match the expected analytical values.
- The resulting payload serializes cleanly to JSON.

