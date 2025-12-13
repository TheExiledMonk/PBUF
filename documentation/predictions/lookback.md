# Prediction Module Dev Doc — lookback

This document describes the `lookback` prediction module implemented in `cosmos2/predictions/modules/lookback.py`. The module registers via `@register_prediction` and provides a JSON-friendly payload that the unified science-run reporting stack consumes alongside the canonical tables and plots.

## Purpose

The `lookback` module integrates the line-of-sight relation

\[
t_L(z) = \int_0^z \frac{\mathrm{d}z'}{(1 + z') H(z')}
\]

to compute the elapsed cosmic time between today and every redshift sample. The module reports the lookback time \(t_L(z)\), the resulting cosmic age \(t(z) = t_0 - t_L(z)\), and the present age \(t_0\). The output is useful for comparing galaxy ages, star-formation histories, and outreach-friendly visualizations.

## CLI usage

The module appears under `cosmos_cli.py predict lookback`. Example:

```bash
python cosmos_cli.py predict lookback --model lcdm --zmin 0 --zmax 10 --points 300 --output-plot --output-table
```

Supported flags:

- `--zmin <float>`: minimum redshift (default `0.0`, must be ≥ 0).
- `--zmax <float>`: maximum redshift (default `10.0`, must exceed `--zmin`).
- `--points <int>`: number of redshift samples (default `300`, minimum `2`).
- `--output-plot`: include the canonical lookback + age plots in the prediction payload.
- `--output-table`: export the `lookback_vs_z` table with `[z, t_L, t(z), mask_valid]`.

These flags are mirrored under `[predictions.module_configs.lookback]` in science-run configs so the unified runner inherits the same behavior.

## Model inputs

The module depends only on the background API exposed by `PredictionModelAdapter`:

- `model.background.H(z)`: the hubble rate \(H(z)\) used directly in the integrand.
- `model.raw_model` metadata is used for logging the classifier name inside the prediction metadata.

No additional model-specific attributes are required, so the module works with both LCDM and PBUF models that provide a `Hubble`/`H` implementation.

## Numerical workflow

1. Build a uniform grid \(z \in [z_{\min}, z_{\max}]\) with `points` entries.
2. Query \(H(z)\) and mask samples where the rate is non-finite or non-positive.
3. For the valid grid, approximate the cumulative integral using the trapezoidal rule on \(\frac{1}{(1+z)H(z)}\).
4. Multiply the integral by the shared `MPC_TO_KM / SECONDS_PER_GYR` factor to convert from \(\text{km/s/Mpc}\) into gigayears.
5. Record \(t_L(z)\), the cosmic age \(t(z)\), and the validity mask; the last valid sample approximates \(t_0\).
6. Summarize \(t_0\), \(t(z=1)\), and \(t(z=6)\) for quick comparison across models.

The module logs a warning if fewer than 10 valid points are present so downstream consumers know the integral may be undersampled.

## Outputs

`run_prediction(model, config)` returns a `PredictionResult` containing:

- `results["z"]`, `"tL_Gyr"`, `"t_age_Gyr"`, and `"mask_valid"`: arrays of length `points`.
- `results["t0_Gyr"]`, `"t_z1_Gyr"`, `"t_z6_Gyr"`: scalar summary values derived from the valid subset.
- `metadata["summary"]`: mirrors the scalars so the reporting section and science-run summaries can render them.
- `metadata["description"]`/`metadata["notes"]`: human-readable strings from the developer spec.
- `metadata["time_unit"] = "Gyr"`, `metadata["valid_points"]`, and `metadata["mask_valid_fraction"]` to help downstream diagnostics.
- Optional `PredictionTable` named `lookback_vs_z` when `--output-table` is set.
- Up to two `PredictionPlot` entries (`lookback_time_vs_z`, `cosmic_age_vs_z`) when `--output-plot` is enabled; both use `z` as the x-axis so combined report plots overlay smoothly.

The payload is JSON serializable (see `PredictionResult.to_dict()`) and includes metadata such as `model`, `created_at`, `version`, and the run-specific grid parameters.

## Reporting integration

The reporting stack consumes the module exactly like other predictions. Highlights:

- A combined overlay plot uses `z` as the shared axis so every model’s \(t_L(z)\) and \(t(z)\) curves appear on the same figure.
- The prediction card lists the `t0_Gyr`, `t_z1_Gyr`, and `t_z6_Gyr` entries (pulled from `metadata.summary`) along with the number of valid samples.
- If no valid points exist the section shows a warning, omits the plots, and leaves the summary scalars null while still showing metadata/notes.
- The `lookback_vs_z` table (when produced) supplies the full grid for comparison with other modules or external tooling.

## Testing

Unit tests instantiate a toy model with \(H(z) = H_0 (1 + z)\) so the integral has a known analytic solution. Tests ensure:

- \(t_L(z)\) is monotonic increasing and \(t(z)\) is monotonic decreasing on the valid grid.
- `t0_Gyr`, `t_z1_Gyr`, and `t_z6_Gyr` match their expected values within tight tolerances after converting via `MPC_TO_KM / SECONDS_PER_GYR`.
- The prediction payload serializes through `json.dumps(result.to_dict())`.

