# Prediction Module Dev Doc — statefinder

This document describes the `statefinder` prediction module implemented in `cosmos2/predictions/modules/statefinder.py`. The module registers via `@register_prediction` and exposes `PredictionResult`/`PredictionPlot` structures that the reporting stack and CLI share.

## Purpose

Statefinder diagnostics (\(r, s\)) provide a geometrical language for deviations from ΛCDM. The pair is defined as

\[
r \equiv \frac{\dddot{a}}{a H^3}, \qquad s \equiv \frac{r - 1}{3(q - 1/2)},
\]

with \(q = -a/E(a) \, \mathrm{d}E/\mathrm{d}a\) and \(E(a) = H(a)/H_0\). In ΛCDM with a cosmological constant the pair sits at the fixed point \((1, 0)\) for all redshifts. This prediction module uses the normalized expansion history \(E(a)\) to compute \(r(z)\), \(s(z)\) and the associated deceleration \(q(z)\), and surfaces the present-day values \(r(0)\) and \(s(0)\) for model comparison.

## CLI usage

The module is wired to the `predict` subcommand, e.g.:

```
cosmos_cli predict statefinder --model lcdm --points 200 --zmax 3.0 --zmin 0.0
```

Supported flags:

- `--zmin <float>`: minimum redshift (default `0.0`, must be ≥ 0).
- `--zmax <float>`: maximum redshift (default `3.0`, must exceed `--zmin`).
- `--points <int>`: number of uniform redshift samples (default `200`, minimum `3`).

These flags also appear under `[predictions.module_configs.statefinder]` in science run configs.

## Model inputs

The module only depends on the shared background API exposed by `PredictionModelAdapter`:

- `model.H(a)`: used to build the normalized expansion rate \(E(a) = H(a)/H_0\).
- `model.raw_model` metadata is used for logging the class name in the prediction metadata.

No additional model-specific datasets or files are assumed, so the module works with both LCDM and PBUF models that satisfy the background API.

## Numerical workflow

1. Build a uniform redshift grid \(z \in [z_{\min}, z_{\max}]\) with `points` samples and set \(a = 1/(1 + z)\).
2. Query \(H(a)\) and normalize by \(H_0 = H(1)\); finite differencing (forward/backward on the edges, central in the interior) approximates \(\mathrm{d}E/\mathrm{d}a\) and \(\mathrm{d}^2E/\mathrm{d}a^2\).
3. Mask samples where \(E\), its derivatives, or the denominators become non-finite or negative.
4. Compute \(q(a)\), \(r(a)\), and \(s(a)\) following the standard kinematic formulas, guarding against small denominators (\(|3(q-1/2)| < 10^{-8}\)).
5. Record \(r(z)\), \(s(z)\), \(q(z)\), \(a(z)\), \(z\), and the validity mask in the prediction payload, and extract the valid \(r_0, s_0\) pair closest to \(z=0\).

If fewer than 10 samples pass the mask, a warning is logged but the module still emits the prediction payload with NaNs filling the invalid bins.

## Outputs

`run_prediction(model, config)` returns a nested dictionary with:

- `"name": "statefinder"` and a `"summary": {"r0": ..., "s0": ...}` pair.
- Arrays `"z"`, `"a"`, `"r"`, `"s"`, `"q"`, and `"mask_valid"` of length `points`.
- `"meta"` metadata including the sampling bounds, timestamp, `"notes"` string, and the human readable `"description"` that references the ΛCDM fixed point.
- A single `PredictionPlot` named `"statefinder_rs"` describing the (r, s) trajectory for the valid samples.

The prediction dict can be serialized to JSON and stored alongside other science-run outputs for report generation.

## Reporting integration

The reporting stack now treats the `statefinder` module specially:

- The combined report section surfaces the r-s trajectory for every model with a dedicated plot saved under `run_dir/predictions/figures`.
- A summary table lists \(r_0\) and \(s_0\) for each model along with the number of valid samples, and the fixed point \((1, 0)\) is flagged in the caption for context.
- If no valid points exist the panel displays a warning, skips the plot, but still lists the null summary entries.

## Testing

Unit tests instantiate a lightweight toy ΛCDM model and assert:

- \(r(z)\) ≈ 1.0 and \(s(z)\) ≈ 0.0 whenever the mask is True.
- Arrays `z`, `a`, `r`, `s`, `q`, and `mask_valid` share the same length.
- The summary fields `r0` and `s0` are within 1e-4 of the fixed point.
- The prediction payload serializes cleanly via `json.dumps`.

