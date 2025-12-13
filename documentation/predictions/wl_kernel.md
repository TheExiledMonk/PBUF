# Prediction Module Dev Doc — wl-kernel

This document describes the lightweight weak-lensing kernel module implemented in
`cosmos2/predictions/modules/wl_kernel.py`. The module registers itself via
`@register_prediction` and produces a JSON-friendly payload that the science-runner,
single-run CLI, and reporting system all understand.

## Purpose

The `wl-kernel` module evaluates the redshift-space efficiency kernel
\[
W(z) \propto \frac{H(z)}{c} \, \chi(z) \int_z^{z_\text{max}} n(z') \frac{\chi(z') - \chi(z)}{\chi(z')} \, \mathrm{d}z'
\]
for a source redshift distribution \(n(z)\). This kernel highlights how lenses at different
redshifts contribute to the background shear signal seen by a survey. The prediction is intentionally
lightweight: it never computes the shear power spectrum \(C_\ell\), and is meant for plotting,
qualitative comparisons, and seeding future WL work.

## CLI usage

Available via `cosmos_cli.py predict wl-kernel`. Example command:

```bash
python cosmos_cli.py predict wl-kernel --model lcdm --zmin 0.0 --zmax 3.0 --points 300 \
    --source-type lsst_like --source-z0 0.3 --source-alpha 2.0 --source-beta 1.5
```

Available flags:

- `--zmin <float>`: minimum lens redshift (default `0.0`, must be ≥ 0).
- `--zmax <float>`: maximum redshift sampled by the kernel (default `3.0`, must exceed `--zmin`).
- `--points <int>`: number of redshift samples on the grid (default `300`, minimum `2`).
- `--no-normalize`: leave `W_norm` equal to the raw kernel instead of scaling to `max(W)=1`.
- `--source-type`: choose a builtin distribution (`"lsst_like"`, `"euclid_like"`, `"simple"`).
- `--source-z0`, `--source-alpha`, `--source-beta`: override the parametric \(n(z) \propto z^\alpha \exp[-(z/z_0)^\beta]\) parameters.

These flags mirror the entries under
`[predictions.module_configs.wl-kernel]` in science-run configs, so both CLI and science-runner
workflows share the same behavior.

## Model inputs

The module requires the background API exposed by `PredictionModelAdapter`:

- `model.background.H(z)`: the Hubble rate used in the kernel prefactor.
- `model.background.c_value()`: speed of light in units consistent with `H(z)` (e.g., km/s/Mpc).
- `model.background.comoving_distance(z)`: transverse comoving distance \(\chi(z)\).

The module does not rely on matter power spectra or additional elastic properties, so it works
across LCDM, PBUF, and toy models that satisfy the minimal background surface.

## Source distribution

`wl_source_distribution(z_grid, config)` (in `cosmos2/predictions/wl_utils.py`) builds a normalized
\(n(z)\) from either a runner-provided config or a builtin parametric form. The config may specify
`type`, `parameters`, and individual `z0`, `alpha`, or `beta` overrides. When no config is provided
the default `"lsst_like"` distribution is used.

## Numerical workflow

1. Build the uniformly spaced redshift grid \(z \in [z_{\min}, z_{\max}]\) with `points` samples.
2. Evaluate \(H(z)\), \(\chi(z)\), and \(c\); mask any points where the background is non-finite or
   the speed of light is invalid. Similarly mask any negative or non-finite \(n(z)\) samples.
3. For each valid lens sample, integrate \(n(z_s)(\chi_s - \chi)/\chi_s\) over sources behind the lens
   using the trapezoidal rule on the masked grid (an O(N²) loop is acceptable for 300 samples).
4. Multiply by \((H(z)/c)\chi(z)\) to build `W_raw`.
5. Normalize `W_raw` to produce `W_norm` when requested; otherwise copy the raw kernel.
6. Define `mask_valid` as entries where `W_norm` and the background remain finite. A warning is logged
   if fewer than 10 valid points survive the filtering.

The summary values (`z_peak`, `W_peak_value`, `z_median`) come from the normalized kernel and are
stored in both the prediction payload and the metadata for reporting.

## Outputs

The prediction result contains:

- `results["z"]`, `["chi"]`, `["n_z"]`, `["W_raw"]`, `["W_norm"]`, and `["mask_valid"]`: arrays aligned with the grid.
- `results["summary"]`: the peak location, peak amplitude, and median lensing redshift (all nullable).
- `results["meta"]`: metadata with grid settings, `n_z_model`, normalization choice, and descriptive text.
- `metadata["summary"]`: the same summary scalars used by reporting.
- No tables are produced but a single plot `wl_kernel_vs_z` visualizes `W_norm(z)` (and `n(z)`) on the same axes.
- The `meta["description"]` string is the human-friendly text mandated by the spec, and `meta["notes"]`
  document how `W(z)` is derived.

All arrays and metadata are JSON serializable via `PredictionResult.to_dict()`.

## Reporting integration

The reporting stack treats `wl-kernel` like other modules. Key points:

- The combined module plot overlays each model’s `W_norm(z)` vs \(z\); the shared `z` axis keeps the
  comparison clean.
- The prediction card lists `z_peak`, `W_peak_value`, and `z_median`, and shows the warning if no valid
  points exist.
- When valid samples exist the `wl_kernel_vs_z` plot is embedded in the module’s panel; otherwise the
  plot is skipped but metadata/summary remain accessible.
- `meta["description"]` and `meta["notes"]` surface in the combined report card so readers understand the
  purpose of the kernel.

## Testing

Unit tests (see `tests/test_predictions.py`) instantiate a toy model with constant \(H(z)\) and
\(\chi(z) = z\). Tests ensure:

- The result serializes through `json.dumps(result.results)` without errors.
- The arrays `z`, `chi`, `n_z`, `W_raw`, `W_norm`, and `mask_valid` all share the same length.
- `W_raw` is finite and non-negative where `mask_valid` is true.
- `W_norm` peaks between redshift ∼0.2–1.5 and the weighted median lies in that range as well.
- The metadata honors custom `source_distribution` configs and respects the `normalize=False` flag.
