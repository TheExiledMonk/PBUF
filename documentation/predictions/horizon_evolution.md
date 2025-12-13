# Prediction Module Dev Doc — horizon-evolution

The `horizon-evolution` module lives at `cosmos2/predictions/modules/horizon_evolution.py`. It registers via `@register_prediction` and emits a JSON-friendly payload describing the physical Hubble radius \(R_{H,\text{phys}} = c/H(z)\), the comoving Hubble radius \(R_{H,\text{com}} = c(1+z)/H(z)\), and a truncated comoving particle horizon \(\chi_p(z) \approx \int_z^{z_{\max}} c/H(z') \,\mathrm{d}z'\). The prediction is numerically stable, stores validity masks, and provides metadata designed to feed the unified science-run reporting stack.

## Purpose

The output highlights how the Hubble scale evolves through cosmic history, making it easy to compare PBUF elastic behavior to standard horizon language. The comoving radius shows whether horizons shrink or grow compared to \(a H\), while the truncated particle horizon provides a finite diagnostic of causal reach. The module also reports key summary scalars \(R_{H,\text{phys}}(z=0)\), \(R_{H,\text{com}}(z=0,1,6)\) so downstream modules can tabulate the zoomed-in values per model.

## CLI usage

Invoke the module with:

```bash
python cosmos_cli.py predict horizon-evolution --model pbuf --zmax 6 --points 200
```

Allowed flags:

- `--zmin <float>`: lowest redshift on the grid (default `0.0`, must be ≥ 0).
- `--zmax <float>`: highest redshift (default `10.0`, must exceed `--zmin`).
- `--points <int>`: length of the redshift grid (default `300`, minimum `2`).
- `--no-plot`: suppress the canonical plot that overlays \(R_{H,\text{com}}\), \(R_{H,\text{phys}}\), and \(\chi_p\).

These flags mirror the `[predictions.module_configs.horizon-evolution]` block in science-run configs so the unified runner inherits the same grid when executing across LCDM and PBUF models.

## Model inputs

The module depends exclusively on the background helpers exposed by `PredictionModelAdapter`:

- `model.background.H(z)` provides \(H(z)\) in the same velocity units as the speed of light (e.g., `km/s/Mpc`).
- `model.background.c_value()` returns the light-speed constant in matching units so that \(c/H\) carries distance units (Mpc when \(H\) is \(km/s/Mpc\)).

No further model-specific attributes are required, which keeps the module compatible with all registered models that implement `Hubble` or equivalent.

## Numerical workflow

1. Build a uniform grid \(z \in [z_{\min}, z_{\max}]\) with `points` samples and compute \(a = 1/(1+z)\).
2. Evaluate \(H(z)\) and mask entries that are non-finite or non-positive. A scalar `c` is validated once and gates the full prediction (an invalid `c` sets all masks to `False`).
3. Compute \(R_{H,\text{phys}} = c/H\) and \(R_{H,\text{com}} = c(1+z)/H\) on the valid grid, and only keep values that are finite and positive.
4. Build the truncated comoving particle horizon \(\chi_p(z)\) by integrating \(c/H\) from each sampled redshift up to \(z_{\max}\) using a reversed trapezoidal sweep. NaNs or non-finite values propagate as expected.
5. The module issues a warning if fewer than 10 valid points survive the masking so downstream consumers know when the grid is undersampled.
6. Four summary scalars are filled using a nearest-valid lookup: \(R_{H,\text{phys}}\) at \(z=0\) and \(R_{H,\text{com}}\) at \(z=\{0,1,6\}\). Missing or invalid neighbors yield `null` entries.

## Outputs

The prediction result includes:

- `results["z"]`, `results["a"]`, `results["R_H_phys"]`, `results["R_H_comoving"]`, `results["chi_particle"]`, `results["mask_valid"]`, and `results["mask_particle"]`: arrays matching the requested grid length.
- `results["summary"]`: maps the four scalars (`R_H0_phys`, `R_H0_comoving`, `R_H_z1_comoving`, `R_H_z6_comoving`) to the nearest valid sample or `None`.
- `results["meta"]`: carries \(z\)-grid parameters plus `distance_unit = "same as c/H(z)"`, `version = "1.0"`, `created_at`, `model_name`, `description`, and the manufacturer-approved narrative notes.
- `metadata`: includes the `model` name, timestamps, grid settings, mask statistics, and the shared description/notes strings that feed the reporting section.
- `plots`: one `PredictionPlot` named `horizon_evolution_vs_z` (unless `--no-plot` is set or there are zero valid points) with `z`, `R_H_comoving`, `R_H_phys`, and `chi_particle` all aligned on the same axis.

The `PredictionResult` serializes cleanly via `result.to_dict()` and is suffixed with `status = "success"` once the computation finishes.

## Reporting integration

The reporting stack now renders a dedicated horizon-evolution card:

- A combined plot overlays each model's \(R_{H,\text{com}}(z)\), optionally shaded with \(\chi_p(z)\) and \(R_{H,\text{phys}}(z)\), to highlight differences across models. This plot is saved under `predictions/figures/horizon_evolution_combined_comparison.png`.
- The card presents a table of `R_H0_phys`, `R_H0_comoving`, `R_H_z1_comoving`, `R_H_z6_comoving`, and the number of valid samples for every model included in the run.
- If no valid samples survive the mask, the plot is omitted, and a warning message explains that the horizon evolution data is unavailable while the summary table still renders (values are `null`).
- The metadata and description text are re-used from `results["meta"]` so the narrative stays consistent between JSON dumps and the HTML report.

## Testing

Unit tests instantiate a toy, flat ΛCDM-like model with \(H(z) = H_0 \sqrt{\Omega_m (1+z)^3 + (1 - \Omega_m)}\) and examine the prediction dictionary:

- Verify the comoving and physical horizons are finite, positive, and that \(R_{H,\text{phys}}\) decreases with \(z\) while \(R_{H,\text{com}}\) peaks near \(z\sim 1\) and then behaves as expected.
- Confirm the truncated particle horizon \(\chi_p(z)\) is non-negative and monotonic (within numerical tolerance) whenever the mask signals valid samples.
- Assert the summary scalars at \(z=\{0,1,6\}\) are finite and match the grid values.
- Ensure `result.to_dict()` serializes without raising and the metadata contains the prescribed description/notes strings.
