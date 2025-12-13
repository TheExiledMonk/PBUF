# fσ₈ Prediction Module

## Purpose

Computes the redshift-space distortion observable `fσ₈(z)` directly from the linear growth factor and the model's present-day `σ₈₀`. The module normalizes `D(a)` to unity at `z=0`, evaluates `f(z)=d ln D / d ln a`, assembles `σ₈(z)=σ₈₀ D(z)`, and exposes the product `fσ₈` on a configurable redshift grid together with anchor values at `z=0`, `0.5`, and `1.0`.

## CLI

```
cosmos_cli predict fsigma8 --model <name> [--zmin 0.0] [--zmax 2.0] [--points 200]
```

- `--zmin` / `--zmax` set the redshift span (defaults 0.0 / 2.0).
- `--points` controls the grid resolution (default 200).

Running the command produces a standard prediction JSON plus a canonical `fsigma8_plot` showing `fσ₈(z)` over the valid mask. The summary contains `fs8_z0`, `fs8_z0p5`, `fs8_z1`, and the normalized `σ₈₀`.

## Model API

The module relies on the Cosmos2 adapter to provide:

- `model.parameters["H0"]` (used to build `E(a)=H(a)/H0`)
- `model.H(a)` (via `PredictionModelAdapter.H` which calls the model's `Hubble`)
- `model.omega_m0()` for the growth solver
- `model.sigma8_today()` for the `σ₈₀` normalization

No model-specific files or tables are assumed; the growth integrator comes from `cosmos2.kernels.common.growth.solve_growth`.

## Outputs

The JSON payload stores the full grid plus masked vectors:

```
{
  "name": "fsigma8",
  "z": [...],
  "a": [...],
  "D_norm": [...],
  "f": [...],
  "sigma8_z": [...],
  "fs8": [...],
  "mask_valid": [...],
  "summary": {
      "fs8_z0": ...,
      "fs8_z0p5": ...,
      "fs8_z1": ...
  },
  "meta": {
      "z_min": ...,
      "z_max": ...,
      "n_points": ...,
      "sigma8_0": ...,
      "model_name": ...,
      "created_at": "...",
      "version": "1.0",
      "notes": "fσ8(z) computed from normalized growth D(a) and σ8_0.",
      "description": "Redshift-space distortion prediction fσ8(z)..."
  },
  "fs8_z0": ...,
  "fs8_z0p5": ...,
  "fs8_z1": ...,
  "sigma8_0": ...
}
```

`meta["description"]` is the same narrative used by the reporting system and states:
“Redshift-space distortion prediction fσ8(z). Computed from the linear growth factor D(a), normalized to D(z=0)=1, and the present-day σ8_0. The module provides f(z), σ8(z), and fσ8(z) on a redshift grid, along with key summary values at z=0, 0.5, and 1.”

## Reporting & Science Runner

Enable the module via the science-runner configuration:

```json
[predictions]
modules = [..., "fsigma8", ...]

[predictions.fsigma8]
zmin = 0.0
zmax = 2.0
points = 200
```

The runner stores the module outputs under `predictions/fsigma8/<model>/result.json` and exports the `fsigma8_plot` for each model plus a combined overlay in the report. Summary tables reference `fs8_z0`, `fs8_z0p5`, and `fs8_z1` for every model entry, and the description shown in reports pulls from `meta["description"]`.

## Testing

Unit tests exercise the module with an analytic toy model (EdS-like `D(a)∝a`, `f=1`, known `σ₈₀`) to confirm:

- `D_norm` at `z=0` is unity.
- `fσ₈(z)` anchors match `σ₈₀ * a`.
- Vector lengths stay consistent and the prediction serializes to JSON cleanly.
