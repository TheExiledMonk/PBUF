# Curvature Prediction Module

## Purpose

Provides the PBUF-specific forecast for today's residual spatial curvature `Ω_k0`, the implied curvature radius, and an optional breakdown of the elastic–geometric closure. This prediction is designed to highlight the +0.01–+0.03 residual curvature signal that PBUF expects to survive beyond ΛCDM.

## CLI

```
cosmos_cli predict curvature --model pbuf [--output-table] [--output-plot] [--diagnostics]
```

The module plugs into the existing `cosmos_cli predict` workflow, respektively exposing the `--output-table` and `--output-plot` helpers plus a `--diagnostics` flag to surface extra entries in the CLI summary.

## Model API

The implementation inspects the high-level prediction adapter:

- reads `H(a)` and the `H0` parameter to reconstruct `Ω_total(a=1)` and the closure defect,
- uses elastic stiffness at `a=1` (if provided) to derive `ε₀` and `k_sat`,
- pulls baryon/matter/alpha parameters from `model.parameters`.

If the adapter cannot evaluate `H` or `H0`, the module returns a structured error with `metadata.error = "missing_geometry_api"`.

## Outputs

Results include:

- `Omega_k0`: residual curvature today.
- `curvature_radius_Mpc`: curvature radius in Mpc (c/(H0*√|Ω_k0|)).
- `closure_today`: total density sum at `a=1`.
- `components`: sub-dictionary with `{Omega_m0, Omega_r0, Omega_b0, Omega_sigma0, alpha, k_sat}`.

If `--diagnostics` is requested, the CLI summary also prints each component plus `epsilon0_today` and `residual_curvature`.

Tables and plots are only emitted when the corresponding flags are turned on; the table mirrors the component breakdown, while the bar chart contrasts the matter/radiation/alpha/elastic pieces.

## Science runner + reports

Because the module registers via the standard prediction registry, it can be enabled in unified science runs by adding `"curvature"` to the `[predictions].modules` list. The generated JSON payload already matches the schema expected by the reporting system.
