# Curvature Identity Prediction Module

## Purpose

Check that the PBUF baryon–saturation–rigidity relations hold for a given fit:  
`Ω_b0 ≈ 2α`, `k_sat ≈ 1 − α`, and `k_max ≈ ε₀ − α`. The module reports the predicted vs. actual values, residuals, and metadata that lets you monitor how tightly the solution locks baryons, curvature, and the elastic sector.

## CLI

```
cosmos_cli predict curvature-identity --model pbuf [--output-table] [--output-plot]
```

The module plugs into the shared `cosmos_cli predict` command. `--output-table` writes the `(actual, predicted, delta)` table, while `--output-plot` emits the residual bar chart.

## Model API

The implementation reads directly from the prediction adapter:

- `model.parameters["alpha"]` and `model.parameters["Omega_b0"]` are mandatory.
- `model.elastic_stiffness(1.0)` (or an equivalent `epsilon0` surface) provides `ε₀` so that `k_max_pred = ε₀ − α`.
- Optional model parameters such as `k_sat`, `k_max`, and `Omega_k0` are used if present to complete the identity check.

If `alpha` or `Omega_b0` are missing, the module returns an error result (`metadata.error = "missing_alpha_or_Omega_b0"`).

## Outputs

- `results`: alpha, epsilon0_today, actual/predicted values for `Ω_b0`, `k_sat`, `k_max`, plus residuals (`ΔΩ_b0`, `Δk_sat`, `Δk_max`). Optional `Omega_k0` and `closure_today` entries appear when available.
- `tables`: `curvature_identity_components` with columns `quantity`, `actual`, `predicted`, `delta`.
- `plots`: `curvature_identity_bar` showing residuals for the quantities with defined deltas.
- `metadata`: includes `model`, `has_k_max`, `timestamp`, and a summary string describing the identity check.

## Science-run integration

Include `"curvature-identity"` in `[predictions].modules` to have unified runs add the identity check to every PBUF fit. The results ship inside the standard prediction payload and will appear alongside other tables/plots in the science report.
