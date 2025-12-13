# Prediction Module Dev Doc — void-size

This document describes the `void-size` prediction module housed under `cosmos2/predictions/modules/void_size.py`. The module registers via `@register_prediction`, wires its CLI flags through `cli/main.py`, and exposes the same `PredictionResult`/`PredictionTable`/`PredictionPlot` structures consumed by the reporting stack.

## Purpose

Compute the typical cosmic void radius \(R_{\text{void}}(z)\) predicted by PBUF scalings that combine growth suppression and elastic slack. The result is expressed in Mpc and can optionally be compared to a ΛCDM-style reference baseline. The module produces:

- A present-day void prediction \(R_{\text{void}}(0)\) plus interpolated values at \(z=0.5\) and \(z=1\).
- Ratios of the PBUF prediction to the ΛCDM reference when `--compare-lcdm` is enabled.
- Optional tables/plots describing the growth factors, S-growth, elastic factor, and resulting void sizes.

This remains a lightweight desktop-friendly prediction built on PBUF-native quantities (`D(z)`, \(\alpha\), \(H(z)\)) with no external cosmological assumptions.

## CLI usage

`void-size` attaches to the `predict` subcommand automatically. Run it with, for example:

```
cosmos_cli predict void-size --model pbuf --points 150 --zmax 1.0 --compare-lcdm --output-plot --output-table
```

The module accepts these CLI flags (and their science-run config equivalents):

- `--zmax <float>`: maximum redshift to sample (default `1.0`).
- `--points <int>`: number of redshift grid points (default `100`).
- `--compare-lcdm`: include LCDM-style ratios and tables (otherwise ratios are set to `None`).
- `--R_ref_Mpc <float>`: reference ΛCDM void radius today (default `25.0` Mpc).
- `--eta-growth <float>`: exponent for the growth suppression scaling (default `0.5`).
- `--gamma-alpha <float>`: strength of the elastic slack multiplier (default `1.0`).
- `--beta-z <float>`: exponent defining the ΛCDM reference redshift evolution (default `0.3`).
- `--output-table`: emit the full `void_radius_vs_z` table.
- `--output-plot`: emit the canonical `R_void_vs_z` (plus ratio) plots.

Module arguments show up verbatim under `predictions.module_configs.void-size` in any science-run config, just like other prediction modules.

## Model inputs

The module uses the `PredictionModelAdapter` (see `cosmos2/predictions/model_api.py`) so it only relies on:

- `model.growth_factor(z)` (via the raw model or a fallback growth solver in this module).
- `model.alpha` (elastic amplitude \(\alpha\), derived from the PBUF thermal metadata).
- `model.H(a)` when the fallback solver needs the expansion history.
- Optional \(H_0\) / \(\Omega_{m0}\) entries from `model.parameters` to initialise the ODE solver.

The module never assumes a file path or external cosmology—everything is pulled from the supplied model instance.

## Numerical specification

Define the ΛCDM-style reference void curve:
\[
R_{\text{void,ref}}(z) = R_{\text{ref},0} (1 + z)^{-\beta},
\]
with \(R_{\text{ref},0} = \texttt{--R_ref_Mpc}\) and \(\beta = \texttt{--beta-z}\).

Growth-based scaling follows:
\[
S_{\text{growth}}(z) =
\begin{cases}
\left(\dfrac{D_{\Lambda\text{CDM}}(z)}{D_{\text{PBUF}}(z)}\right)^{\eta}, & \text{if }\texttt{--compare-lcdm}\\
D_{\text{PBUF}}(z)^{-\eta}, & \text{otherwise}
\end{cases},
\]
with \(\eta = \texttt{--eta-growth}\).

Elastic slack contributes:
\[
S_{\text{elastic}} = 1 + \gamma_\alpha \alpha,
\]
with \(\gamma_\alpha = \texttt{--gamma-alpha}\).

Finally,
\[
R_{\text{void,PBUF}}(z) = R_{\text{void,ref}}(z) \cdot S_{\text{growth}}(z) \cdot S_{\text{elastic}}.
\]

The module samples \(z \in [0, z_{\max}]\) on a uniform grid and evaluates \(D(z)\) either via the model’s `growth_factor` or the shared `cosmos2.kernels.common.growth.solve_growth` integrator.

## Outputs

`run_prediction(model, config)` returns the following payload:

```json
{
  "name": "void-size",
  "version": "v1",
  "summary": "Predicted cosmic void radius R_void(z) with PBUF elastic + growth scaling",
  "results": {
      "zmax": <float>,
      "R_void_z0_Mpc": <float>,
      "R_void_z0p5_Mpc": <float>,
      "R_void_z1_Mpc": <float>,
      "R_ref_z0_Mpc": <float>,
      "ratio_PBUF_over_LCDM_z0": <float or null>,
      "ratio_PBUF_over_LCDM_z0p5": <float or null>,
      "ratio_PBUF_over_LCDM_z1": <float or null>,
      "eta_growth": <float>,
      "gamma_alpha": <float>,
      "alpha": <float>
  },
  "tables": [],
  "plots": [],
  "metadata": {
      "model": "...",
      "compare_lcdm": <bool>,
      "R_ref_Mpc": <float>,
      "eta_growth": <float>,
      "gamma_alpha": <float>,
      "beta_z": <float>,
      "timestamp": "...",
      "points": <int>
  }
}
```

`results` ratio entries are `null` when `--compare-lcdm` is absent. `alpha` surfaces the elastic amplitude fed into the prediction, and `metadata` tracks the model identity plus the sampling/config knobs.

## Tables & plots

When `--output-table`/`output_table` is set, the module emits a `PredictionTable` named `void_radius_vs_z` with columns:

- `z`, `a`
- `D_PBUF`, `D_LCDM` (LCDM column is `null` when the comparison is disabled)
- `S_growth`, `S_elastic`
- `R_PBUF_Mpc`, `R_LCDM_Mpc` (`R_LCDM_Mpc` mirrors the reference curve)
- `ratio`

The table rows mirror the sampling grid so downstream tooling can replot or recreate the curve.

`--output-plot`/`output_plot` produces:

1. `R_void_vs_z`: line plot of `R_PBUF_Mpc` and `R_ref_Mpc` vs \(z\).
2. `void_ratio_vs_z`: line plot of the PBUF/LCDM ratio (only when `--compare-lcdm` is active).

Each plot includes `xlabel`/`ylabel` metadata for reporting.

## Science-run integration

Drop the module into any science run by enabling the `[predictions]` block:

```toml
[predictions]
enabled = true
modules = ["void-size"]

[predictions.void-size]
zmax = 1.0
points = 100
compare_lcdm = true
R_ref_Mpc = 25.0
eta_growth = 0.5
gamma_alpha = 1.0
beta_z = 0.3
output_table = true
output_plot = true
```

When the unified runner executes science configs with this block, it wires the prediction through the standard `PredictionManager` and the reporting pipeline attaches the `void-size` narratives/tables/plots to the final PDF/HTML bundle.

## Error handling

- If the model lacks `alpha`, the module returns `status="error"` with `metadata.error = "missing_alpha"`.
- If no growth factor can be computed (raw model lacks `growth_factor` and the fallback integrator fails), the module returns `status="error"` with `metadata.error = "missing_growth_api"`.
- Invalid CLI values raise `ValueError` so the caller can correct the request.

## Testing checklist

1. `cosmos_cli predict void-size --model pbuf --compare-lcdm --output-table --output-plot` → expects printed `R_void` numbers and ratio plot/table descriptors.
2. `cosmos_cli predict void-size --model lcdm` → fails early with `missing_alpha` metadata.
3. Science-run config with `[predictions]` injecting `"void-size"` → reporting output includes the void-size summary, ratios, and (if requested) the plots/tables.
4. Unit tests ensure `D_{\text{PBUF}} < D_{\Lambda\text{CDM}}` produces ratios > 1 and that the table/plot descriptors are present when the flags are set.
