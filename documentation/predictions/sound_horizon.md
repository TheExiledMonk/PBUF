# Prediction Module Dev Doc — sound-horizon

This document explains how the `sound-horizon` prediction module plugs into the new prediction registry (`cosmos2/predictions/registry.py`) and the unified science-run system (`cosmos2/science_runner/unified_runner.py`). The module lives under `cosmos2/predictions/modules/sound_horizon.py`, registers itself through the `@register_prediction` decorator, and exports metadata that both the standalone `cosmos_cli predict` group and the unified runner consume.

## Purpose
Compute the baryon sound horizon scale \(r_d\) for any registered model and compare it to a ΛCDM reference when the flag is requested. PBUF expectations are a slightly smaller \(r_d\) (≈2–3%) because of temperature-dependent elasticity and altered photon-baryon wave propagation.

This prediction is critical enough that the unified science runner can emit an explicit **Sound Horizon Prediction** section in the final PDF/HTML report when `[predictions]` is enabled in the science config.

## CLI usage
The `predict` subcommand in `cli/main.py` wires every module found via `predictions_available()` into the CLI tree. Run the sound-horizon module with:

```
cosmos_cli predict sound-horizon --model pbuf --param H0=74 --param Rmax=9e7
```

Save outputs with `--save-json result.json`, `--save-table path/`, and `--save-plots path/` if desired. The CLI always persists a `PredictionResult` payload (see `cosmos2/predictions/structures.py`). Logs indicate the selected module and the model it targeted.

## Optional CLI/Config arguments
The module exposes the following knobs:

- `--resolution <int>`: number of integration points (default `2000`). Uses a log-spaced grid in scale factor \(a\).
- `--compare-lcdm`: triggers a second evaluation using the LCDM module and stores the fractional difference \(\Delta r_d / r_d\) in the results.
- `--output-plot`: requests the canonical integrand plot.
- `--output-table`: requests the integrand table.
- Module-specific overrides can be passed via `--param key=value` so the CLI resolves the model before the prediction runs.

When the unified runner runs predictions, these flags map to the `predictions.module_configs.sound-horizon` section inside the `[predictions]` block (see next section).

## Model inputs
The `PredictionModelAdapter` (see `cosmos2/predictions/model_api.py`) exposes the minimal API the module expects:

- `H(a)` through `PredictionModelAdapter.background`.
- `sound_speed(a)` if provided by the model; otherwise fall back to the computed speed in this module.
- `temperature(a)` when the model derives \(\rho_\gamma(a)\) or \(R(a)\) from a thermal history.
- `drag_redshift()` or `drag_scale_factor()`; the module should look for either and fall back to a standard approximation when both are absent.

No path assumptions are necessary because the adapter already normalises the interface.

## Numerical specification
The core integral is
\[
\displaystyle r_d = \int_0^{a_\text{drag}} \frac{c_s(a)}{a^2 H(a)}\, da,
\]
with \(a_\text{drag}\) provided by the model or the fallback drag calculation.

Implementation notes:

1. Build a log-spaced grid in \(a\) (e.g., `np.logspace(log10(a_min), log10(a_drag), resolution)`).
2. Evaluate \(H(a)\), \(c_s(a)\), and the integrand \(c_s/(a^2 H)\) on that grid.
3. Integrate using `trapz` or Simpson’s rule so the implementation stays model neutral.
4. Store both `r_d` in Mpc and in km (`r_d_Mpc`, `r_d_km`).

Sound speed rules:

- If `model.sound_speed` exists, use it directly.
- Otherwise compute
\[
R(a) = \frac{3 \rho_b(a)}{4 \rho_\gamma(a)}, \qquad c_s(a) = \frac{c}{\sqrt{3 (1 + R(a))}}.
\]

Compute \(\rho_b(a)\) and \(\rho_\gamma(a)\) from today's \(\Omega_{b0}\) and \(\Omega_{\gamma0}\) or from `model.temperature(a)` if the model exposes a thermal history. The adapter already normalises scale factors and redshifts, so the module just needs to multiply by the proper densities.

## Outputs
`run_prediction(model, config)` must return a `PredictionResult` with these primary fields:

```json
{
  "name": "sound-horizon",
  "summary": "Baryon sound horizon r_d predicted by the model",
  "results": {
      "r_d_Mpc": float,
      "r_d_km": float,
      "delta_vs_lcdm": float | null,
      "a_drag": float,
      "z_drag": float
  },
  "tables": [...],
  "plots": [...],
  "metadata": {
      "resolution": int,
      "model": str,
      "timestamp": "ISO string"
  }
}
```

All fields are optional except `name`, `summary`, and `results`. The metadata block should record the grid resolution and model identity; the `timestamp` kicks off reporting/logging hooks.

## Tables & plots
If `--output-table` (or the config flag) is set, emit a `PredictionTable` called `sound_horizon_integrand` with columns `a`, `H(a)`, `c_s(a)`, and `integrand`. Rows must mirror the integration grid so downstream tooling can reconstruct or replot the integrand.

If `--output-plot` is requested, emit at least one `PredictionPlot` named `sound_horizon_curve` with data:

- `x`: scale factor `a`.
- `y`: integrand `c_s(a)/(a^2 H(a))`.
- Labels: `xlabel`: `scale factor a`, `ylabel`: `c_s / (a^2 H)`.

The module can optionally add a second plot showing the cumulative integral vs. `a` to illustrate where most of the contribution arises.

## Optional LCDM comparison
When `--compare-lcdm` is passed, load the LCDM model through the same registry, compute its \(r_d\), and return
\[
\Delta r_d / r_d = \frac{r_{d,\text{model}} - r_{d,\Lambda\text{CDM}}}{r_{d,\Lambda\text{CDM}}}.
\]

Store this value under `results.delta_vs_lcdm`. If the LCDM model is unavailable (e.g., missing dependencies), record `null` and log a warning instead of letting the module crash.

## Science-run integration
The unified runner picks up predictions whenever the science config includes:

```json
"predictions": {
  "enabled": true,
  "modules": ["sound-horizon"],
  "module_configs": {
    "sound-horizon": {
      "resolution": 2000,
      "compare_lcdm": true,
      "output_plot": true,
      "output_table": true
    }
  }
}
```

`cosmos2/science_runner/unified_runner.py` instantiates `PredictionManager`, persists each `PredictionResult` under `predictions/<module>/<model>/result.json`, and writes `predictions_summary.json` with timestamps, module list, and per-model payloads. The final report builder (`reporting_system/core/report_generator.py`) renders a **Sound Horizon Prediction** section (it references `reporting_system/core/panel_builders.py` for layout) showing `r_d` for both the model and LCDM plus `Δr_d` percent.

## Error handling
- If the model lacks the required Hubble, sound-speed, or drag helpers, mark the result as `status="error"` and include a `metadata["not_supported"]` entry describing the missing surface instead of raising. That lets reports show a “not supported” block.
- If the integration fails partway, return partial `results`/`tables` and retain `status="error"` (the unified runner already wraps every call in a try/except and logs the exception).
- The unified runner should never crash because of a prediction module; it already wraps calls in `PredictionResult` guards and writes a summary even when one module fails.

## Testing steps
1. `cosmos_cli predict sound-horizon --model pbuf` → expect a printed `r_d` plus saved JSON/table/plot (if requested).
2. `cosmos_cli predict sound-horizon --model lcdm --compare-lcdm` → prints both `r_d` values and the computed `Δr_d` percent.
3. Update a science-run config to include a `[predictions]` section (see above), run `python cosmos_cli.py science --config <path>` and confirm the generated report includes a **Sound Horizon Prediction** block with the model/LCDM values and percent difference.
