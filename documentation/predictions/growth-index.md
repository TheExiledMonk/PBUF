# Growth Index Prediction Module

## Purpose

Computes the redshift-dependent growth index `γ(z)` and the RSD observable `fσ₈(z)` for the PBUF model, highlight­ing Ωₘ-driven departures from ΛCDM, and packaging the results for tagging DESI, BOSS, 6dF, WiggleZ, and Euclid RSDs. The output explicitly quotes `γ(z)` at z=0, 0.5, and 1.0, fσ₈ at the same anchors, the σ₈ normalisation, and optional ΛCDM-like reference ratios.

## CLI

```
cosmos_cli predict growth-index --model pbuf [--zmin 0.0] [--zmax 3.0] [--points 300]
                                      [--compare-lcdm] [--output-table]
                                      [--output-plot]
```

- `--zmin` / `--zmax` control the redshift span (defaults 0.0 / 3.0)
- `--points` selects the grid resolution (default 300)
- `--compare-lcdm` adds a ΛCDM-like reference trace for `γ(z)` and `fσ₈(z)`
- `--output-table` / `--output-plot` trigger table and plot exports used in CLI summaries and reports
- `--output-table` / `--output-plot` trigger table and plot exports used in CLI summaries and reports

The CLI summary mirrors the example in the spec: γ(z) and fσ₈(z) at 0, 0.5, and 1.0, plus ratios against ΛCDM at z=0 when requested, and a list of generated tables/plots.

## Model API

The prediction relies on the high-level adapter that already serves the growth module:

- `model.growth_factor(a)` for the linear growth `D(a)`
- `model.H(z)` to build `E(z)=H(z)/H0`
- `model.Omega_m_of_z(z)` when available (fallback: compute from `Ω_m0` and `E(z)`)
- `model.sigma8` (or the PBUF rule for σ₈ when V11 settings dictate) to anchor `fσ₈`

If any of these contracts are missing, the module returns the structured error stub described below.

## Physics specification

1. **Growth rate `f(z)`**
   - Build the redshift grid between `zmin` and `zmax` with `points` entries.
   - Convert to scale factor `a = 1/(1+z)`.
   - Evaluate `D(a)` via `model.growth_factor(a)` and estimate the logarithmic derivative `f(z)=d ln D / d ln a` with finite differences (central where possible, forward/backward at edges).

2. **Matter fraction `Ωₘ(z)`**
   - Prefer `model.Omega_m_of_z(z)`. Otherwise compute
     ```
     Ωₘ(z)=Ωₘ0 (1+z)^3 / E(z)^2,   E(z)=H(z)/H0
     ```

3. **Growth index `γ(z)`**
   - Step A: apply a Savitzky–Golay filter (order 3, window ≈ 21) to `D(z)` so the subsequent derivative is noise suppressed.
   - Step B: compute `f(z) = d ln D / d ln a` using the smoothed `D(z)` grid.
   - Step C: lightly re-smooth `f(z)` with a narrower SG window (≈ 11) before forming the growth index.
   - Step D: evaluate `γ(z)=ln f / ln Ωₘ` only when both `f(z)` and `Ωₘ(z)` remain positive.
   - Step E: smooth the resulting `γ(z)` once more with the SG filter (window ≈ 11) so the curves stay free of jagged wiggles while still obeying `f≈Ωₘ^γ`.
   - The smoothing parameters (polyorder and window lengths) are recorded in the prediction metadata for reproducibility.

4. **RSD observable `fσ₈(z)`**
   - Compute `σ₈(z)=σ₈,0 D(z)/D(0)` using `model.sigma8`.
   - Then `fσ₈(z) = f(z) σ₈(z)`.
   - Report anchors at z=0, 0.5, and 1.0; include full table/plot if requested.

5. **ΛCDM reference (`--compare-lcdm`)**
   - Construct a minimal ΛCDM model sharing `H0`, `Ωₘ0`, `Ωᵣ0` but with `Ωσ(a)=0` to recompute `D_LCDM`, `f_LCDM`, `γ_LCDM`, and `fσ₈_LCDM`.
   - Emit ratio curves
     ```
     R_γ(z)=γ_LCDM(z) / γ_PBUF(z),
     R_fσ₈(z)=fσ₈_LCDM(z) / fσ₈_PBUF(z)
     ```
   - Only report ratios when both numerator and denominator exist.

## Outputs

### `run_prediction` contract

```json
{
  "name": "growth-index",
  "summary": "Growth index γ(z) and fσ8(z) compared to a ΛCDM-like reference",
  "results": {
      "zmin": <float>,
      "zmax": <float>,
      "gamma0": <float>,
      "gamma0p5": <float>,
      "gamma1": <float>,
      "f0": <float>,
      "f0p5": <float>,
      "f1": <float>,
      "fs8_0": <float or null>,
      "fs8_0p5": <float or null>,
      "fs8_1": <float or null>,
      "sigma8_0": <float or null>,
      "gamma_LCDM_0": <float or null>,
      "ratio_gamma_0": <float or null>,
      "ratio_fs8_0": <float or null>
  },
  "tables": [
      {
          "name": "growth_index_vs_z",
          "columns": ["z", "a", "D", "f", "Omega_m", "gamma", "D_LCDM",
                      "f_LCDM", "gamma_LCDM", "ratio_gamma"],
          "data": [...]
      },
      {
          "name": "fsigma8_vs_z",
          "columns": ["z", "D", "f", "sigma8", "fsigma8",
                      "fsigma8_LCDM", "ratio_fsigma8"],
          "data": [...]
      }
  ],
  "plots": [
      {
          "name": "growth_index_plot",
          "type": "line",
          "x": [...],
          "y": [...],
          "xlabel": "redshift z",
          "ylabel": "γ(z)"
      },
      {
          "name": "fsigma8_plot",
          "type": "line",
          "x": [...],
          "y": [...],
          "xlabel": "redshift z",
          "ylabel": "f σ₈(z)"
      }
  ],
  "metadata": {
      "model": <model_name>,
      "compare_lcdm": <bool>,
      "points": <int>,
      "smoothing": {
          "savgol_polyorder": <int>,
          "D_window": <int>,
          "f_window": <int>,
          "gamma_window": <int>
      },
      "timestamp": "..."
  }
}
```

Metadata records the SG smoothing parameters so the published γ(z) curve can be reconstructed exactly.

Tables and plots are optional and only emitted when `--output-table` / `--output-plot` are supplied (and when valid data exists).

### CLI summary example

Follows the prototype in the spec, showing `γ(0)`, `γ(0.5)`, `γ(1.0)`, `fσ₈` anchors, optional ΛCDM γ(0) / ratio, and the list of tables/plots.

## Reporting and Science Runner

Enable the module by adding `"growth-index"` to `[predictions].modules`, then set the desired parameters:

[predictions.growth-index]
zmin = 0.0
zmax = 3.0
points = 300
compare_lcdm = true


The generated payload already matches the reporting schema, so reports can insert the plots/tables into the growth-index section mentioned in the request.

## Errors

If the model lacks the necessary growth/background API, return:

```json
{
  "name": "growth-index",
  "summary": "Growth index prediction unsupported (missing growth or background API).",
  "metadata": {"error": "missing_growth_or_background"}
}
```

If `σ₈` is missing, the `fsigma8` entries in `results` are set to `null`, but the growth-index values still return normally.

## Testing considerations

- Ensure `γ(z)` gently decreases at low `z` (PBUF signature of weaker late-time gravity).
- Verify `fσ₈(z)` stays suppressed relative to the ΛCDM reference (when requested) to match the S₈ tension narrative.
- Check tables/plots generate clean payloads with no `None` values for valid points.
