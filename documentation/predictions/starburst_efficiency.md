# Starburst Efficiency Prediction Module

PBUF predicts relative starburst amplification stems from elastic-soft vacuum collapse and faster enrichment: the
collapse amplification `C(z) ∝ ε₀(z)^{-1/2}` combines with the metallicity boost `M(z) = 1 + γ Z_rel(z)` to give
`S(z) = C(z) ⋅ M(z)` relative to today’s reference SFR. This module exposes those trends for CLI/pipeline use,
compares against a slow ΛCDM-like reference when requested, and directly ties into ALMA/JWST high-redshift dusty
star-forming galaxy observations.

## CLI interface

### Command

```bash
cosmos_cli predict starburst-efficiency \
  --model pbuf \
  --zmax 20 \
  --points 300
```

### Options

- `--zmax <float>` (default `20`): highest redshift in the scan (controls the low end of the scale factor integral).
- `--points <int>` (default `300`): number of redshift samples for the history.
- `--gamma-metallicity <float>` (default `1.0`): γ controls how much the normalized metallicity history boosts the collapse-driven efficiency.
- `--beta-lcdm <float>` (defaults to γ): β tunes the ΛCDM reference amplification `S_LCDM(z)=1+β Z_rel_LCDM(z)`.
- `--compare-lcdm`: include the reference curve and the PBUF/LCDM ratio in the output (plots, tables, and results).
- `--output-plot`: emit the `S_vs_z` plot (plus `S_ratio_vs_z` when comparing) inside the prediction payload.
- `--output-table`: write the `starburst_efficiency_vs_z` table with per-z collapse/metallicity/S values.

CLI summary (illustrative):

```
[cosmos_cli] Prediction starburst-efficiency (v1) for model PBUF
  S(z=6): 2.1
  S(z=10): 3.4
  S(z=15): 4.8
  Peak S ~ 5.2 at z ~ 17.4
  compared to LCDM:
    S_LCDM(z=6): 1.3
    S_LCDM(z=10): 1.5
  ratio PBUF/LCDM at z=10: 2.25
  plots: S_vs_z, S_ratio_vs_z
  tables: starburst_efficiency_vs_z
```

## Physics summary

- **Collapse amplification**: `C(z)=ε₀(z)^{-1/2}` with `ε₀(a)` exposed by `model.elastic_stiffness(a)` (or via `epsilon0_of_T`/thermal tables). Smaller stiffness at high z makes collapse easier.
- **Enrichment boost**: integrate the normalized effective enrichment `∝ 1/ε₀(a)` (optionally modulated by `model.star_formation_efficiency(a)`) from `a_min = 1/(1+zmax)` to today to build `Z_rel(z)=Z(z)/Z(0)`. The metallicity lever `M(z)=1+γ Z_rel(z)` doubles the efficiency when fully enriched (`γ=1`).
- **Reference ΛCDM**: `Z_rel_LCDM(a)` is a slow integral over `E_LCDM(a) ∝ a`, normalized so `Z_rel_LCDM(a=1)=1`. `S_LCDM(z)=1+β Z_rel_LCDM(z)` with `β` defaulting to `γ`.
- **Starburst efficiency**: `S(z)=C(z)⋅M(z)` optionally multiplied by `f_gas(z)` (unity in v1). When `--compare-lcdm` is active, the output also reports `S(z)/S_LCDM(z)` and the `S_ratio_vs_z` plot.

## Model API expectations

1. `model.elastic_stiffness(a)` (or `model.epsilon0_of_T(T)`/thermal tables via the adapter) to build `ε₀(a)` and `C(z)`.
2. Optional `model.star_formation_efficiency(a)` to weight the enrichment integral; defaults to `1` if absent.
3. No additional APIs are required, but future models can expose `metallicity(z)` if a quicker relative-metallicity surface becomes available.

If the stiffness metric is missing, the prediction returns an error payload with `metadata.error="missing_stiffness_or_metallicity_api"` instead of raising.

## Outputs

- **Results**
  - `zmax`, `S_at_z6`, `S_at_z10`, `S_at_z15`
  - `peak_S_value`, `peak_S_redshift`
  - `S_over_LCDM_at_z6` and `S_over_LCDM_at_z10` (when `--compare-lcdm`)
  - `metadata` records `model`, `points`, `zmax`, `compare_lcdm`, `gamma_metallicity`, `beta_lcdm`, `timestamp`, and a short `summary`.

- **Tables** (`--output-table`)
  - `starburst_efficiency_vs_z`: columns `["z","a","epsilon0","Z_rel","C","M","S","S_LCDM","S_ratio"]`. `S_LCDM` and `S_ratio` are `null` when `--compare-lcdm` is not set.

- **Plots** (`--output-plot`)
  - `S_vs_z`: canonical `S(z)` series plus the LCDM reference when requested.
  - `S_ratio_vs_z`: `S(z)/S_LCDM(z)` trend (only present with `--compare-lcdm`).

## Science report integration

Enable the module in `[predictions]` blocks so the unified runner attaches the starburst narrative to the report:

```toml
[predictions]
enabled = true
modules = ["starburst-efficiency"]

[predictions.starburst-efficiency]
zmax = 20
points = 300
compare_lcdm = true
gamma_metallicity = 1.0
beta_lcdm = 1.0
output_plot = true
output_table = true
```

Reports can then decorate the section with:

> **Starburst Efficiency Prediction**  
> PBUF predicts amplified early starbursts driven by vacuum softness and enrichment:  
> `S(z=6) ≈ …`, `S(z=10) ≈ …`, `S(z=15) ≈ …`, with a peak of `S ≈ …` near `z ≈ …`.  
> When compared to ΛCDM, ratios of ≈2–4× (z=6–12) are teased out in the `S_ratio_vs_z` plot.

## Error handling

Missing stiffness/thermal APIs produce an error `PredictionResult` with:

```json
{
  "name": "starburst-efficiency",
  "metadata": {
    "error": "missing_stiffness_or_metallicity_api",
    "summary": "Starburst efficiency prediction unsupported (missing stiffness or metallicity API)."
  },
  "status": "error"
}
```

This lets downstream runners/logging note that a model cannot support the starburst calculation.
