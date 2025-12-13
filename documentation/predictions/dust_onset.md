# Dust Onset Prediction Module

This module predicts the redshift range where elastic-enhanced enrichment has accumulated enough “dust potential” to allow efficient dust formation. The proxy is a normalized integral of the ε₀-based activity history, optionally modulated by any `star_formation_efficiency(a)` exposed by the model. The onset estimate is directly testable with JWST/ALMA high‑z dusty galaxies.

## CLI interface

### Command

```bash
cosmos_cli predict dust-onset --model pbuf --zmax 30 --points 300
```

### Options

- `--zmax <float>` (default: `30.0`): highest redshift included in the scan; controls the low end of the scale-factor integration.
- `--points <int>` (default: `300`): number of grid samples for the z/a series.
- `--mode <str>` (default: `"simple"`): currently the only supported mode for the dust potential proxy.
- `--threshold-fraction <float>` (default: `0.05`): fraction of the final cumulative dust potential required before announcing an onset.
- `--output-plot`: include the `P_norm_vs_z` plot descriptor in the prediction payload.
- `--output-table`: include the `dust_potential_vs_z` table in the prediction payload.

## Model API expectations

Dust onset uses the same elastic stiffness surface as the metallicity module. The prediction requires:

- `model.elastic_stiffness(a)` or `model.epsilon0_of_T(T)` via `PredictionModelAdapter.elastic_stiffness`.
- Optional `model.star_formation_efficiency(a)` if the model exposes an efficiency surface; otherwise unity weighting is assumed.
- The prediction may reuse any metallicity helpers in the future, but v1 keeps the logic self-contained.

If the elastic stiffness API is missing, the module short-circuits with an error result containing `metadata.error = "missing_elastic_stiffness_api"`.

## Physics and scaling

The effective dust activity function is

```
D(a) = ε₀(a)⁻¹ × S(a)
```

where `S(a)` is the optional star-formation efficiency (defaults to `1`). The accumulated dust potential is

```
P(a) = ∫_{a_min}^{a} D(a') da'
```

with `a_min = 1 / (1 + zmax)` and `P_norm(a) = P(a) / P(1.0)`. Dust onset is defined as `P_norm(a_dust) = f_{crit}` with `f_{crit}` configurable via `threshold_fraction` (defaults to `0.05`).

## Prediction contract

The module implements:

```python
def run_prediction(model, config) -> PredictionResult:
    ...
```

### Config fields

- `zmax` (float): scan upper limit (default `30.0`).
- `points` (int): sample count (default `300`).
- `mode` (str): current value must be `"simple"`.
- `threshold_fraction` (float): dust-onset criterion (default `0.05`).
- `output_plot` / `output_table` (bool): toggle optional descriptors.

### Result structure

`PredictionResult.results` contains:

- `z_dust_on`: onset redshift (`None` if threshold is never reached within the scan).
- `a_dust_on`: corresponding scale factor.
- `threshold_fraction`: configured `f_crit`.
- `P_norm_at_z6`, `P_norm_at_z10`, `P_norm_at_z15`: interpolated normalized potentials at those redshifts.

`PredictionResult.metadata` records:

- `model`: model name.
- `points`, `mode`, `threshold_fraction`: configuration summary.
- `timestamp`: ISO timestamp when the prediction ran.
- `summary`: short narrative (“Elastic-enhanced dust potential crosses …”).

If `--output-table` is enabled, `tables` includes:

```json
{
  "name": "dust_potential_vs_z",
  "columns": ["z", "a", "epsilon0", "D_eff", "P_norm"],
  ...
}
```

If `--output-plot` is enabled, `plots` includes `P_norm_vs_z` with `{"z": ..., "P_norm": ...}` and axis labels.

## Science report integration

Enable the module in any science config by adding it to `[predictions]`. Example:

```toml
[predictions]
enabled = true
modules = ["dust-onset"]

[predictions.dust-onset]
zmax = 30
points = 300
threshold_fraction = 0.05
mode = "simple"
```

The science runner runs the module for every configured model after the fits finish and emits a report section summarizing the threshold_fraction, `z_dust_on`, and the `P_norm(z)` values (z=6,10,15). The aggregated `predictions_summary.json` records the per-model payloads for downstream report generation.

## Error handling

If the model cannot supply `elastic_stiffness`/`epsilon0`, `run_prediction` returns an error `PredictionResult` whose metadata holds `error="missing_elastic_stiffness_api"` and a short summary describing the fallback. Downstream consumers can use this payload to note that dust-onset is unsupported for that model.
