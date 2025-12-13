# GW Propagation Prediction

Predict how gravitational waves and photons traverse the rigid PBUF medium. The module integrates the model Hubble `H(z)` together with the elastic-wave-speed estimate to build the photon and GW luminosity distances, the arrival-time difference, and the instantaneous `c_GW/c_EM` ratio. In this v1 implementation `c_GW = c_EM` everywhere, so PBUF predicts `D_L^{GW}(z)=D_L^{EM}(z)` and `Delta_t_{GW-EM}(z)=0` while still exposing machinery for future delta_c deviations.

## CLI flags

```bash
python cosmos_cli.py predict gw-propagation --model <model> [options]
```

| Flag | Purpose |
| --- | --- |
| `--zmax <float>` | Maximum redshift for the integration (default 5.0). |
| `--points <int>` | Number of samples between 0 and `zmax` (default 200). |
| `--z-key "<csv>"` | Comma-separated redshifts whose summaries appear in `results[z_keys]`. |
| `--anchor-equal-c0` | Enforce `c_EM(0)=c_GW(0)` (default). |
| `--no-anchor-equal-c0` | Allow a different local normalization. |
| `--output-table` | Emit the canonical `gw_propagation_vs_z` table (`DL`, `t`, `R_D`, `Delta_t`, `R_c`). |
| `--output-plot` | Include `DL_ratio_vs_z` and `Delta_t_vs_z` plots in the payload. |

## Prediction payload

The module always returns `results` structured as:

```python
{
  "zmax": float,
  "points": int,
  "z_keys": [z1, z2, ...],
  "RD_at_z": [R_D(z1), ...],
  "DL_EM_at_z": [...],
  "DL_GW_at_z": [...],
  "Delta_t_GW_EM_at_z": [...],  # in seconds
  "Rc_at_z": [...],             # c_GW / c_EM
}
```

`metadata` includes the familiar `"model"`, `"timestamp"`, `"summary"` string, and the booleans `anchor_equal_c0` plus `used_wave_speed` that document whether the elastic-wave-speed API contributed the shape of `c_EM(z)`. When `--output-table` is requested the `gw_propagation_vs_z` table records the full redshift grid with the columns `z`, `DL_EM_Mpc`, `DL_GW_Mpc`, `R_D`, `t_EM_s`, `t_GW_s`, `Delta_t_s`, and `R_c`. Enabling `--output-plot` yields the `DL_ratio_vs_z` and `Delta_t_vs_z` line descriptors.

## Physics summary

The EM luminosity distance is built from

```
D_L^{EM}(z) = (1 + z) times the integral from 0 to z of [c_EM(z') / H(z')] dz' ,
```

with `c_EM(z) = c * f(z)` taken from the wave-speed module when available (otherwise `f(z)=1`). Likewise the GW distance uses `c_GW(z) = c_EM(z) (1 + delta_c(z))` and the arrival times integrate 1/((1+z)H) times the appropriate speed ratio. For v1 `delta_c(z) = 0`, so `R_D(z)=1`, `Delta_t_{GW-EM}(z)=0`, and `R_c(z)=1` across the sampled redshift range. This prediction therefore reiterates PBUF's strong assertion that the gravitational and electromagnetic distance ladders must agree, matching the GW170817 constraint.

## Science-run integration

Enable the module in science-run configs via the `[predictions]` section. Example `pbuf.toml` snippet:

```toml
[predictions]
enabled = true
modules = ["gw-propagation"]

[predictions.gw-propagation]
zmax = 5.0
points = 200
z_key = "0.01,0.1,0.5,1.0,2.0"
anchor_equal_c0 = true
```

The runner applies the same options per model, so both PBUF and LCDM can produce the shared `R_D=1` baseline in v1 while keeping the door open for a future elastic-sector GW coupling.

## Error handling

-- Missing `H(z)` -> the module signals `metadata["error"]="missing_H_api"` and `status="error"` (the prediction cannot proceed without the expansion history).
-- Missing wave-speed/stiffness -> `used_wave_speed=false`, `c_EM(z)=c_GW(z)=c`, and the rest of the calculation continues with constant light speed.
