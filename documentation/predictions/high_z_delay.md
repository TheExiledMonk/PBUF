# High-z Propagation Delay Prediction

PBUF is elastic in the early universe, so wave packets (photons, GWs, neutrinos, etc.) accumulate an extra propagation time when the effective wave speed \(c_\mathrm{eff}(z)\) softens relative to the light speed \(c\). This prediction module compares the travel time from redshift \(z\) to today against a constant-\(c\) baseline while reusing the model's native expansion history \(H(z)\).

## Physics overview

- The effective wave speed is \(c_\mathrm{eff}(z) = c \cdot \epsilon_0(z)\), where \(\epsilon_0(z)\) is fetched from `model.elastic_stiffness(a)` (or via `epsilon0_of_T`/thermal tables) and \(a = 1/(1+z)\).
- The comoving propagation delay is
  \[\Delta t(z) = \int_0^z \left(\frac{1}{c_\mathrm{eff}(z')} - \frac{1}{c}\right) c \, \frac{dz'}{(1+z')H(z')} = \int_0^z \left(\frac{1}{\epsilon_0(z')} - 1\right) \frac{dz'}{(1+z')H(z')}.\]
- v1 keeps the expansion rate from the selected model as the reference (constant-\(c\) propagation through the same \(H(z)\)), so the delay arises purely from the elastic wave speed modulation.

## CLI usage

```bash
cosmos_cli predict high-z-delay --model pbuf --zmax 20 --points 400
```

Options:

- `--zmax`: maximum redshift to scan (default 20).
- `--points`: number of samples between \(z=0\) and `zmax` (default 400).
- `--reference`: name of the constant-\(c\) baseline; v1 currently only accepts `same-H-constant-c` so \(H_\mathrm{ref}(z)=H(z)\).
- `--zgrid`: comma-separated list of redshifts whose Delta t values are promoted into the summary result (defaults to `1,3,6,10,20`).
- `--output-plot`: include the canonical `delta_t_vs_z` and `c_eff_over_c_vs_z` descriptors.
- `--output-table`: export the `propagation_delay_vs_z` table (columns `z`, `a`, `H`, `c_eff_over_c`, `delta_t`).

CLI summary output looks like:

```
[cosmos_cli] Prediction high-z-delay (v1) for model PBUF
  zmax: 20.0
  points: 400
  max_delay_Gyr: 0.150
  delay_at_z1_Gyr: 0.002
  delay_at_z3_Gyr: 0.010
  delay_at_z6_Gyr: 0.035
  delay_at_z10_Gyr: 0.080
  delay_at_z20_Gyr: 0.150
  metadata: model=PBUFModel, timestamp=..., reference=same-H-constant-c, zgrid=[1.0,3.0,6.0,10.0,20.0]
  tables: propagation_delay_vs_z
  plots: delta_t_vs_z, c_eff_over_c_vs_z
```

```
PBUF predicts that light from z=10 arrives about 0.08 Gyr later than you would infer assuming a strictly constant wave speed in an infinitely rigid vacuum.
```

## Outputs

- **Results**: always include `zmax`, `points`, `max_delay_Gyr`, and a `delay_at_z<N>_Gyr` entry for each value from `zgrid`. Values are in gigayears.
- **Tables**: `propagation_delay_vs_z` contains the full grid with `H` in km/s/Mpc, `c_eff_over_c`, and `delta_t` in Gyr.
- **Plots**: `delta_t_vs_z` (extra travel time) and `c_eff_over_c_vs_z` (effective wave speed) are emitted when `--output-plot` is set.
- **Metadata**: includes `model`, ISO `timestamp`, `zmax`, `points`, `reference`, and the resolved `zgrid`. Consumer-friendly strings are available in the format `delay_at_z<N>_Gyr` so the CLI summary can highlight a handful of illustrative redshifts.

## Science report integration

Enable the module alongside fits by adding it to the `[predictions]` section of any science config:

```toml
[predictions]
enabled = true
modules = ["high-z-delay"]

[predictions.high-z-delay]
zmax = 20
points = 400
```

The report renders a dedicated "High-z Propagation Delay Prediction" entry with the sampled Delta t values plus the two canonical plots. Typical narrative:

> **High-z Propagation Delay Prediction**  
> Using the elastic wave speed \(c_\mathrm{eff}(z)\), PBUF predicts an extra travel time for high-redshift signals (photons, GWs, etc.) compared to a constant-c reference:  
> Delta t(z=1) ~ ... Gyr  
> Delta t(z=3) ~ ... Gyr  
> Delta t(z=6) ~ ... Gyr  
> Delta t(z=10) ~ ... Gyr  
> Delta t(z=20) ~ ... Gyr  
> [Insert Delta t(z) plot]  
> [Insert c_\mathrm{eff}(z)/c plot]  
> This can be tested by future high-z SN, GRB, and standard siren timing measurements.

## Error handling

If the selected model cannot expose `elastic_stiffness(a)`, `epsilon0_of_T`, or the legacy thermal tables, the module returns a failed result with `metadata.error = "missing_wave_speed_api"`, a short summary explaining the missing API, and no tables or plots. The unified science workflow can still show the prediction entry with the status note so downstream readers know the module is unsupported for that model.
