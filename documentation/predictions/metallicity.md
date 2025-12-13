# Metallicity Evolution Prediction

PBUF predicts that the softer early vacuum drives faster collapse and hotter starbursts, producing more rapid metal enrichment than a naive LambdaCDM-style background. The metallicity prediction module converts the elastic stiffness history into an effective enrichment efficiency, integrates the resulting `Z(a)` curve, and normalizes `Z(z)` relative to today so you can directly compare early growth between PBUF and a reference model.

## Physics overview

- Elastic softness enters through `epsilon0(a)`, which is read from `model.elastic_stiffness(a)` (or `epsilon0_of_T`/thermal tables for legacy models).
- The effective enrichment efficiency is `E(a) ∝ 1/epsilon0(a)`, optionally sharpened by `model.star_formation_efficiency(a)` when that helper exists.
- `Z(a)` is computed as the cumulative integral of `E(a)` from the maximum redshift down to today and normalized so `Z(a=1)=1`.
- The LambdaCDM reference uses `E_{LambdaCDM}(a) ∝ a^beta` with `beta~1` as a slow, smooth enrichment; the metallicity boost `Z_PBUF/Z_{LambdaCDM}` captures the early-time amplification.

## CLI usage

```bash
python cosmos_cli.py predict metallicity \
  --model pbuf \
  --zmin 2 --zmax 20 --points 200 \
  --mode simple \
  [--compare-lcdm] \
  [--output-plot] \
  [--output-table]
```

Options:

- `--zmin`, `--zmax`, `--points`: define the redshift grid on which `Z(z)` and `epsilon0(z)` are evaluated (defaults 2->20 with 200 steps).
- `--mode`: currently only `"simple"` is implemented; other modes will be recognised for future work.
- `--compare-lcdm`: compute the LambdaCDM-like reference curve and the PBUF boost (required for the boost plot/table).
- `--output-plot`, `--output-table`: include the `metallicity_vs_z` table, optional `metallicity_vs_z_lcdm` table, the `Z_rel_vs_z` plot, and the `Z_boost_vs_z` plot based on the computed series.

CLI output summarises:

```
[cosmos_cli] Prediction metallicity (v1) for model PBUF
  z range: 2.0 -> 20.0
  Z(z=2) / Z(0): 0.45
  Z(z=6) / Z(0): 0.18
  Z(z=10) / Z(0): 0.09
  metallicity boost vs LCDM:
    at z=6: 1.7x
    at z=10: 2.3x
  metadata: points=200, mode=simple, compare_lcdm=True
  tables: metallicity_vs_z, metallicity_vs_z_lcdm
  plots: Z_rel_vs_z, Z_boost_vs_z
```

## Outputs

- **Results**: numeric fields include `zmin`, `zmax`, `Z_over_Z0_at_z2`, `Z_over_Z0_at_z6`, `Z_over_Z0_at_z10`, `boost_vs_lcdm_at_z6`, and `boost_vs_lcdm_at_z10` (the last two appear when `--compare-lcdm` is set). The CLI also exposes human-readable strings like `z range` and `Z(z=2) / Z(0)` so the summary lists the key observables.
- **Tables** (when `--output-table`): `metallicity_vs_z` contains columns `z`, `a`, `epsilon0`, `E_eff`, `Z_rel`; `metallicity_vs_z_lcdm` contains `z` and `Z_rel_lcdm` and is emitted only with `--compare-lcdm`.
- **Plots** (when `--output-plot`): `Z_rel_vs_z` shows the normalized metallicity for the requested grid; `Z_boost_vs_z` plots the PBUF/LCDM ratio and requires `--compare-lcdm`.
- **Metadata**: every run stamps `model` (class name), `points`, `mode`, `compare_lcdm`, `timestamp`, and a short `summary`.

## Science report integration

Enable predictions in a science config to run the metallicity module alongside fits:

```
[predictions]
enabled = true
modules = ["metallicity"]

[predictions.metallicity]
zmin = 2
zmax = 12
points = 200
mode = "simple"
compare_lcdm = true
```

The unified report section renders the metallicity curves, boost plot, and numerical highlights. A typical narrative is:

> **Metallicity Evolution Prediction**  
> Using elastic-enhanced enrichment, PBUF predicts faster buildup of heavy elements at high redshift:  
> `Z(z=6) ~ 0.18 Z(0)`, vs ~0.10 in the simple LambdaCDM reference (~1.8x boost).  
> `Z(z=10) ~ 0.09 Z(0)`, vs ~0.04 in the reference model (~2.3x boost).  
> (Include `Z(z)/Z(0)` and `Z_BOOST` plots.)  
> This is directly testable with JWST + ALMA emission-line metallicities at z >= 6.

## Error handling

If the selected model cannot expose `elastic_stiffness`, `epsilon0_of_T`, or the legacy thermal tables, the module returns an error prediction with `metadata.error = "missing_elastic_stiffness_api"` and `metadata.summary` describing the missing API instead of raising. The metallicity plot/table output is suppressed in that case.
