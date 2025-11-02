# `python cli.py fit coord`

The coordinate basin walker drives 1‑D sweeps across each cosmological parameter to map out χ² basins and (optionally) perform parallel sampling of the viable “island” interior. This reference collects every flag accepted by the CLI entry point so you can script runs without digging through `cli.py`.

## Quick usage

```bash
# Basic LCDM run using the built-in dataset bundle
python cli.py fit coord \
  --output data/results/basin_scan_lcdm.json \
  --model lcdm

# PBUF with explicit datasets, 16 workers, convergence cycles, and basin sampling
python cli.py fit coord \
  --output data/results/basin_scan_pbuf.json \
  --model pbuf \
  --datasets cmb,sn_pantheon,bao_iso,bao_aniso,cc,rsd,sn_sh0es \
  --max-workers 16 \
  --converge --max-cycles 6 --improvement-tol 0.01 \
  --island-samples 200 --island-delta 20 --island-seed 42
```

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model {pbuf,lcdm}` | `pbuf` | Cosmological family to optimize. |
| `--datasets <csv>` | *(auto)* | Comma-separated list of dataset aliases (e.g. `cmb,sn_pantheon`). If omitted, a model-aware default bundle is chosen (`cmb,sn_pantheon,bao_iso` plus optional BAO extras when `--include-bao` is present). |
| `--include-bao` | off | Adds both `bao_iso` and `bao_aniso` to the dataset list (deduped). |
| `--phase6a` | off | Enforces Phase 6a physics validation during scans. |
| `--delta-chi2 <float>` | `20.0` | Defines the χ² band (`min χ² + Δ`) used for basin width reporting. |
| `--output <path>` | *(required)* | Destination JSON written after the run completes (contains scans, fiducial evaluation, and optional island summary). |
| `--seed-json <path>` | *(unset)* | Seed parameters JSON to override the reference starting point. |
| `--eps0 <float>` | *(unset)* | Convenience override for the PBUF elastic stiffness baseline when seeding. |
| `--skip-second-pass` | off | Disables the tightening sweep (only coarse pass executes). |
| `--quiet` | off | Suppresses walker logging (still prints the CLI summary). |
| `--no-progress` | off | Turns off tqdm progress bars even when available. |

### Parallel / convergence controls

| Flag | Default | Description |
|------|---------|-------------|
| `--max-workers <int>` | *(auto → `min(os.cpu_count(),16)`)* | Caps the process pool used for parallel axis scans and island sampling. `1` forces serial execution. |
| `--converge` | off | Enables the multi-cycle refinement loop. |
| `--max-cycles <int>` | `6` | Maximum refinement cycles when `--converge` is active (minimum 1). |
| `--improvement-tol <float>` | `1e-2` | Early-stop threshold: halt when the χ² improvement per cycle drops below this value. |

### Island (basin interior) sampling

These flags trigger the optional “island center” stage which samples the hyper-rectangle defined by the pass‑2 scan edges, keeps models within `min χ² + Δ`, and reports the most interior viable point.

| Flag | Default | Description |
|------|---------|-------------|
| `--island-samples <int>` | `0` (disabled) | Number of random samples to cast inside the basin. Values ≳150 are suggested for 5D PBUF runs. |
| `--island-delta <float>` | `20.0` | χ² window (`min χ² + Δ`) that defines the viable core. |
| `--island-seed <int>` | *(unset)* | Optional RNG seed for reproducible sampling. |

## Output structure (high level)

The generated JSON always includes:

- `fiducial_params` – optimal parameters found by the coordinate sweeps.
- `fiducial_chi2` – χ² at the fiducial point, with dataset breakdowns if available.
- `axis_scans` – one entry per parameter sweep with per-point status, best value, and basin edges.
- `convergence` – present only when `--converge` is enabled (cycle history and stop reason).
- `island_center` – present only when `--island-samples > 0`; records sample counts, χ² thresholds, and the most interior viable parameter set.

## Tips

- Use `--skip-second-pass` only when probing very broad basins; otherwise the tightening sweep substantially improves the island estimate.
- When you plan to run `--island-samples`, keep `--max-workers` above `2` to accelerate both the axis scans and the post-run sampling.
- The JSON is self-contained: you can feed `island_center.center_params` back into downstream pipelines (e.g. `cli.py fit joint`) to stress-test a single cosmology across dataset bundles.
