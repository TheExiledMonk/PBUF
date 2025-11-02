# Grid-Based Cosmology Evaluator

This repository now ships with a deterministic, v1-style evaluator that scores
every cosmology on a predefined parameter grid. The new workflow replaces the
older multi-stage “survivor/refine” pipeline entirely.

## Design Principles

1. **Pure functions:** Each dataset evaluator instantiates a fresh LCDM or PBUF
   model from the raw parameter dictionary and returns a χ² value. No global
   caches, no shared state.
2. **Independent cosmologies:** Every point on the grid is treated as a
   standalone universe. There is no survivor pruning, thresholding, or
   cross-contamination between evaluations.
3. **Deterministic execution:** Given a grid definition and dataset list, the
   resulting χ² table is fully reproducible. Parallel execution (optional) never
   changes the outcome.

## Default Parameter Grids

The defaults are deliberately modest to keep runtimes predictable. They can be
overridden via a JSON configuration file.

| Model | Axes (values) |
|-------|---------------|
| LCDM  | H₀ ∈ [64, 72] (5 linear samples), Ωₘ₀ ∈ [0.27, 0.35] (5 linear samples), Ωᵣ₀ = 9.2×10⁻⁵, Ωₖ₀ = 0 |
| PBUF  | H₀ ∈ [64, 72] (4 samples), Ωₘ₀ ∈ [0.27, 0.35] (4 samples), Ωᵣ₀ = 9.2×10⁻⁵, Ωₖ₀ = 0, α ∈ 10^[-4.5,-2.5] (4 log samples), Rmax ∈ 10^[7,10] (4 log samples), k_sat ∈ {0.6, 1.0, 1.4, 2.0} |

Each grid point evaluates the chosen datasets (`cmb`, `sn`, `cc`, `rsd` by
default, with optional BAO sets). Physics checks now operate in binary mode: a
cosmology either passes every guardrail (and is scored) or fails and is excluded
from the ranking. There are no additive penalties or soft biases.

## Running the Evaluator

```bash
python cli.py fit grid --model both
python cli.py fit grid --model pbuf --datasets cmb,sn --grid-config grids/pbuf_dense.json
python cli.py fit grid --model lcdm --include-bao --workers 4 --tag nightly
python cli.py fit grid --model pbuf --refine-top 50 --refine-fraction 0.03 --refine-points 5
```

- `--datasets` accepts comma-separated names or `all`.
- `--include-bao` appends `bao_iso` and `bao_aniso` to the default list.
- `--grid-config` points to a JSON file that either contains explicit axes for
  the requested model or top-level keys `lcdm`/`pbuf`.
- `--workers` enables parallel processing (order-independent).

### Grid Configuration Format

```json
{
  "lcdm": {
    "H0": { "min": 66.0, "max": 70.0, "num": 5, "scale": "linear" },
    "Om0": [0.29, 0.31, 0.33],
    "Ok0": [0.0],
    "Or0": [9.2e-5]
  },
  "pbuf": {
    "H0": [65.0, 67.0, 69.0],
    "Om0": [0.3],
    "alpha": { "min": -5.5, "max": -3.0, "num": 6, "scale": "log" },
    "Rmax": { "min": 6.5, "max": 9.5, "num": 5, "scale": "log" },
    "k_sat": [0.6, 1.0, 1.5, 2.0]
  }
}
```

- Linear axes interpret `min`/`max` as literal values.
- Log axes interpret `min`/`max` as base-10 exponents.
- Lists are taken verbatim.

### Local Refinement

After the coarse grid finishes you can optionally explore a small hypercube
around the best performers without writing custom scripts:

```bash
python cli.py fit grid --model both --refine-top 100 --refine-fraction 0.05 --refine-points 3
```

- `--refine-top` selects how many top-ranked cosmologies seed a local grid.
- `--refine-fraction` defines the ± fractional range applied to each varying
  parameter (e.g., 0.05 → ±5%).
- `--refine-points` is the number of samples per axis inside that local grid.

The refinement results are appended to the main JSON artifact and marked in the
metadata (`origin: "refine"`), so you can easily see whether the global best
model came from the initial grid or a follow-up sweep.

## Output

Each run writes `data/results/grid_<model>_<timestamp>[ _tag].json` with:

- full parameter list per cosmology,
- per-dataset χ² values,
- total χ² and ranking,
- metadata (datasets used, grid axes, timestamp, worker count).

This structure makes it easy to compare LCDM and PBUF best-fit scores or to
inspect the entire table for follow-up analyses.
