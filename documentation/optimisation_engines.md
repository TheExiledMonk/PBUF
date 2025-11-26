# Optimisation engines (cosmos2 CLI)

The `cli.py optimise` command now runs entirely on the cosmos2 sampler stack. Bounds still come from `configs/basin_walker/<model>_bounds.json`, but proposal generation and χ² evaluation use the cosmos2 BasinWalker plus fit registry.

## Engines
- `grid_search` seeds the walker with a coarse grid over the first few parameters (`--samples` sets the grid density) and evaluates a single batch of proposals. Use `--scatter` to control batch size.
- `basin` runs multiple batches (`--seeds`) with optional Latin-hypercube scatter (`--scatter`) and local refinement per batch (`--refine`). Proposals flow through the same cosmos2 BasinWalker used by the threaded science runner.
- Both engines call the cosmos2 fit registry (`cmb`, `sn`, `bao_iso`, `bao_aniso`, `cc`, `rsd`, `wl_s8`, `lensing_cross`, `galaxy_pk`, `sh0es`) with optional weights from `--dataset-weight`.

## Diagnostics & sanity
- `--sanity` uses the same cosmos2 evaluator and reports per-dataset χ². PBUF still reuses the legacy thermal table via the LUT bridge when instantiating models.
- Phase-6a/7a gating remains inside the kernels; invalid candidates return `inf` χ² and are skipped automatically.
