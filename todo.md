# PBUF Rebuild Plan (cosmos2)

We will discard the current cosmos2 PBUF implementation and reintroduce a clean, per-model module layout inspired by the legacy `cosmos_old` PBUF, keeping models isolated.

## Phase 0: Cleanup & layout
- [x] Remove existing cosmos2 PBUF code paths (models/kernels/fits) to avoid conflicts.
- [x] Add per-model subdirectories under `cosmos2/models` (e.g., `cosmos2/models/pbuf/`, `cosmos2/models/lcdm/`) and keep no shared pieces, models need to be seperated 100%.

## Phase 1: Port legacy PBUF math (one-to-one)
- [x] Copy legacy PBUF background/growth/elastic/equation helpers into `cosmos2/models/pbuf/` without altering math.
- [x] Copy PBUF CMB/distance helpers (r_s, D_M) exactly as in `cosmos_old`.
- [x] Copy PBUF parameter handling/closure logic intact (Omega_k from closure, Omega_sigma as in legacy).
- [x] Copy PBUF phase6a/phase7a sanity checks exactly and wire them to the model.

## Phase 2: Model surface & factory
- [x] Implement a new `PBUFModel` in `cosmos2/models/pbuf/model.py` that wraps the ported helpers and matches the old `CosmologyModel` behavior (Hubble/DM/DA/DV/mu/fs8/sigma8/CMB).
- [x] Update `cosmos2/models/model_factory.py` to load the new per-model PBUF package; ensure LCDM remains untouched and isolated.

## Phase 3: Fits & datasets (PBUF only)
- [x] Port legacy PBUF fit evaluators (CMB/BAO/SN/CC/RSD/etc.) into `cosmos2/models/pbuf/fits.py` (or split files) with the same equations/cov usage as `cosmos_old`.
- [x] Update fit registry/runner to point PBUF fits to the new implementation; keep LCDM separate.

## Phase 4: Tests & sanity scripts
- [x] Add regression tests comparing ported PBUF outputs to the legacy reference (`cosmos_old`) for a fixed parameter set (E(a), H(a), D_M, r_s, CMB priors, chi2).
- [x] Restore/update sanity scripts (background and CMB) to target the new PBUF package and verify closure + finite chi2.

## Phase 5: Documentation & cleanup
- [x] Document the new per-model layout and removal path for PBUF (so deleting it won’t break LCDM).
- [x] Remove dead scaffolding left from the previous broken PBUF; ensure imports don’t leak across models.

## Future step: Numba parity
- [x] After the clean PBUF port is stable, convert math helpers to Numba one-by-one, validating each against the legacy results to ensure 100% matching outputs.

## Possible future refactors (performance/mt)
- [x] Make integrators fully nopython: refactor Simpson/bisection to accept numba-callable integrands (no Python closures/thermal lookups) and drop forceobj; otherwise keep as pure Python to avoid numba warnings.
- [x] GrowthTable nopython: expose a numba RHS that inlines omega_sigma/thermal table lookups (numba-friendly table struct) so table build runs in nopython mode.
- [x] ThermalTable fast path: redesign around static arrays/field indices to allow numba getters; only if profiling shows table.get dominates.
- [x] Parallel fits: evaluate independent fits (cmb/bao/sn/cc/rsd/...) in parallel where thread-safe/read-only to improve throughput without changing math.
