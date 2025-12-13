# Weak Lensing (Cosmic Shear) — KiDS-1000 Dual-Model Plan

Goal: end-to-end ξ± pipeline that can swap cosmology backends (ΛCDM vs PBUF) while keeping the weak lensing likelihood identical. Shared layer handles data, ordering, transforms, χ²; backends only supply distances, growth, P(k, z).

## Data standardization (shared)
- Source: `data/standardized/weak_lensing_kids1000_raw_v1.npz` (xi_plus/xi_minus 5×5×9, θ[9], cov[270×270], nz[5×119], tomo edges, meta).
- Converter: `toolbox/converter.py::_convert_wl_kids1000` already parses FITS → raw NPZ; keep symmetry checks and n(z).
- Produce standardized file: `data/standardized/weak_lensing_kids1000_v1.npz` with keys:
  - `data_vector` length 270 ordered as all ξ⁺ (tomo pairs × θ) then all ξ⁻.
  - `tomo_pairs` = 15 pairs (i ≤ j from 5 bins); `theta_bins` in radians (preserve original units in `meta`).
  - `covariance` (270×270), `n_of_z` (5×Nz), `z_grid` (Nz).
  - `shear_m` (5,) if available; allow `shear_m_prior` in `meta`.
  - `meta.data_order = "xi_plus_then_xi_minus"` and `meta.kids_ordering_notes` describing flatten rule.
- Registry: `cosmos2/data/registry.py` maps `weak_lensing_kids1000` → standardized dataset; keep `weak_lensing_kids1000_raw` for debugging.

Flattening rule (KiDS): loop tomo_pairs in the standard (i ≤ j) order, fill θ bins for ξ⁺ first, then repeat the same ordering for ξ⁻.

## Shared WL math
- Backend interface (minimal):
  - `chi_of_z(z_array)`, `H_of_z(z_array)`, `P_m_of_kz(k_array, z_array, nonlinear=False)`.
  - Optional helpers: `growth_D_of_z(z_array)`, `Omega_m_of_z(z_array)` if IA needs it.
- Lensing efficiency:
  ```
  W_i(z) ∝ H(z) χ(z) ∫_z^∞ dz' n_i(z') χ(z') (χ(z') − χ(z)) / χ(z')
  ```
- Limber Cℓ:
  ```
  C_ℓ,ij = ∫ dz W_i(z) W_j(z) / χ^2(z) * P_m(k=(ℓ+0.5)/χ(z), z)
  ```
- ξ± transforms (start simple):
  ```
  ξ⁺(θ) = Σℓ (2ℓ+1)/(4π) C_ℓ J₀(ℓθ)
  ξ⁻(θ) = Σℓ (2ℓ+1)/(4π) C_ℓ J₄(ℓθ)
  ```
  Begin with direct Bessel sums; FFTLog/Hankel is the upgrade path.
- Model → data vector: compute ξ⁺, ξ⁻ per (pair, θ), apply shear calibration `(1+m_i)(1+m_j)`, flatten in KiDS ordering (ξ⁺ block then ξ⁻ block).
- χ²: `(m − d)^T C^{-1} (m − d)` using the standardized covariance.

## ΛCDM backend hooks
- Distances: reuse existing ΛCDM pipeline (BAO/CMB/SN code) for `E(z)`, `H(z)=H0E(z)`, comoving `χ(z)=∫ c/H`.
- Growth: linear D(z) consistent with ΛCDM parameters.
- P_m(k, z):
  - Preferred: CLASS/CAMB hook (nonlinear optional).
  - Analytic fallback: Eisenstein–Hu transfer + primordial `A_s, n_s`, scaled by D(z); optional Halofit mapping.
- Nuisance knobs exposed to WL fit: shear m per bin, IA amplitude (+ redshift scaling), photo-z shift per bin, baryon feedback or scale cuts.

## PBUF backend hooks
- Distances: `E^2(a)=Ω_m0/a^3 + Ω_r0/a^4 + α/a^2 + Ω_σ(a)`; `χ(z)=∫ c/H`. If Ω_k0 allowed, ensure `D_M` uses curvature mapping (no flat assumption).
- Growth: PBUF ODE `D'' + (3/a + (dE/da)/E) D' − 1.5 Ω_m0 /(a^5 E^2) D = 0`.
- P_m(k, z):
  - Minimum viable: `P_lin = A_s (k/k*)^(n_s−1) T^2(k) D^2(z)` using EH-style transfer reused from CMB scalar plumbing; same Halofit path as ΛCDM if enabled.
- Nuisance layer identical to ΛCDM (IA, m-bias, photo-z shifts, baryons) so the likelihood stays model-agnostic.

## Fit wiring (shared)
- Add `cosmos2/fits/weak_lensing_kids1000.py`:
  - Loads standardized dataset.
  - Calls backend interface to get ξ± vector and returns χ².
  - Registers in `cosmos2/fits/registry.py`.
- Runner keys: dataset `weak_lensing_kids1000`, fit `wl_kids1000`.
- Scale cuts / ℓ range: must follow KiDS prescriptions; define θ cuts per pair if required and map to ℓ limits consistently.
  - Default profile uses θ_min^+ = 0.5′ and θ_min^- = 4.2′ for all pairs.
  - Per-pair overrides can be supplied via `meta.wl.scale_cut_table` (list/tuple of 2 or 4 arcmin values per `(i,j)` key) or via the environment:
    - Set `COSMOS2_WL_KIDS1000_CONFIG=/path/to/wl_config.yaml` (or JSON) or export JSON directly in `COSMOS2_WL_CONFIG`.
    - Example YAML:
      ```yaml
      wl:
        apply_scale_cuts: true
        scale_cut_table:
          "0-3": [0.5, 300, 5.5, 300]   # (min_plus, max_plus, min_minus, max_minus) in arcmin
          [2,4]: [0.5, 300]             # shared for xi+/xi-
      ```
  - ℓ-range, FFTLog/Bessel toggle, IA/photo-z/shear-m can be set in the same `wl` block and will flow through the fit runner and reports.

## Validation and tests
- Unit tests (use synthetic WL NPZs in `data/standardized/wl*.npz`):
  - Verify ordering, 15 tomo pairs, covariance application, deterministic χ².
- Smoke tests:
  - Load KiDS standardized dataset.
  - Run prediction with mocked P(k, z) (e.g., power law) → finite ξ±, finite χ².

## Implementation order (recommended)
1. Standardize KiDS → `weak_lensing_kids1000_v1.npz` + registry alias.
2. Define backend interface + stubs for ΛCDM and PBUF.
3. Implement shared analytic linear P(k, z) module (EH transfer).
4. Implement Limber Cℓ and ξ± transform (direct Bessel first).
5. Wire KiDS WL fit module + registry support.
6. Add tests + smoke run in science runner.
7. Add nonlinear mapping or CLASS/CAMB option.
8. Expose nuisance parameters (IA, photo-z, m-bias, baryons).
