Weak-lensing upgrade plan (post-pipeline completion)
====================================================

Context
- KiDS-1000 WL pipeline is wired end-to-end and model-agnostic; interface must remain unchanged.
- Goal: promote WL from pipeline-validated to science-grade accuracy.
- Scope: accuracy and nuisance-layer upgrades only; no architecture or dataset conversion work.

In scope (adds)
- High-fidelity matter P(k, z): Eisenstein–Hu transfer, linear spectrum, Halofit; optional CLASS/CAMB hook.
- Accurate ξ± transforms: FFTLog/Hankel with Bessel fallback.
- WL nuisance parameters: intrinsic alignment (IA), photo-z shifts, shear m-bias, baryon placeholder.
- KiDS angular scale cuts with consistent masking of data vector and covariance.
- Reporting and safety flags to gate WL influence on rankings.

Out of scope (must remain untouched)
- Dataset conversion (already complete).
- Backend interface redesign; WL stays geometry + growth only.
- Elastic/background changes beyond WL surface.

Current state (baseline)
- `WeakLensingBackend.P_m_of_kz(k, z)` exists but uses a σ₈-normalized analytic power-law placeholder.
- ξ± computed via direct Bessel approximations in `cosmos2/wl/bessel.py`; suitable for validation only.
- Nuisance layers (IA, photo-z shifts, shear m, baryons) effectively absent/zeroed.
- No KiDS θ scale cuts; WL results can impact rankings without safety guardrails.

Phase A — Matter power spectrum
- Step 1: Add Eisenstein–Hu transfer `cosmos2/power/transfer_eh.py` (wiggle + no-wiggle; inputs Ω_m h², Ω_b h², h; outputs T(k [h/Mpc])). Shared by ΛCDM and PBUF.
- Step 2: Linear spectrum `cosmos2/power/pk_linear.py` implementing `P_lin(k, z) = A_s (k/k_*)^{n_s-1} T²(k) D²(z)` using backend growth D(z); no σ₈ renormalization.
- Step 3: Halofit `cosmos2/power/halofit.py` mapping P_lin → P_nl; exposed via `P_m_of_kz(k, z, nonlinear=True)`.
- Step 4 (optional): CLASS/CAMB hook `cosmos2/power/external/class_backend.py` with caching, optional import, and fallback to EH+Halofit when unavailable.

Phase B — ξ± transforms
- Implement FFTLog/Hankel `cosmos2/wl/fftlog.py`: log-spaced ℓ grid; C_ℓ → ξ⁺ via J₀, C_ℓ → ξ⁻ via J₄; stable across KiDS θ range.
- Wire `cosmos2/wl/theory.py` to use FFTLog by default with existing Bessel path as fallback for constrained environments.
- Validation: compare FFTLog vs Bessel; percent-level agreement at large θ, flagged discrepancies at small θ (expected).

Phase C — Nuisance layer
- IA: add `cosmos2/wl/ia.py` with NLA baseline; parameters A_IA (and optional η_IA redshift scaling); modify spectra to include GG + GI + II; toggle via config `wl.include_ia`.
- Photo-z shifts: support Δz_i per tomographic bin by shifting n_i(z) → n_i(z + Δz_i) with renormalization; expose as fit parameters with priors.
- Shear calibration (m-bias): load KiDS priors; apply `(1 + m_i)(1 + m_j)` factor to ξ; allow fixed or marginalized modes.
- Baryonic feedback (v1 placeholder): either simple suppression model or conservative small-scale cuts; keep interface ready.

Phase D — Likelihood hygiene
- KiDS scale cuts: add `cosmos2/wl/scale_cuts.py` to mask data vector and covariance per tomo-pair θ limits; config example:
  ```
  wl:
    apply_scale_cuts: true
    scale_cut_profile: kids_default
  ```
- Reporting + safety flags: WL reports must include P(k, z) provider (analytic/EH/Halofit/CLASS), ξ± method (Bessel/FFTLog), nuisance status (IA/photo-z/m-bias on/off), scale cuts (applied/not). Until all upgrades are active, WL χ² is diagnostic/beta and must not affect global ranking.

Definition of science-grade complete
- EH + Halofit (or CLASS backend) active.
- FFTLog ξ± active (Bessel only as fallback).
- IA + photo-z + m-bias implemented and configurable.
- KiDS scale cuts applied before χ².
- Same pipeline yields stable χ² for ΛCDM and PBUF with unchanged interface.
