# PBUF Equations

## Constants & units
- `c = 299792.458 km/s` (from `C_LIGHT` in `cosmos2/models/pbuf/utils.py` and the kernels).
- `H(a) = H0 * E(a)` with `E(a)` from the dimensionless expansion rate `E = sqrt(E^2)` (`cosmos2/kernels/pbuf_distances.py`).

## Background expansion
- `Omega_total(a) = Omega_m0 / a^3 + Omega_r0 / a^4 + alpha / a^2 + Omega_sigma(a)` (`cosmos2/kernels/pbuf_distances.py`).
- `E^2(a) = Omega_total(a)` and `H(a) = H0 * E(a)` (`cosmos2/kernels/pbuf_distances.py`).
- Comoving distance:
  ```
  D_C(z) = ∫_0^z (c / H(z')) dz'
  ```
  (`cosmos2/models/pbuf/distances.py` and `cosmos2/kernels/pbuf_distances.py`, integrand uses `C_LIGHT / H`).
- Angular diameter and luminosity distances:
  `D_A = D_C / (1 + z)`, `D_L = (1 + z)^2 * D_A`, `D_H = c / H`. Derived volume distance `D_V` is constructed from `D_M` (same as `D_C` in flat space) and `H(z)` in `cosmos2/models/pbuf/model.py`.

## Elastic sector
- `k_max(a) = epsilon0_T(a) - alpha_T(a)` (`cosmos2/kernels/pbuf_elastic.py`).
- `decay = exp(-a / Rmax)` and `S = 1 - (1 - k_max) * decay` feed every `omega_sigma` evaluation via the kernels.
- Raw elastic contribution:
  `Omega_sigma_raw(a) = alpha(a) * (1 - decay) * S`.
- Normalization modes:
  `Omega_sigma(a) = Omega_sigma_raw(a)` in `free` mode or `sigma_rescale * Omega_sigma_raw(a)` when `omega_normalization == "flat_today"` (`cosmos2/models/pbuf/elastic.py` and `cosmos2/models/pbuf/normalization.py`).

## Parameter normalization
- Baryon and matter closure: `Omega_b0 = 2 * alpha`, `Omega_m0 = Omega_b0 / BARYON_FRACTION` with `BARYON_FRACTION = 0.135` (`cosmos2/models/pbuf/normalization.py`).
- Closure target at the present epoch:
  `sigma_target = 1 - Omega_m0 - Omega_r0 - alpha`
  and `sigma_rescale = sigma_target / Omega_sigma_raw(a=1)` so that `Omega_total(a=1) = 1` in `flat_today` mode (`cosmos2/models/pbuf/normalization.py`).

## CMB & photon-sector scalars
- Photon density parameter today:
  `Og0 = Omega_r0 * 2 / g_star(a=1)` (`cosmos2/kernels/pbuf_cmb.py` and `cosmos2/models/pbuf/cmb.py` describe the usage of `PHOTON_G_DEGREES = 2`).
- Baryon-to-photon momentum ratio:
  `R_b(z) = 0.75 * (Omega_b0 / Og0) * (1 / a^3) * (T0 / Tz)^4` with `a = 1/(1+z)` (`R_b_kernel` in `cosmos2/kernels/pbuf_cmb.py`).
- Sound speed:
  `c_s(z) = c / sqrt(3 * (1 + R_b(z)))` (`cosmos2/kernels/pbuf_cmb.py`).
- Hu & Sugiyama decoupling redshift:
  ```
  g1 = (0.0783 * Obh2^-0.238) / (1 + 39.5 * Obh2^0.763)
  g2 = 0.560 / (1 + 21.1 * Obh2^1.81)
  z_* = 1048 * (1 + 0.00124 * Obh2^-0.738) * (1 + g1 * Omh2^g2)
  ```
  (`z_star_hu_sugiyama_kernel` in `cosmos2/kernels/pbuf_cmb.py`).
- Eisenstein & Hu drag epoch:
  ```
  b1 = 0.313 * Omh2^-0.419 * (1 + 0.607 * Obh2^0.674)
  b2 = 0.238 * Omh2^0.223
  z_drag = (1291 * Omh2^0.251 / (1 + 0.659 * Omh2^0.828)) * (1 + b1 * Obh2^b2)
  ```
  (`z_drag_eh_kernel`).
- Sound horizon integral:
  ```
  r_s(z_target) = ∫_0^{a(z_target)} c_s(a) / (a^2 H(a)) da
  ```
  with `a = 1 / (1 + z_target)` and the integrand from `sound_integrand_kernel` (`cosmos2/kernels/pbuf_cmb.py` and `cosmos2/models/pbuf/cmb.py`).
- Distance priors:
  `R = sqrt(Omega_m0) * (H0 * D_M / c)`, `l_A = pi * D_M / r_d`, `theta_star = r_d / D_M`, where `D_M` is the comoving distance at `z_*` and `r_d` is the drag-scale sound horizon (`dev/pbuf_units_map.md` and `cosmos2/models/pbuf/cmb.py`).

## Structure growth
- Linear growth equation:
  ```
  D''(a) + (3/a + (dE/da)/E) * D'(a) - 1.5 * Omega_m0 / (a^5 E^2) * D(a) = 0
  ```
  (`growth_ode_rhs` in `cosmos2/models/pbuf/growth.py` mirrors `cosmos2/kernels/pbuf_growth_rhs.py`).
- Growth-derived scalars:
  `sigma8 ≈ 1 - Omega_m0` and `S8(gamma) = sigma8 * (Omega_m0 / 0.3)^gamma` (`PBUFModel` constructing `_sigma8` and `S8` in `cosmos2/models/pbuf/model.py`).

## Void-size scaling
- Reference ΛCDM curve:
  `R_void,ref(z) = R_ref,0 * (1+z)^{-beta}` (`cosmos2/predictions/modules/void_size.py` exposes `--beta-z` and `--R_ref_Mpc`).
- Growth scaling:
  `S_growth(z) = (D_LCDM(z) / D_PBUF(z))^{eta}` when comparing to LCDM, otherwise `D_PBUF(z)^{-eta}` (`void_size` uses `eta = --eta-growth`).
- Elastic slack:
  `S_elastic = 1 + gamma_alpha * alpha` (`--gamma-alpha` and the `alpha` parameter recorded by the model).
- Final prediction:
  `R_void,PBUF(z) = R_void,ref(z) * S_growth(z) * S_elastic`.
