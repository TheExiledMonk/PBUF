PBUF constant audit (cosmos2, per-model layout)
===============================================

Legend: ✅ allowed, ⚠️ needs context, ❌ forbidden.

- ✅ `C_LIGHT = 299792.458` (`cosmos2/models/pbuf/utils.py`, reused by model/fits) – physical c.
- ✅ Radiation default `Omega_r0 = 9e-5` (`cosmos2/models/pbuf/params.py`) – single-source radiation prior carried over from legacy.
- ✅ σ₈ seed `_DEFAULT_SIGMA8 = 0.811` (`cosmos2/models/pbuf/model.py`) – parity anchor against the legacy outputs; not a prior.
- ✅ Phase-7a thresholds (`cosmos2/models/pbuf/phase7a.py`, overridable via `configs/phase7a/pbuf.json`): `alpha_max_abs=0.1`, `alpha_step_max=0.01`, `epsilon0_max=2.0`, `epsilon0_step_max=0.02`, `k_sat_step_max=0.02`, `Rmax_min=1e5`, `Rmax_max=1e10`, `Rmax_step_factor=3.0`, `df_max=2.0`, `curv_ratio_max=5e6`, `curv_ratio_fraction=0.1`, `a_min=1e-9`, `a_max=1.0`, `n_a=500`, `early_lcdm_tol=1e-3`, `closure_tol=1e-5`, `alpha_deriv_max=5.0`, `epsilon_deriv_max=3.0`, `H_monotonic_rel_tol=0.0`.
- ✅ Growth ODE coefficients: friction `3.0`, mass `1.5` (`cosmos2/models/pbuf/growth.py`); numerical epsilons (e.g., `1e-5`, `1e-12`) only.
- ✅ Integrator/mesh knobs: Simpson steps `4096` for distance/sound horizons (`cosmos2/models/pbuf/cmb.py`), small epsilons in utils/distances to avoid division by zero.
- ⚠️ EH-style recombination/drag fits in `cosmos2/models/pbuf/cmb.py`: `1291.0`, `1048.0`, `0.313`, `0.238`, `0.251`, `0.607`, `0.659`, `0.674`, `0.0783`, `0.223`, `0.238`, `0.560`, `21.1`, `39.5`, exponents `-0.419`, `-0.238`, `-0.738`, `0.763`, `0.223`, `0.251`, `0.828`, `1.81` – phenomenological (EH98-style); document when citing.
- ⚠️ `z_star` fallback `1090.0` in `cosmos2/models/pbuf/fits.py` when dataset metadata omits it – Planck-like placeholder; prefer supplying z_* explicitly.
- ✅ Photon fraction `photon_fraction = 2 / g_star(a=1)` (`cosmos2/models/pbuf/cmb.py`) – derived from thermal table metadata, no hidden Λ term.
