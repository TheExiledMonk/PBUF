PBUF physics entry points (cosmos2 per-model layout)
====================================================

cosmos2/models/pbuf/model.py
- `PBUFModel`: builds thermal table (via `cosmos2.pbuf.microphysics.ensure_thermal_table`), normalizes parameters, caches `a_grid`, and exposes the public surface (H/DM/DA/DV/mu/fs8/sigma8/CMB). Delegates math to per-module helpers and runs Phase-6a/7a sanity gates.

cosmos2/models/pbuf/distances.py
- Background helpers: `omega_total_at_a`, `E`, `H`, `H_z`. Distance integrals: `comoving_distance`, `angular_diameter_distance` (flat implementation, c/H0 applied in the integrand).

cosmos2/models/pbuf/elastic.py
- Elastic sector from the thermal table: `epsilon_of_a`, `alpha_of_a`, `kmax_of_a`, `omega_sigma_raw_of_a`, `omega_sigma_of_a` (includes normalization mode handling).

cosmos2/models/pbuf/normalization.py
- Parameter normalization/closure: resolves α from metadata, enforces `omega_normalization` (`flat_today` rescales Ω_σ(a=1) to close 1 − Ω_m0 − Ω_r0 − α).

cosmos2/models/pbuf/growth.py
- Growth RHS `growth_ode_rhs(a, y, params, table)` (D'' + (3/a + E'/E)D' − 1.5 Ω_m0/(a⁵E²)D = 0).

cosmos2/models/pbuf/cmb.py
- Sound horizon helpers (`sound_horizon`, `sound_horizon_drag`), baryon loading (`R_b`, `c_s`), and Hu–Sugiyama `z_star`. `compute_cmb_output` returns (R, l_A, θ*, z_*, r_s, r_d).

cosmos2/models/pbuf/fits.py
- Dataset evaluators (CMB/BAO/SN/CC/RSD/WL/galaxy_pk/SH0ES) plus `build_pbuf_joint_chi2`/`resolve_pbuf_joint_fits` and per-fit registry `PBUF_FIT_REGISTRY`. Growth rate helper `_growth_rate` reused by `PBUFModel.fs8`.

cosmos2/models/pbuf/phase6a.py and phase7a.py
- Sanity suites wired into `PBUFModel.is_valid` (`make_phase6a_checker`, `check_pbuf_phase7a_sanity`).

cosmos2/pbuf/microphysics.py
- Thermal table bootstrap/export (`ensure_thermal_table`, `export_thermal_table`) feeding `ThermalTable` used by the PBUF model.
