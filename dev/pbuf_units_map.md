# PBUF distance and unit conventions (cosmos2)

- **Hubble parameter**: `H0` is supplied and stored in km/s/Mpc. Background grids set `H(a) = H0 * E(a)` in `cosmos2/models/pbuf/distances.py` (LCDM still uses `cosmos2/kernels/lcdm_math.py`).
- **Speed of light**: `c = 299_792.458 km/s` (`cosmos2/models/pbuf/utils.py`). All PBUF distances below are in Mpc.
- **Comoving distance**: `D_C(z) = ∫_0^z c / H(z') dz'` with the `(c/H0)` factor inside the integrand in `cosmos2/models/pbuf/distances.comoving_distance` (flat implementation).
- **Derived distances**: `D_A = D_C / (1 + z)` in `cosmos2/models/pbuf/distances.angular_diameter_distance`; `D_L = (1 + z) * D_M = (1 + z)^2 * D_A`; `D_V`/`D_H` in `PBUFModel` reuse the same `c/H0` convention.
- **Sound horizons**: `r_s` / `r_d` come from integrating `c_s / (a^2 H(a))` (units of Mpc) in `cosmos2/models/pbuf/cmb.py`, with `c_s` built from the same `c` and baryon/photon densities.
- **CMB distance priors**: `cosmos2/models/pbuf/fits.run_cmb_fit` (and `PBUFModel.cmb`) use `D_M` from the distance grid, `R = sqrt(Omega_m0) * (H0 * D_M / c)`, `l_A = π D_M / r_d`, and `theta_star = r_d / D_M`, all in Mpc.
- **PBUF closure term**: `cosmos2/models/pbuf/normalization.py` enforces `omega_normalization`; the `flat_today` mode rescales Ω_σ(a=1) so `Ω_m0 + Ω_r0 + α + Ω_σ(a=1) = 1`, keeping the LCDM limit aligned when elasticity is off.

In short:

```
D_C(z) = (c / H0) * ∫_0^z dz' / E(z')  (flat, so D_M = D_C)
D_A = D_M / (1 + z); all returned in Mpc.
```
