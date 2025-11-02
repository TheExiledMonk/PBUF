from cosmos.physics.sigma8 import sigma8_from_primordial, E_LCDM
from cosmos.pbuf.equations_adapter import E_PBUF

# --- LCDM baseline (Planck-like) ---
lcdm_params = dict(H0=67.36, Om0=0.315, Or0=9.2e-5, Ok0=0.0)
E_LCDM_fn = lambda a: E_LCDM(a, **lcdm_params)

sigma8_lcdm = sigma8_from_primordial(
    As=2.1e-9,
    ns=0.965,
    h=lcdm_params["H0"] / 100.0,
    Om0=lcdm_params["Om0"],
    Obh2=0.02237,
    E_of_a=E_LCDM_fn,
    calibrate_against=dict(
        As=2.1e-9,
        ns=0.965,
        h=lcdm_params["H0"] / 100.0,
        Om0=lcdm_params["Om0"],
        Obh2=0.02237,
        E_of_a=E_LCDM_fn,
        sigma8_ref=0.811,
    ),
    calibration_key="planck18_lcdm",
)
print(f"LCDM σ8 ≈ {sigma8_lcdm:.4f} (should be ~0.811)")

# --- PBUF elastic model ---
pbuf_params = dict(
    H0=70.0,
    Om0=0.33,
    Or0=9.2e-5,
    Ok0=0.0,
    alpha=0.03,
    Rmax=1e7,
    k_sat=0.86,
    eps0=0.71,
    n_alpha=0.4,
    n_eps=-0.5,
)
E_PBUF_fn = lambda a: E_PBUF(a, **pbuf_params)

sigma8_pbuf = sigma8_from_primordial(
    As=2.1e-9,
    ns=0.965,
    h=pbuf_params["H0"] / 100.0,
    Om0=pbuf_params["Om0"],
    Obh2=0.02237,
    E_of_a=E_PBUF_fn,
    calibrate_against=dict(
        As=2.1e-9,
        ns=0.965,
        h=lcdm_params["H0"] / 100.0,
        Om0=lcdm_params["Om0"],
        Obh2=0.02237,
        E_of_a=E_LCDM_fn,
        sigma8_ref=0.811,
    ),
    calibration_key="planck18_lcdm",
)
print(f"PBUF σ8 (free growth) ≈ {sigma8_pbuf:.4f}")
print(f"Elastic suppression ratio σ8_PBUF / σ8_LCDM = {sigma8_pbuf / sigma8_lcdm:.3f}")
