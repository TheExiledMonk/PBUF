"""
Test: Closure relation (Ω_total ≈ 1) for LCDM and PBUF.

Goal:
Verify that both cosmological models report consistent closure today (a = 1).

LCDM:
    Ω_total = Ω_m + Ω_r + Ω_k + Ω_Λ

PBUF:
    Ω_total = Ω_m + Ω_r + Ω_k + Ω_σ(a=1)
"""

import numpy as np
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF

# ---------------------------------------------------------
# LCDM closure test
# ---------------------------------------------------------
H0 = 67.4
h = H0 / 100.0
Om0 = 0.315
Ol0 = 0.685
Ok0 = 0.0
Or0 = 5e-5

lcdm = LCDM(
    omega_m=Om0,
    omega_lambda=Ol0,
    h=h,
    omega_k=Ok0,
    omega_r=Or0
)

omega_total_lcdm = Om0 + Or0 + Ok0 + Ol0
print(f"LCDM Ω_total(a=1): {omega_total_lcdm:.12f}")
assert np.isclose(omega_total_lcdm, 1.0, rtol=1e-4), \
    f"LCDM closure failed: Ω_total={omega_total_lcdm}"

# ---------------------------------------------------------
# PBUF closure test
# ---------------------------------------------------------
alpha = 5e-4
Rmax = 1e8
k_sat = 1.2
eps0 = 0.7
Ok0 = 0.0
Or0 = 5e-5
Om0 = 0.315

pbuf = PBUF(
    omega_m=Om0,
    h=h,
    alpha=alpha,
    Rmax=Rmax,
    k_sat=k_sat,
    eps0=eps0,
    omega_k=Ok0,
    omega_r=Or0
)

omega_sigma_today = pbuf.omega_sigma(1.0)
omega_total_manual = pbuf.omega_m + pbuf.omega_r + pbuf.omega_k + omega_sigma_today

omega_total_model = pbuf.closure_today()

print(f"PBUF Ω_total(a=1) manual (no Λ): {omega_total_manual:.12f}")
print(f"PBUF Ω_total(a=1) via model (no Λ): {omega_total_model:.12f}")

assert np.isclose(omega_total_model, omega_total_manual, rtol=1e-6), \
    "PBUF closure calculation mismatch"

print("✅ Closure relation test arithmetic consistent for both LCDM and PBUF.")
