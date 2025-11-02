"""
Test: Supernova χ² evaluation for LCDM and PBUF.

Goal:
Verify that chi_squared_sn() computes stable, positive-definite chi²
values from SN data (μ, z) and covariance, and that results make sense
for both LCDM and PBUF.

We build a small synthetic dataset using the LCDM μ(z) as "data" and
check that:
  - LCDM χ² ≈ 0 (self-consistency)
  - PBUF χ² > 0 (expected for unoptimized parameters)
"""

import numpy as np
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.helper.distances import transverse_comoving_distance
from cosmos.helper.constants import MPC_TO_PC

# ---------------------------------------------------------
# 1. Helper functions
# ---------------------------------------------------------
def luminosity_distance(z, model):
    """Luminosity distance D_L(z) = (1 + z) * D_M(z) [Mpc]."""
    return (1 + z) * transverse_comoving_distance(z, model)

def distance_modulus(z, model):
    """Distance modulus μ = 5 log10(D_L / 10 pc)."""
    DL = luminosity_distance(z, model) * MPC_TO_PC  # convert Mpc → pc
    return 5.0 * np.log10(DL / 10.0)

def chi_squared_sn(z_data, mu_data, cov, model):
    """Compute χ² = Δμᵀ C⁻¹ Δμ."""
    mu_model = np.array([distance_modulus(z, model) for z in z_data])
    diff = mu_model - mu_data
    cov_inv = np.linalg.inv(cov)
    chi2 = float(diff.T @ cov_inv @ diff)
    return chi2

# ---------------------------------------------------------
# 2. Cosmological models
# ---------------------------------------------------------
H0 = 67.4
h = H0 / 100.0
Om0 = 0.315
Ok0 = 0.0
Or0 = 5e-5
Ol0 = 0.685
Obh2 = 0.022

alpha = 5e-4
Rmax  = 1e9
k_sat = 0.8

lcdm = LCDM(
    omega_m=Om0,
    omega_lambda=Ol0,
    h=h,
    omega_k=Ok0,
    omega_r=Or0,
    omega_b=Obh2 / (h**2),
)

pbuf = PBUF(
    omega_m=Om0,
    h=h,
    alpha=alpha,
    Rmax=Rmax,
    k_sat=k_sat,
    omega_k=Ok0,
    omega_r=Or0,
    omega_b=Obh2 / (h**2),
)

# ---------------------------------------------------------
# 3. Synthetic "Pantheon-like" dataset
# ---------------------------------------------------------
z_data = np.linspace(0.01, 1.0, 20)
mu_data = np.array([distance_modulus(z, lcdm) for z in z_data])

# Create simple diagonal covariance: σ_μ = 0.1 mag
sigma_mu = 0.1
cov = np.diag(np.full_like(z_data, sigma_mu**2))

# ---------------------------------------------------------
# 4. χ² evaluation
# ---------------------------------------------------------
chi2_lcdm = chi_squared_sn(z_data, mu_data, cov, lcdm)
chi2_pbuf = chi_squared_sn(z_data, mu_data, cov, pbuf)

print("LCDM χ² =", chi2_lcdm)
print("PBUF χ² =", chi2_pbuf)

# ---------------------------------------------------------
# 5. Assertions
# ---------------------------------------------------------
assert np.isfinite(chi2_lcdm) and chi2_lcdm < 1e-6, \
    f"LCDM χ² should be ~0 but got {chi2_lcdm}"
assert np.isfinite(chi2_pbuf) and chi2_pbuf > 0.0, \
    f"PBUF χ² should be positive but got {chi2_pbuf}"

if chi2_pbuf > chi2_lcdm:
    print("\n✅ PBUF χ² > LCDM χ² — expected for non-optimized parameters.")
else:
    print("\n⚠️ PBUF χ² <= LCDM χ² — recheck synthetic data scaling.")

print("✅ Supernova χ² test completed successfully.")
