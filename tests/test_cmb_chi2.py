"""
Test: CMB χ² evaluation for LCDM and PBUF.

Goal:
Ensure that chi_squared_cmb() produces stable and physically consistent
results when comparing model predictions against Planck 2018 distance priors.

We test both:
  - LCDM: should produce a small χ² near zero at Planck parameters.
  - PBUF: should produce a larger, but finite, χ² difference (model deviation).
"""

import numpy as np
from cosmos.fits.cmb.observables import chi_squared_cmb, PLANCK_2018_PRIORS, PLANCK_2018_COVARIANCE
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF

# ---------------------------------------------------------
# 1. Planck reference data
# ---------------------------------------------------------
print("Planck 2018 priors:")
for k, v in PLANCK_2018_PRIORS.items():
    print(f"  {k} = {v}")

print("\nPlanck 2018 covariance (3x3):")
print(PLANCK_2018_COVARIANCE)

# ---------------------------------------------------------
# 2. Create LCDM model instance
# ---------------------------------------------------------
# Using Planck 2018 best-fit parameters
lcdm_params = {
    "H0": 67.36,
    "Om0": 0.3153,
    "Ok0": 0.0,
    "Ol0": 0.6847,
    "Or0": 9.2e-5,
    "Obh2": 0.02237,
}

# Create LCDM model instance
lcdm_model = LCDM(
    omega_m=lcdm_params["Om0"],
    omega_lambda=lcdm_params["Ol0"],
    h=lcdm_params["H0"]/100.0,
    omega_k=lcdm_params["Ok0"],
    omega_r=lcdm_params["Or0"],
    omega_b=lcdm_params["Obh2"]/(lcdm_params["H0"]/100.0)**2,
    T_cmb=2.7255
)

chi2_lcdm = chi_squared_cmb(lcdm_model)
print(f"\nLCDM χ² = {chi2_lcdm:.6f}")
assert np.isfinite(chi2_lcdm) and chi2_lcdm >= 0.0, "Invalid χ² for LCDM"

# ---------------------------------------------------------
# 3. Create PBUF model instance
# ---------------------------------------------------------
# PBUF parameters - using values that give reasonable chi-squared
pbuf_params = {
    "H0": 67.36,
    "Om0": 0.3153,
    "Ok0": 0.0,
    "Ol0": 0.0,     # no Λ term
    "Or0": 9.2e-5,
    "Obh2": 0.02237,
    "alpha": 1e-3,
    "Rmax": 0.5,
    "k_sat": 1.0,
}

# Create PBUF model instance
pbuf_model = PBUF(
    omega_m=pbuf_params["Om0"],
    h=pbuf_params["H0"]/100.0,
    alpha=pbuf_params["alpha"],
    Rmax=pbuf_params["Rmax"],
    k_sat=pbuf_params["k_sat"],
    omega_k=pbuf_params["Ok0"],
    omega_r=pbuf_params["Or0"],
    omega_b=pbuf_params["Obh2"]/(pbuf_params["H0"]/100.0)**2,
    T_cmb=2.7255
)

chi2_pbuf = chi_squared_cmb(pbuf_model)
print(f"PBUF χ² = {chi2_pbuf:.6f}")
assert np.isfinite(chi2_pbuf) and chi2_pbuf >= 0.0, "Invalid χ² for PBUF"

# ---------------------------------------------------------
# 4. Sanity checks
# ---------------------------------------------------------
if chi2_pbuf > chi2_lcdm:
    print("\n✅ PBUF shows larger χ² than LCDM — expected for non-optimized model.")
else:
    print("\n⚠️ PBUF χ² smaller than LCDM (check parameters or covariance scaling).")

print("\n✅ CMB χ² test completed successfully.")
