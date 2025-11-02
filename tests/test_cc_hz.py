"""
Test: Cosmic Chronometer (CC) H(z) comparison for LCDM and PBUF.

Goal:
Check that the theoretical H(z) predictions behave correctly
and yield sensible χ² values against mock CC data.

We test:
  - monotonic increasing H(z)
  - smooth differences between LCDM and PBUF
  - proper χ² calculation

We use synthetic mock H(z) data (perfect LCDM baseline).
"""

import numpy as np
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF

# -------------------------------------------------------------
# 1. Setup models (same baseline as before)
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# 2. Synthetic CC data (LCDM baseline)
# -------------------------------------------------------------
z_data = np.array([0.07, 0.12, 0.20, 0.35, 0.60, 1.00, 1.50, 2.00])
H_data = np.array([lcdm.H(z) for z in z_data])  # baseline truth
sigma_H = 0.05 * H_data  # assume 5% errors

# -------------------------------------------------------------
# 3. Model predictions
# -------------------------------------------------------------
H_LCDM = np.array([lcdm.H(z) for z in z_data])
H_PBUF = np.array([pbuf.H(z) for z in z_data])

# -------------------------------------------------------------
# 4. Compute χ² for both
# -------------------------------------------------------------
def chi2(H_model, H_data, sigma):
    return float(np.sum(((H_model - H_data) / sigma)**2))

chi2_lcdm = chi2(H_LCDM, H_data, sigma_H)
chi2_pbuf = chi2(H_PBUF, H_data, sigma_H)

# -------------------------------------------------------------
# 5. Print report
# -------------------------------------------------------------
print("z     H_LCDM     H_PBUF     ΔH (PBUF-LCDM)")
for i, z in enumerate(z_data):
    print(f"{z:4.2f}  {H_LCDM[i]:9.4f}  {H_PBUF[i]:9.4f}   {H_PBUF[i]-H_LCDM[i]:9.4f}")

print("\nChi-squared summary:")
print(f"  LCDM χ² = {chi2_lcdm:.4f}")
print(f"  PBUF χ² = {chi2_pbuf:.4f}")

# -------------------------------------------------------------
# 6. Sanity checks
# -------------------------------------------------------------
assert np.all(np.diff(H_LCDM) > 0), "H(z) not increasing for LCDM"
assert np.all(np.diff(H_PBUF) > 0), "H(z) not increasing for PBUF"
assert np.all(np.isfinite(H_LCDM)) and np.all(np.isfinite(H_PBUF)), "NaN in H(z)"
assert chi2_lcdm < 1e-6, "LCDM χ² must be ≈0 for perfect self-fit"
assert chi2_pbuf > chi2_lcdm, "PBUF χ² must be larger for non-optimized model"

print("\n✅ Cosmic Chronometer H(z) test passed.")
