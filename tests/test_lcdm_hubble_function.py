"""
Test: LCDM Hubble function correctness

Goal:
Verify that cosmos.lcdm.model.LCDM.H(z) and hubble_function(a)
numerically reproduce the analytical Friedmann relation.

Expected equation:
    H(a)^2 / H0^2 = Ω_m a^-3 + Ω_r a^-4 + Ω_k a^-2 + Ω_Λ
"""

import numpy as np
from cosmos.lcdm.model import LCDM

# ---------------------------------------------------------
# Configuration for the test (Planck-like parameters)
# ---------------------------------------------------------
H0 = 67.4             # km/s/Mpc
h  = H0 / 100.0
Om0 = 0.315
Ol0 = 0.685
Ok0 = 0.0
Or0 = 5e-5
z_test = 1.0          # redshift to test
a_test = 1.0 / (1.0 + z_test)

# ---------------------------------------------------------
# Manual Friedmann calculation
# ---------------------------------------------------------
def friedmann_manual(a):
    """Manual computation of H(a)/H0 from basic equation."""
    return np.sqrt(Om0 / a**3 + Or0 / a**4 + Ok0 / a**2 + Ol0)

expected_ratio = friedmann_manual(a_test)

# ---------------------------------------------------------
# Model calculation
# ---------------------------------------------------------
model = LCDM(
    omega_m=Om0,
    omega_lambda=Ol0,
    h=h,
    omega_k=Ok0,
    omega_r=Or0
)

# Compute from both interfaces
H_model_z = model.H(z_test)
H_model_a = model.hubble_function(a_test)

ratio_model_z = H_model_z / model.h0
ratio_model_a = H_model_a / model.h0

# ---------------------------------------------------------
# Comparison
# ---------------------------------------------------------
tol = 1e-6

print(f"Expected H/H0: {expected_ratio:.8f}")
print(f"Model H(z)/H0: {ratio_model_z:.8f}")
print(f"Model H(a)/H0: {ratio_model_a:.8f}")

assert np.isclose(ratio_model_z, expected_ratio, rtol=tol), \
    f"H(z) mismatch: expected {expected_ratio}, got {ratio_model_z}"

assert np.isclose(ratio_model_a, expected_ratio, rtol=tol), \
    f"H(a) mismatch: expected {expected_ratio}, got {ratio_model_a}"

print("✅ LCDM Hubble function test passed.")
