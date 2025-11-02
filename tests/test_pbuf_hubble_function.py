"""
Test: PBUF Hubble function correctness

Goal:
Verify that cosmos.pbuf.model.PBUF.H(z) and hubble_function(a)
match the restored curvature-based elastic sector:

    E²(z) = E²_LCDM(z) + σ_eff(z) S(z; k_sat) + Δ_rad(z; k_sat)
"""

import numpy as np
from cosmos.pbuf.model import PBUF
from cosmos.pbuf.equations import E2_pbuf

# ---------------------------------------------------------
# Configuration for the test
# ---------------------------------------------------------
H0 = 67.4          # km/s/Mpc
h  = H0 / 100.0
Om0 = 0.315
Ok0 = 0.0
Or0 = 5e-5
alpha = 5e-4
Rmax = 1e9
k_sat = 1.2
eps0 = 0.7
z_test = 1.0
a_test = 1.0 / (1.0 + z_test)

# ---------------------------------------------------------
# Manual computation
# ---------------------------------------------------------
params = {
    "H0": H0,
    "Om0": Om0,
    "Or0": Or0,
    "Ok0": Ok0,
    "alpha": alpha,
    "Rmax": Rmax,
    "k_sat": k_sat,
    "eps0": eps0,
    "n_alpha": 0.0,
    "n_eps": 0.0,
    "n_R": 0.0,
}

expected_ratio = np.sqrt(E2_pbuf(a_test, params))

# ---------------------------------------------------------
# Model calculation
# ---------------------------------------------------------
model = PBUF(
    omega_m=Om0,
    h=h,
    alpha=alpha,
    Rmax=Rmax,
    k_sat=k_sat,
    eps0=eps0,
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

print("✅ PBUF Hubble function test passed.")
