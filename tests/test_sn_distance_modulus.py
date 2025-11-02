"""
Test: Supernova distance–modulus relations for LCDM and PBUF.

Goal:
Verify that luminosity distances and distance moduli computed from each model
are self-consistent and physically reasonable.

We check:
  μ(z) = 5 * log10(D_L / 10 pc)
  D_L(z) = (1 + z) * D_M(z)

and compare model predictions at representative redshifts.
"""

import numpy as np
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.helper.distances import transverse_comoving_distance
from cosmos.helper.constants import MPC_TO_PC

# ---------------------------------------------------------
# 1. Common parameters
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

# ---------------------------------------------------------
# 2. Build both models
# ---------------------------------------------------------
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
# 3. Helper functions for test
# ---------------------------------------------------------
def luminosity_distance(z, model):
    """D_L(z) = (1+z) * D_M(z) in Mpc."""
    return (1 + z) * transverse_comoving_distance(z, model)

def distance_modulus(z, model):
    """μ(z) = 5 log10(D_L / 10 pc)."""
    DL = luminosity_distance(z, model) * MPC_TO_PC  # convert Mpc → pc
    return 5.0 * np.log10(DL / 10.0)

# ---------------------------------------------------------
# 4. Compute for representative redshifts
# ---------------------------------------------------------
z_values = np.array([0.01, 0.1, 0.5, 1.0, 2.0])

mu_lcdm = [distance_modulus(z, lcdm) for z in z_values]
mu_pbuf = [distance_modulus(z, pbuf) for z in z_values]

# ---------------------------------------------------------
# 5. Display comparison
# ---------------------------------------------------------
print("z     μ_LCDM     μ_PBUF     Δμ (PBUF-LCDM)")
for z, m1, m2 in zip(z_values, mu_lcdm, mu_pbuf):
    print(f"{z:<4} {m1:10.4f} {m2:10.4f} {m2 - m1:10.4f}")

# ---------------------------------------------------------
# 6. Checks
# ---------------------------------------------------------
# Low redshift limit: both must follow μ ≈ 5 log10(cz/H0) + 25
z_small = 0.01
mu_low_lcdm = distance_modulus(z_small, lcdm)
mu_expected = 5 * np.log10((3e5 * z_small) / H0) + 25  # c in km/s
assert abs(mu_low_lcdm - mu_expected) < 0.05, \
    f"Low-z limit mismatch: μ_LCDM={mu_low_lcdm}, expected={mu_expected}"

# Ensure both models yield finite and monotonic μ(z)
assert np.all(np.diff(mu_lcdm) > 0), "LCDM μ(z) not monotonic"
assert np.all(np.diff(mu_pbuf) > 0), "PBUF μ(z) not monotonic"

print("\n✅ Supernova distance–modulus test passed.")
