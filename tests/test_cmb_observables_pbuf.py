"""
Test: CMB distance-prior observables for PBUF.

We verify that the module cmb_observables(model) returns R, l_A, theta_star
that match their defining equations, using the PBUF background.

Definitions:

    R = sqrt(Ω_m) * (H0 / c) * D_M(z*)
    l_A = π * D_M(z*) / r_s(z*)
    theta_* = r_s(z*) / D_M(z*)

where
    D_M(z)  : transverse comoving distance [Mpc]
    r_s(z*) : sound horizon at decoupling [Mpc]
    H0      : km/s/Mpc
    c       : km/s
"""

import numpy as np

from cosmos.pbuf.model import PBUF
from cosmos.fits.cmb.observables import cmb_observables, redshift_star
from cosmos.helper.constants import C_LIGHT
from cosmos.helper.distances import (
    transverse_comoving_distance,
    sound_horizon,
)

# --------------------------
# 1. Build a Planck-like PBUF model (normalized for closure)
# --------------------------
H0 = 67.5
h  = H0 / 100.0
Om0 = 0.315
alpha = 0.001
Rmax = 1e9
k_sat = 0.8
Ok0 = 0.0
Or0 = 5e-5      # radiation today
Obh2 = 0.022    # for sound speed / r_s

model = PBUF(
    omega_m=Om0,
    h=h,
    alpha=alpha,
    Rmax=Rmax,
    k_sat=k_sat,
    omega_k=Ok0,
    omega_r=Or0,
    omega_b=Obh2 / (h**2),
)

# --------------------------
# 2. Get observables from code under test
# --------------------------
obs = cmb_observables(model)
R_model          = obs["R"]
lA_model         = obs["la"]
theta_model      = obs["theta_star"]
z_star_model     = obs["z_star"]

print("Returned by cmb_observables(model):")
print(f"  z_star      = {z_star_model}")
print(f"  R           = {R_model}")
print(f"  l_A         = {lA_model}")
print(f"  theta_star  = {theta_model}")

# --------------------------
# 3. Manual recomputation of the same quantities
# --------------------------
# 3a. get decoupling redshift (should match your code's convention)
z_star = redshift_star(model)

# 3b. transverse comoving distance to z_star
DM = transverse_comoving_distance(z_star, model)  # Mpc

# 3c. comoving sound horizon at decoupling
rs_star = sound_horizon(model, z_drag=z_star)     # Mpc

# 3d. manual formulas
R_manual = np.sqrt(Om0) * (model.h0 / (C_LIGHT / 1000.0)) * DM

lA_manual = np.pi * DM / rs_star

theta_manual = rs_star / DM

print("\nManual recomputation:")
print(f"  R_manual          = {R_manual}")
print(f"  lA_manual         = {lA_manual}")
print(f"  theta_manual      = {theta_manual}")

# --------------------------
# 4. Compare
# --------------------------
rtol = 1e-6

assert np.isclose(R_model, R_manual, rtol=rtol), \
    f"R mismatch: model={R_model}, manual={R_manual}"

assert np.isclose(lA_model, lA_manual, rtol=rtol), \
    f"l_A mismatch: model={lA_model}, manual={lA_manual}"

assert np.isclose(theta_model, theta_manual, rtol=rtol), \
    f"theta_* mismatch: model={theta_model}, manual={theta_manual}"

print("\n✅ PBUF CMB observables test passed.")
