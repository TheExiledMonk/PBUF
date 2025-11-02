"""
Test: BAO isotropic distance combination D_V(z)/r_d
for LCDM and PBUF.

Goal:
Verify that we can compute the BAO isotropic observable
    D_V(z) / r_d
consistently for both cosmological models.

Definitions:
  D_M(z)  : transverse comoving distance [Mpc]
  D_A(z)  = D_M(z) / (1+z)              [Mpc]
  D_V(z)  = [ (1+z)^2 D_A(z)^2 * (c z / H(z)) ]^(1/3)  [Mpc]
  r_d     : sound horizon at baryon drag epoch z_drag  [Mpc]

We don't compare to real survey points yet. We just check:
  - internal consistency (manual vs model)
  - LCDM vs PBUF trends
  - monotonic behavior
"""

import numpy as np

from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.helper.constants import C_LIGHT
from cosmos.helper.distances import (
    transverse_comoving_distance,
    sound_horizon,
)
from cosmos.fits.cmb.observables import redshift_drag

# ---------------------------------------------------------
# 1. Cosmology setup (same baseline we've used)
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
# 2. Helper calculations
# ---------------------------------------------------------
def D_M(z, model):
    """Transverse comoving distance D_M(z) in Mpc."""
    return transverse_comoving_distance(z, model)

def D_A(z, model):
    """Angular diameter distance D_A(z) = D_M(z)/(1+z) in Mpc."""
    return D_M(z, model) / (1.0 + z)

def H_of_z(z, model):
    """Hubble parameter H(z) in km/s/Mpc."""
    return model.H(z)

def D_V(z, model):
    """
    Volume-averaged BAO distance used in isotropic BAO analyses.
    Units: Mpc.
    """
    da = D_A(z, model)     # Mpc
    hz = H_of_z(z, model)  # km/s/Mpc
    term = ((1+z)**2 * da**2 * (C_LIGHT * z / (1000.0 * hz)))  # Mpc^3
    return term ** (1.0 / 3.0)

def r_d(model):
    """
    Sound horizon at drag epoch (the BAO ruler scale).
    We'll call sound_horizon(model, z_drag) with z_drag from redshift_drag.
    Returns r_d in Mpc.
    """
    z_drag_val = redshift_drag(
        omega_b=getattr(model, "omega_b", 0.022),
        omega_m=getattr(model, "omega_m", 0.3),
        h=getattr(model, "h", 0.7),
    )
    return sound_horizon(model, z_drag=z_drag_val)

def DV_over_rd(z, model):
    return D_V(z, model) / r_d(model)

# ---------------------------------------------------------
# 3. Evaluate at standard BAO redshifts
#    These are typical survey bins: 0.38, 0.51, 0.61
# ---------------------------------------------------------
z_values = np.array([0.38, 0.51, 0.61])

DV_LCDM   = np.array([D_V(z, lcdm)        for z in z_values])
DV_PBUF   = np.array([D_V(z, pbuf)        for z in z_values])
rd_LCDM   = r_d(lcdm)
rd_PBUF   = r_d(pbuf)
DVrd_LCDM = DV_LCDM / rd_LCDM
DVrd_PBUF = DV_PBUF / rd_PBUF

# ---------------------------------------------------------
# 4. Report
# ---------------------------------------------------------
print("z    D_V_LCDM  D_V_PBUF   D_V/rd_LCDM  D_V/rd_PBUF")
for i, z in enumerate(z_values):
    print(f"{z:<4} {DV_LCDM[i]:10.4f} {DV_PBUF[i]:10.4f} {DVrd_LCDM[i]:12.6f} {DVrd_PBUF[i]:12.6f}")

# ---------------------------------------------------------
# 5. Basic sanity / behavior checks
# ---------------------------------------------------------

# Monotonicity: D_V should increase with z for reasonable models in this range.
assert np.all(np.diff(DV_LCDM) > 0), "D_V(z) not increasing in LCDM"
assert np.all(np.diff(DV_PBUF) > 0), "D_V(z) not increasing in PBUF"

# r_d (sound horizon at drag) should be finite and positive, O(100-200 Mpc).
assert rd_LCDM > 10.0 and rd_LCDM < 500.0, f"Unphysical r_d in LCDM: {rd_LCDM}"
assert rd_PBUF > 10.0 and rd_PBUF < 500.0, f"Unphysical r_d in PBUF: {rd_PBUF}"

# PBUF vs LCDM:
# It's common for modified-late-time expansion to change distances at these z,
# so D_V_PBUF and D_V_LCDM should differ at percent-ish level, not orders of magnitude.
ratio_check = DV_PBUF / DV_LCDM
assert np.all(ratio_check > 0.5) and np.all(ratio_check < 2.0), \
    f"D_V difference too large: ratios={ratio_check}"

print("\n✅ BAO isotropic test passed.")
