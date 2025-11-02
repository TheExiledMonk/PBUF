"""
Test: BAO anisotropic observables for LCDM and PBUF.

We compute two standard anisotropic BAO quantities:

    DM_over_rd(z) = D_M(z) / r_d
    DH_over_rd(z) = D_H(z) / r_d = c / [H(z) * r_d]

where:
    D_M(z) is the transverse comoving distance [Mpc]
    H(z)   is the Hubble rate [km/s/Mpc]
    r_d    is the sound horizon at the drag epoch [Mpc]
    c      is the speed of light [km/s]

We check:
 - monotonic behavior with redshift
 - finite, positive values for both models
 - PBUF vs LCDM trends are sensible
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

# ----------------------------------------
# 1. Cosmology setup
# ----------------------------------------
H0   = 67.4
h    = H0 / 100.0
Om0  = 0.315
Ok0  = 0.0
Or0  = 5e-5
Ol0  = 0.685
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

# ----------------------------------------
# 2. Helpers
# ----------------------------------------
def D_M(z, model):
    """Transverse comoving distance [Mpc]."""
    return transverse_comoving_distance(z, model)

def H_of_z(z, model):
    """Hubble rate [km/s/Mpc]."""
    return model.H(z)

def r_d(model):
    """Sound horizon at the drag epoch [Mpc]."""
    z_drag_val = redshift_drag(
        omega_b=getattr(model, "omega_b", 0.022),
        omega_m=getattr(model, "omega_m", 0.3),
        h=getattr(model, "h", 0.7),
    )
    return sound_horizon(model, z_drag=z_drag_val)

def bao_aniso_observables(z, model):
    """
    Returns:
        DM_over_rd   = D_M(z) / r_d
        DH_over_rd   = (C_LIGHT / 1000.0) / (H(z) * r_d)
    both dimensionless.
    """
    DM = D_M(z, model)
    Hz = H_of_z(z, model)
    rd = r_d(model)
    DM_over_rd = DM / rd
    DH_over_rd = (C_LIGHT / 1000.0) / (Hz * rd)
    return DM_over_rd, DH_over_rd

# ----------------------------------------
# 3. Evaluate at BAO-ish redshifts
# ----------------------------------------
z_values = np.array([0.38, 0.51, 0.61])

DMrd_LCDM = []
DHrd_LCDM = []
DMrd_PBUF = []
DHrd_PBUF = []

for z in z_values:
    dm_l, hz_l = bao_aniso_observables(z, lcdm)
    dm_p, hz_p = bao_aniso_observables(z, pbuf)
    DMrd_LCDM.append(dm_l)
    DHrd_LCDM.append(hz_l)
    DMrd_PBUF.append(dm_p)
    DHrd_PBUF.append(hz_p)

DMrd_LCDM = np.array(DMrd_LCDM)
DHrd_LCDM = np.array(DHrd_LCDM)
DMrd_PBUF = np.array(DMrd_PBUF)
DHrd_PBUF = np.array(DHrd_PBUF)

# ----------------------------------------
# 4. Report
# ----------------------------------------
print("z    DM/rd_LCDM  DM/rd_PBUF   DH/rd_LCDM    DH/rd_PBUF")
for i, z in enumerate(z_values):
    print(f"{z:<4} {DMrd_LCDM[i]:12.6f} {DMrd_PBUF[i]:12.6f} {DHrd_LCDM[i]:14.6f} {DHrd_PBUF[i]:14.6f}")

# ----------------------------------------
# 5. Sanity checks
# ----------------------------------------

# D_M / r_d should grow with z (farther transverse distance per BAO ruler).
assert np.all(np.diff(DMrd_LCDM) > 0), "D_M/r_d not increasing for LCDM"
assert np.all(np.diff(DMrd_PBUF) > 0), "D_M/r_d not increasing for PBUF"

# D_H / r_d should fall with z as H(z) grows.
assert np.all(np.diff(DHrd_LCDM) < 0), "D_H/r_d not decreasing for LCDM"
assert np.all(np.diff(DHrd_PBUF) < 0), "D_H/r_d not decreasing for PBUF"

# No nonsense numbers
assert np.all(np.isfinite(DMrd_LCDM)) and np.all(DMrd_LCDM > 0)
assert np.all(np.isfinite(DMrd_PBUF)) and np.all(DMrd_PBUF > 0)
assert np.all(np.isfinite(DHrd_LCDM)) and np.all(DHrd_LCDM > 0)
assert np.all(np.isfinite(DHrd_PBUF)) and np.all(DHrd_PBUF > 0)

print("\n✅ BAO anisotropic test passed.")
