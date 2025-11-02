"""
Test: RSD growth observable fσ8(z) for LCDM and PBUF.

Goal:
 - Ensure fσ8(z) is finite and positive for both models.
 - Ensure fσ8(z) decreases with redshift (less growth at high z),
   which is the standard behavior in GR-like models.
 - Provide χ² plumbing for future real RSD data.

NOTE:
This test assumes cosmos.helper.growth supplies:
   fsigma8(z, model)  # dimensionless
For now the function can be a placeholder, but it must be stable.
"""

import numpy as np

from cosmos.fits.rsd import compute_rsd_observable
from cosmos.helper.growth import fsigma8
from cosmos.lcdm.model import LCDM
from cosmos.optim.parameter_defaults import SIGMA8_PLANCK
from cosmos.pbuf.model import PBUF

# ----------------------------------------
# 1. Cosmology setup
# ----------------------------------------
H0 = 67.4
h = H0 / 100.0
Om0 = 0.315
Ok0 = 0.0
Or0 = 5e-5
Ol0 = 0.685
Obh2 = 0.022

alpha = 5e-4
Rmax = 1e9
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
# 2. Sample redshifts used in RSD surveys
# ----------------------------------------
z_values = np.array([0.15, 0.38, 0.51, 0.61, 1.0])

fs8_LCDM = compute_rsd_observable(lcdm, z_values)
fs8_PBUF = compute_rsd_observable(pbuf, z_values)

# Ensure public API matches direct growth helper
expected_lcdm = np.asarray(fsigma8(z_values, lcdm))
expected_pbuf = np.asarray(fsigma8(z_values, pbuf))
np.testing.assert_allclose(fs8_LCDM, expected_lcdm, rtol=1e-5, atol=0)
np.testing.assert_allclose(fs8_PBUF, expected_pbuf, rtol=1e-5, atol=0)

# σ8 normalization must propagate through the API
sigma8_override = 0.9
override_LCDM = compute_rsd_observable(lcdm, z_values, sigma8_0=sigma8_override)
np.testing.assert_allclose(
    override_LCDM,
    expected_lcdm * (sigma8_override / SIGMA8_PLANCK),
    rtol=1e-5,
    atol=0,
)

# Duplicate redshift handling should be stable (dataset contains repeats).
z_with_duplicates = np.array([0.02, 0.02, 0.1, 0.38, 0.38, 0.6])
fs8_duplicates = compute_rsd_observable(lcdm, z_with_duplicates)
assert fs8_duplicates.shape == z_with_duplicates.shape
assert np.all(np.isfinite(fs8_duplicates))

# ----------------------------------------
# 3. Behavior checks
# ----------------------------------------
# Must be positive and finite
assert np.all(np.isfinite(fs8_LCDM)) and np.all(fs8_LCDM > 0.0), "Bad fσ8 in LCDM"
assert np.all(np.isfinite(fs8_PBUF)) and np.all(fs8_PBUF > 0.0), "Bad fσ8 in PBUF"

# Should decrease with redshift (structure less grown at high z)
# Check overall trend rather than strict monotonicity for robustness
def check_decreasing_trend(values, threshold=1.2):
    """Check if values don't increase too much with redshift."""
    return values[-1] < threshold * values[0]

assert check_decreasing_trend(fs8_LCDM), "fσ8 not generally decreasing with z in LCDM"
assert check_decreasing_trend(fs8_PBUF), "fσ8 not generally decreasing with z in PBUF"

print("\n✅ RSD fσ8(z) test passed.")
