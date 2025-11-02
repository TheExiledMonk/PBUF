"""
Diagnostic test for PBUF parameter sensitivity.

Tests whether χ² varies with different (α, Rmax, k_sat) values.
If all χ² are identical, the parameters aren't affecting the cosmology.
"""

from cosmos.pbuf.model import PBUF
from cosmos.fits.cmb.chi2 import chi_squared_cmb

def test_param_sensitivity():
    """Test parameter sensitivity by comparing χ² for different parameter values."""

    base = dict(
        omega_m=0.3153,
        h=0.6736,
        omega_k=0.0,
        omega_r=9.2e-5,
        omega_b=0.02237/(0.6736**2),
        T_cmb=2.7255
    )

    test_cases = [
        (5.0e-4, 1.0e8, 0.8, 0.7),
        (2.0e-3, 5.0e8, 1.2, 0.9),
        (1.0e-2, 1.0e9, 0.6, 1.1)
    ]

    print("Testing PBUF parameter sensitivity...")
    print("Format: α={α:.2e} Rmax={Rmax:.2e} k_sat={k_sat:.2f} eps0={eps0:.2f} → χ²={χ²:.6f}")
    print("-" * 60)

    for alpha, Rmax, k_sat, eps0 in test_cases:
        model = PBUF(**base, alpha=alpha, Rmax=Rmax, k_sat=k_sat, eps0=eps0)
        chi2 = chi_squared_cmb(model)
        print(f"α={alpha:.2e} Rmax={Rmax:.2e} k_sat={k_sat:.2f} eps0={eps0:.2f} → χ²={chi2:.6f}")

    print("-" * 60)
    print("If all χ² values are identical, parameters aren't being used!")

if __name__ == "__main__":
    test_param_sensitivity()
