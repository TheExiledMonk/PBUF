"""
Detailed diagnostic for PBUF parameter sensitivity.

Tests both χ² and individual CMB observables to isolate the issue.
"""

import numpy as np
from cosmos.pbuf.model import PBUF
from cosmos.fits.cmb.observables import cmb_observables
from cosmos.fits.cmb.chi2 import chi_squared_cmb

def test_detailed_sensitivity():
    """Test both observables and χ² for different parameter values."""

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

    print("Detailed PBUF parameter sensitivity analysis...")
    print("=" * 80)
    print(f"{'α':<10} {'Rmax':<10} {'k_sat':<8} {'ε₀':<8} {'R':<12} {'l_A':<10} {'θ*':<10} {'χ²':<10}")
    print("-" * 80)

    for alpha, Rmax, k_sat, eps0 in test_cases:
        model = PBUF(**base, alpha=alpha, Rmax=Rmax, k_sat=k_sat, eps0=eps0)

        # Check CMB observables
        obs = cmb_observables(model)
        chi2 = chi_squared_cmb(model)

        print(f"{alpha:.2e} {Rmax:.2e} {k_sat:.2f}   {eps0:.2f}   {obs['R']:.6f} {obs['la']:.3f} {obs['theta_star']:.6f} {chi2:.6f}")

        # Check elastic contribution at different redshifts
        for z in [0, 1, 10, 100, 1000]:
            try:
                elastic_frac = model.density_parameters_at_z(z)['omega_sigma']
                print(f"  z={z:4d}: Ω_σ/Ω_total = {elastic_frac:.6f}")
            except:
                pass
        print()

if __name__ == "__main__":
    test_detailed_sensitivity()
