"""
Quick test of optimizer with smaller grid to verify it's working.
"""

import numpy as np
from cosmos.pbuf.model import PBUF
from cosmos.fits.cmb.chi2 import chi_squared_cmb

def quick_optimizer_test():
    """Test optimizer with a small grid."""

    # Fixed background parameters
    fixed_bg = {
        "H0": 67.36,
        "Om0": 0.3153,
        "Or0": 9.2e-05,
        "Ok0": 0.0,
    }

    # Small grid for testing
    alphas = [5.0e-4, 2.0e-3, 1.0e-2]      # 3 values
    Rmaxs = [1.0e8, 5.0e8, 1.0e9]         # 3 values
    k_sats = [0.6, 1.0, 1.5]              # 3 values
    eps0 = 0.7

    best_chi2 = float('inf')
    best_params = None

    print("Quick optimizer test (3x3x3 = 27 evaluations)...")
    print("=" * 60)

    for alpha in alphas:
        for Rmax in Rmaxs:
            for k_sat in k_sats:
                # Build model
                h = fixed_bg["H0"] / 100.0
                omega_b = 0.02237 / (h**2)

                model = PBUF(
                    omega_m=fixed_bg["Om0"],
                    h=h,
                    alpha=alpha,
                    Rmax=Rmax,
                    k_sat=k_sat,
                    eps0=eps0,
                    omega_k=fixed_bg["Ok0"],
                    omega_r=fixed_bg["Or0"],
                    omega_b=omega_b,
                )

                # Compute chi2
                chi2 = chi_squared_cmb(model)

                print(f"α={alpha:.2e} Rmax={Rmax:.2e} k_sat={k_sat:.2f} ε0={eps0:.2f} → χ²={chi2:.6f}")
                # Track best
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = (alpha, Rmax, k_sat)
    print(f"Best parameters: α={best_params[0]:.2e}, Rmax={best_params[1]:.2e}, k_sat={best_params[2]:.2f}")
    print(f"Best χ²: {best_chi2:.6f}")

if __name__ == "__main__":
    quick_optimizer_test()
