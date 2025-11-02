"""
Quick test of physics priors with PBUF optimizer.
"""

import numpy as np
from cosmos.pbuf.optimizer import _pbuf_objective_logparams

def quick_pbuf_test():
    """Test PBUF optimization with physics priors."""

    # Fixed background parameters (Planck-like)
    fixed_bg = {
        "H0": 67.36,
        "Om0": 0.3153,
        "Or0": 9.2e-05,
        "Ok0": 0.0,
    }

    print("Testing PBUF optimization with physics priors...")
    print("=" * 60)

    # Test a few parameter combinations
    test_cases = [
        (1e-3, 1e9, 1.1),    # physically reasonable
        (1e-2, 5e8, 0.8),    # also reasonable
        (1e-1, 1e6, 0.5),    # edge case
        (1.0, 0.1, 1.0),     # should be heavily penalized
    ]

    for alpha, Rmax, k_sat in test_cases:
        log_params = [np.log10(alpha), np.log10(Rmax), np.log10(k_sat)]
        chi2 = _pbuf_objective_logparams(log_params, fixed_bg)
        print(f"α={alpha:.1e} Rmax={Rmax:.1e} k_sat={k_sat:.1f} → χ²={chi2:.6f}")

    print("\n" + "=" * 60)
    print("✅ Quick test complete!")

if __name__ == "__main__":
    quick_pbuf_test()
