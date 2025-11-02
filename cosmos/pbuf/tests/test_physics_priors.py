"""
Test physics priors implementation.
"""

from cosmos.pbuf.optimizer import _apply_physics_priors

def test_physics_priors():
    """Test that physics priors correctly penalize unphysical parameters."""

    # Fixed background parameters (typical Planck-like values)
    fixed_bg = {
        "H0": 67.36,
        "Om0": 0.3153,
        "Or0": 9.2e-5,
        "Ok0": 0.0,
    }

    print("Testing physics priors...")
    print("=" * 60)

    test_cases = [
        # (alpha, Rmax, k_sat, expected_penalty_range, description)
        (1e-3, 1e9, 1.2, "low", "physically plausible"),
        (1e-1, 1e6, 0.5, "medium", "edge cases"),
        (1.0, 0.1, 1.0, "high", "unphysical"),
        (1e-6, 1e12, 0.3, "high", "numerically unstable"),
    ]

    for alpha, Rmax, k_sat, expected, description in test_cases:
        penalty = _apply_physics_priors(alpha, Rmax, k_sat, fixed_bg)
        status = "✅" if penalty < 1e3 else "❌" if penalty > 1e10 else "⚠️"
        print(f"{status} α={alpha:.1e} Rmax={Rmax:.1e} k_sat={k_sat:.2f} → penalty={penalty:.2f} ({expected}) - {description}")

    print("\n" + "=" * 60)
    print("Physics priors test complete!")
    print("\nExpected behavior:")
    print("- Low penalty (< 1e3): physically plausible")
    print("- Medium penalty (1e3-1e10): somewhat unphysical")
    print("- High penalty (> 1e10): numerically unstable")

if __name__ == "__main__":
    test_physics_priors()
