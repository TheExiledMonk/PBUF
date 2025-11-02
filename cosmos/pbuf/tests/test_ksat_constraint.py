"""
Ensure the restored optimizer allows k_sat > 1 without hard rejects.
"""

from cosmos.pbuf.optimizer import _apply_physics_priors


def test_ksat_above_unity_remains_allowed():
    fixed_bg = {
        "H0": 67.36,
        "Om0": 0.3153,
        "Or0": 9.2e-5,
        "Ok0": 0.0,
    }
    alpha = 1.0e-3
    Rmax = 1.0e9

    for k_sat in (0.5, 1.0, 2.0, 3.0):
        penalty = _apply_physics_priors(alpha, Rmax, k_sat, fixed_bg)
        assert penalty < 1e20, f"k_sat={k_sat} should not trigger a hard penalty"
