import numpy as np

from cosmos2.wl.scale_cuts import build_scale_cut_mask, build_custom_scale_cuts, kids_default_scale_cuts


def test_custom_scale_cut_table_overrides_defaults() -> None:
    n_bins = 2
    theta = np.deg2rad(np.array([1.0, 5.0, 10.0]))  # 60, 300, 600 arcsec
    pairs = [(0, 0), (0, 1), (1, 1)]
    default = kids_default_scale_cuts(n_bins, xi_plus_min_arcmin=0.5, xi_minus_min_arcmin=4.2, xi_plus_max_arcmin=300.0, xi_minus_max_arcmin=300.0)
    # Override xi- min for pair (0,1) to 8 arcmin (~0.00233 rad)
    table = {(0, 1): (None, None, np.deg2rad(8.0 / 60.0), None)}
    cuts = build_custom_scale_cuts(n_bins, table, default=default)
    n_theta = theta.size
    theta2 = np.array([np.deg2rad(2.0 / 60.0), theta[1], theta[2]])
    mask2 = build_scale_cut_mask(theta2, pairs, cuts)
    pair_idx = 1  # (0,1)
    xi_minus_start = (len(pairs) * n_theta) + pair_idx * n_theta
    assert bool(mask2.combined[xi_minus_start]) is False  # first theta dropped
    assert bool(mask2.combined[xi_minus_start + 1])
    assert bool(mask2.combined[xi_minus_start + 2])
