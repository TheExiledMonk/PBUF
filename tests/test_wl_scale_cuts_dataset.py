import numpy as np

from cosmos2.data.registry import load_weak_lensing_kids1000
from cosmos2.wl.kids import tomo_pairs
from cosmos2.wl.scale_cuts import apply_scale_cuts, build_scale_cut_mask, kids_default_scale_cuts


def test_scale_cuts_reduce_vector_and_cov_consistently() -> None:
    dataset = load_weak_lensing_kids1000()
    data_vector = np.asarray(dataset["data_vector"], dtype=float)
    covariance = np.asarray(dataset["covariance"], dtype=float)
    theta_bins = np.asarray(dataset["theta_bins"], dtype=float)
    n_of_z = np.asarray(dataset["n_of_z"], dtype=float)

    pairs_arr = dataset.get("tomo_pairs")
    if pairs_arr is None or (isinstance(pairs_arr, np.ndarray) and pairs_arr.size == 0):
        pairs_arr = tomo_pairs(n_of_z.shape[0])
    pairs = [tuple(pair) for pair in np.asarray(pairs_arr, dtype=int)]
    cuts = kids_default_scale_cuts(n_of_z.shape[0])
    mask = build_scale_cut_mask(theta_bins, pairs, cuts)

    dv_cut, cov_cut = apply_scale_cuts(data_vector, covariance, mask)
    assert dv_cut.shape[0] == cov_cut.shape[0] == cov_cut.shape[1]
    # Ensure some points were actually removed
    assert dv_cut.shape[0] < data_vector.shape[0]
    # Official KiDS cuts should retain a deterministic number of entries
    assert dv_cut.shape[0] == 225
