"""Photo-z shift utilities for tomographic n(z)."""

from __future__ import annotations

import numpy as np


def apply_photoz_shifts(n_of_z: np.ndarray, z_grid: np.ndarray, shifts: np.ndarray | None) -> np.ndarray:
    """
    Shift each tomographic n_i(z) → n_i(z + Δz_i) with renormalization.

    Parameters
    ----------
    n_of_z : (n_bins, n_z)
        Original redshift distributions.
    z_grid : (n_z,)
        Redshift grid corresponding to n(z).
    shifts : (n_bins,)
        Δz_i for each tomographic bin (can be None/zeros).
    """

    nz = np.asarray(n_of_z, dtype=float)
    z = np.asarray(z_grid, dtype=float)
    if shifts is None:
        return nz
    shifts = np.asarray(shifts, dtype=float)
    if shifts.size != nz.shape[0]:
        raise ValueError("photo-z shifts length must match number of tomographic bins.")

    shifted = np.zeros_like(nz)
    for i in range(nz.shape[0]):
        dz = float(shifts[i])
        # Sample the shifted distribution; values outside the original grid are set to zero.
        shifted[i] = np.interp(z + dz, z, nz[i], left=0.0, right=0.0)
        area = np.trapezoid(shifted[i], z)
        if area > 0.0:
            shifted[i] /= area
    return shifted


__all__ = ["apply_photoz_shifts"]
