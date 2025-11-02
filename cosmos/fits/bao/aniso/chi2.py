"""
Compute χ² for anisotropic BAO fits.

This module provides functions to compute chi-squared statistics
for anisotropic BAO measurements using the standardized PBUF data format.
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset
from cosmos.fits._dataset_loader import load_bao_aniso_dataset
from .observables import compute_bao_anisotropic_observables

def chi_squared_bao_aniso(model, data=None):
    """
    Compute χ² for anisotropic BAO constraints using standardized format.

    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model with DM_over_rd and DH_over_rd methods
    data : dict or None
        Standardized BAO anisotropic dataset (PBUF Data Object v1). If None, loads default.

    Returns
    -------
    float
        Total χ² summed across all BAO redshift bins.

    Notes
    -----
    Uses the standardized PBUF data format with keys: name, type, z, obs, err, cov, meta.
    For anisotropic BAO, obs contains concatenated [DM/rd, D_H/rd] measurements.
    """
    # Load and validate data using standard schema
    if data is None:
        data = load_bao_aniso_dataset()

    # Ensure data follows PBUF Data Object v1 schema
    data = ensure_standard_dataset(data, "BAO_ANISO")

    # Extract standardized fields
    z = data["z"]
    obs = data["obs"]  # Concatenated [DM/rd, D_H/r_d] measurements
    err = data["err"]
    cov = data["cov"]

    n_points = len(z)
    expected_size = 2 * n_points
    if obs.shape[0] != expected_size:
        raise ValueError(
            f"Anisotropic BAO dataset has {obs.shape[0]} observables but expected {expected_size}"
        )
    if err is not None and err.shape[0] != expected_size:
        raise ValueError(
            f"Anisotropic BAO error vector has shape {err.shape} but expected ({expected_size},)"
        )

    # Compute model predictions
    preds = compute_bao_anisotropic_observables(model, z)
    DM_model = preds["DM_over_rd"]
    DH_model = preds["DH_over_rd"]

    # For anisotropic BAO, obs is interleaved as [DM1, D_H1, DM2, D_H2, ...]
    # We need to separate them back into DM and D_H components
    DM_obs = obs[0::2]  # Even indices: DM measurements
    DH_obs = obs[1::2]  # Odd indices: D_H measurements

    # Split errors accordingly
    if err is not None:
        DM_err = err[0::2]
        DH_err = err[1::2]
    else:
        DM_err = None
        DH_err = None

    chi2_total = 0.0

    if cov is not None:
        if cov.shape != (expected_size, expected_size):
            raise ValueError(
                f"Anisotropic BAO covariance shape {cov.shape} does not match expected {(expected_size, expected_size)}"
            )
        if not np.allclose(cov, cov.T, atol=1e-12):
            raise ValueError("Anisotropic BAO covariance matrix must be symmetric.")

        # Attempt to detect whether the covariance matches the interleaved observable order.
        def _allclose(a, b):
            return np.allclose(a, b, rtol=5e-3, atol=1e-10)

        diag = np.clip(np.diag(cov), 0.0, None)
        inferred_err = np.sqrt(diag)

        def _permute_block_to_interleaved(matrix):
            """Reorder covariance that is stored as [DM..., DH...] to interleaved order."""
            perm = np.empty(expected_size, dtype=int)
            perm[0::2] = np.arange(n_points)
            perm[1::2] = np.arange(n_points, expected_size)
            return matrix[np.ix_(perm, perm)]

        # Determine ordering: interleaved vs block (all DM then all DH).
        if err is not None:
            # Compare provided uncertainties with diagonal entries.
            interleaved_match = _allclose(inferred_err[0::2], err[0::2]) and _allclose(
                inferred_err[1::2], err[1::2]
            )
            if not interleaved_match:
                half = expected_size // 2
                block_match = _allclose(np.sqrt(diag[:half]), err[0::2]) and _allclose(
                    np.sqrt(diag[half:]), err[1::2]
                )
                if block_match:
                    cov = _permute_block_to_interleaved(cov)
                    diag = np.clip(np.diag(cov), 0.0, None)
                    inferred_err = np.sqrt(diag)
                else:
                    raise ValueError(
                        "Anisotropic BAO covariance ordering does not match the provided uncertainties."
                    )
        else:
            # Use covariance signal to guess ordering when err is missing.
            interleaved_signal = 0.0
            block_signal = 0.0
            half = expected_size // 2
            for i in range(n_points):
                interleaved_signal += abs(cov[2 * i, 2 * i + 1]) + abs(cov[2 * i + 1, 2 * i])
                block_signal += abs(cov[i, half + i]) + abs(cov[half + i, i])
            if block_signal > interleaved_signal * 1.1:
                cov = _permute_block_to_interleaved(cov)
                diag = np.clip(np.diag(cov), 0.0, None)
                inferred_err = np.sqrt(diag)

        if err is None:
            # Populate errors from covariance for downstream calculations.
            DM_err = inferred_err[0::2]
            DH_err = inferred_err[1::2]
        elif err is not None:
            # After reordering make sure the diagonal agrees.
            if not (_allclose(inferred_err[0::2], err[0::2]) and _allclose(inferred_err[1::2], err[1::2])):
                raise ValueError(
                    "Anisotropic BAO covariance diagonal does not agree with the supplied uncertainties."
                )

        # Use full covariance matrix if available
        cov_inv = np.linalg.inv(cov)
        # Interleave predictions to match data format: [DM1, D_H1, DM2, D_H2, ...]
        pred_interleaved = np.empty_like(obs)
        pred_interleaved[0::2] = DM_model  # Even indices: DM predictions
        pred_interleaved[1::2] = DH_model  # Odd indices: D_H predictions
        diff = obs - pred_interleaved
        chi2_total = float(diff.T @ cov_inv @ diff)
    else:
        # Use individual errors (diagonal approximation)
        if DM_err is None or DH_err is None:
            raise ValueError(
                "Anisotropic BAO dataset must include 1σ uncertainties when covariance is not provided."
            )
        for i in range(n_points):
            diff_DM = DM_obs[i] - DM_model[i]
            diff_DH = DH_obs[i] - DH_model[i]
            chi2_total += (diff_DM / DM_err[i])**2 + (diff_DH / DH_err[i])**2

    return max(float(chi2_total), 0.0)
