"""
LCDM cosmological parameter optimizer.

This module implements a two-phase hybrid optimization strategy for fitting
LCDM cosmological parameters against CMB distance priors.

Phase 1: Coarse global grid search over (H0, Om0, Ok0)
Phase 2: Local refinement around top candidates using scipy.optimize.minimize

The optimizer enforces physical closure: Ω_total(a=1) = 1 with Ω_Λ ≥ 0.
"""

import numpy as np
from scipy.optimize import minimize
from cosmos.lcdm.model import LCDM
from cosmos.fits.cmb.chi2 import chi_squared_cmb


def _build_lcdm_model(H0, Om0, Or0, Ok0):
    """
    Build LCDM model from physical parameters.

    Parameters
    ----------
    H0 : float
        Hubble constant [km/s/Mpc]
    Om0 : float
        Matter density parameter today
    Or0 : float
        Radiation density parameter today
    Ok0 : float
        Curvature density parameter today

    Returns
    -------
    LCDM
        LCDM model instance

    Raises
    ------
    ValueError
        If derived Ω_Λ < 0 (unphysical)
    """
    Ol0 = 1.0 - (Om0 + Or0 + Ok0)
    if Ol0 < 0:
        raise ValueError("Negative Omega_Lambda, unphysical.")

    h = H0 / 100.0
    omega_b = 0.02237 / (h**2)  # physical-ish baryon density

    return LCDM(
        omega_m=Om0,
        omega_lambda=Ol0,
        h=h,
        omega_k=Ok0,
        omega_r=Or0,
        omega_b=omega_b,
        T_cmb=2.7255
    )


def _lcdm_objective(x, Or0_fixed):
    """
    Objective function for LCDM optimization.

    Parameters
    ----------
    x : array-like
        [H0, Om0, Ok0] parameters
    Or0_fixed : float
        Fixed radiation density parameter

    Returns
    -------
    float
        χ² value (or large penalty for unphysical parameters)
    """
    H0, Om0, Ok0 = x
    Ol0 = 1.0 - (Om0 + Or0_fixed + Ok0)

    if Ol0 < 0:
        return 1e30  # Large penalty for unphysical closure

    try:
        model = _build_lcdm_model(H0, Om0, Or0_fixed, Ok0)
        return chi_squared_cmb(model)
    except Exception:
        return 1e30  # Penalty for model construction failures


def optimise_against_cmb(verbose: bool = True) -> dict:
    """
    Optimize LCDM parameters against CMB distance priors.

    Uses a two-phase approach:
    1. Coarse grid search over (H0, Om0, Ok0) with physical constraints
    2. Local refinement around top candidates using scipy.optimize.minimize

    Parameters
    ----------
    verbose : bool, optional
        Print progress information

    Returns
    -------
    dict
        Optimization results with keys:
        - success: bool
        - message: str
        - best_params: dict of best-fit parameters
        - best_chi2: float
        - history: list of attempted parameter sets
    """
    Or0_fixed = 9.2e-5  # Planck 2018 value

    # Phase 1: coarse grid search
    H0_grid = np.linspace(60.0, 80.0, 7)      # 7 points
    Om0_grid = np.linspace(0.25, 0.35, 7)     # 7 points
    Ok0_grid = np.linspace(-0.01, 0.01, 5)    # 5 points

    candidates = []
    history = []

    for H0 in H0_grid:
        for Om0 in Om0_grid:
            for Ok0 in Ok0_grid:
                Ol0 = 1.0 - (Om0 + Or0_fixed + Ok0)
                if Ol0 < 0:
                    continue  # Skip unphysical parameter sets

                chi2 = _lcdm_objective([H0, Om0, Ok0], Or0_fixed)
                candidates.append((chi2, (H0, Om0, Ok0)))

                history.append({
                    "stage": "grid",
                    "params": {
                        "H0": H0,
                        "Om0": Om0,
                        "Ok0": Ok0,
                        "Or0": Or0_fixed,
                        "Ol0": Ol0
                    },
                    "chi2": chi2,
                })

                if verbose:
                    print(f"[GRID] H0={H0:.2f} Om0={Om0:.3f} Ok0={Ok0:+.4f} -> χ²={chi2:.6f}")

    # Sort by χ² and keep top candidates
    candidates.sort(key=lambda item: item[0])
    top_candidates = candidates[:10]  # Keep 10 best from grid

    # Phase 2: local refinement around each survivor
    bounds = [
        (60.0, 80.0),   # H0 [km/s/Mpc]
        (0.1, 0.6),     # Om0
        (-0.1, 0.1),    # Ok0
    ]

    best_refined = None

    for chi2_seed, (H0_seed, Om0_seed, Ok0_seed) in top_candidates:
        res = minimize(
            lambda x: _lcdm_objective(x, Or0_fixed),
            x0=np.array([H0_seed, Om0_seed, Ok0_seed], dtype=float),
            bounds=bounds,
            method="L-BFGS-B",
        )

        H0_fit, Om0_fit, Ok0_fit = res.x
        chi2_fit = _lcdm_objective(res.x, Or0_fixed)
        Ol0_fit = 1.0 - (Om0_fit + Or0_fixed + Ok0_fit)

        history.append({
            "stage": "refine",
            "params": {
                "H0": H0_fit,
                "Om0": Om0_fit,
                "Ok0": Ok0_fit,
                "Or0": Or0_fixed,
                "Ol0": Ol0_fit,
            },
            "chi2": chi2_fit,
        })

        if verbose:
            print(f"[REFINE] H0={H0_fit:.4f} Om0={Om0_fit:.5f} Ok0={Ok0_fit:+.5f} χ²={chi2_fit:.6f}")

        # Track the best solution
        if (best_refined is None) or (chi2_fit < best_refined[0]):
            best_refined = (chi2_fit, H0_fit, Om0_fit, Ok0_fit)

    if best_refined is None:
        return {
            "success": False,
            "message": "No valid LCDM solution found.",
            "best_chi2": float('inf'),
            "best_params": {},
            "history": history,
        }

    best_chi2, H0_best, Om0_best, Ok0_best = best_refined
    Or0_best = Or0_fixed
    Ol0_best = 1.0 - (Om0_best + Or0_best + Ok0_best)

    return {
        "success": True,
        "message": "LCDM optimization complete.",
        "best_chi2": float(best_chi2),
        "best_params": {
            "H0": float(H0_best),
            "Om0": float(Om0_best),
            "Or0": float(Or0_best),
            "Ok0": float(Ok0_best),
            "Ol0": float(Ol0_best),
        },
        "history": history,
    }
