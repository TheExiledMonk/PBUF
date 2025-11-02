"""
PBUF cosmological parameter optimizer.

This module implements a two-phase hybrid optimization strategy for fitting
PBUF cosmological parameters against CMB distance priors.

Phase 1: Coarse global grid search over (alpha, Rmax, k_sat) in log-space
Phase 2: Local refinement around top candidates using scipy.optimize.minimize

The optimizer enforces physical constraints: k_sat > 0, alpha > 0, Rmax > 0.
"""

import numpy as np
from math import log10
from scipy.optimize import minimize
from cosmos.pbuf.model import PBUF
from cosmos.fits.cmb.chi2 import chi_squared_cmb


def _build_pbuf_model(H0, Om0, Or0, Ok0, alpha, Rmax, k_sat,
                      eps0=0.7, n_alpha=0.0, n_eps=0.0, n_R=0.0):
    """
    Build PBUF model from physical parameters.

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
    alpha : float
        Elastic amplitude parameter
    Rmax : float
        Saturation scale factor for elastic response
    k_sat : float
        Rigidity fraction (must be > 0)
    eps0 : float
        Baseline elastic stiffness value
    n_alpha, n_eps, n_R : float
        Power-law evolution indices for α(z), ε(z), Rmax(z)

    Returns
    -------
    PBUF
        PBUF model instance

    Raises
    ------
    ValueError
        If parameters are outside physical bounds
    """
    h = H0 / 100.0
    omega_b = 0.02237 / (h**2)  # physical-ish baryon density

    return PBUF(
        omega_m=Om0,
        h=h,
        alpha=alpha,
        Rmax=Rmax,
        k_sat=k_sat,
        eps0=eps0,
        n_alpha=n_alpha,
        n_eps=n_eps,
        n_R=n_R,
        omega_k=Ok0,
        omega_r=Or0,
        omega_b=omega_b,
        T_cmb=2.7255
    )


def _apply_physics_priors(alpha, Rmax, k_sat, fixed_bg):
    """
    Apply physical plausibility priors to penalize unphysical or implausible
    parameter regions while allowing exploration near the edges.

    Returns a penalty term (float) that can be added to χ².

    Parameters
    ----------
    alpha : float
        Elastic amplitude parameter
    Rmax : float
        Saturation scale factor for elastic response
    k_sat : float
        Rigidity fraction (k_sat > 0)
    fixed_bg : dict
        Background parameters containing H0, Om0, Or0, Ok0

    Returns
    -------
    float
        Soft penalty to add to χ² (0 means fully physical)
    """

    penalty = 0.0

    # --- 1. Numerical sanity (hard guards) ---
    if (not np.isfinite(alpha)) or (not np.isfinite(Rmax)) or (not np.isfinite(k_sat)):
        return 1e30
    if alpha < 0 or Rmax <= 0 or k_sat <= 0:
        return 1e30

    # --- 2. Physical plausibility (soft priors) ---

    # Matter density: Planck 2018 constraints ~0.3 ± 0.05
    Om0 = fixed_bg.get("Om0", 0.3)
    if not (0.25 <= Om0 <= 0.35):
        penalty += ((Om0 - 0.3) / 0.05) ** 2 * 1e2

    # Curvature: must be nearly flat |Ωk| < 0.005
    Ok0 = fixed_bg.get("Ok0", 0.0)
    if abs(Ok0) > 0.005:
        penalty += ((abs(Ok0) - 0.005) / 0.005) ** 2 * 1e3

    # Radiation density: should be fixed to ~9e-5
    Or0 = fixed_bg.get("Or0", 9.2e-5)
    if abs(Or0 - 9.2e-5) / 9.2e-5 > 0.2:  # 20% deviation allowed
        penalty += ((Or0 - 9.2e-5) / 9.2e-5) ** 2 * 1e3

    # Alpha: physically small, late-time correction
    if alpha < 1e-6 or alpha > 1e-1:
        penalty += (np.log10(alpha) - np.log10(1e-3)) ** 2 * 1e3

    # Rmax: should correspond to late-time onset (10⁶–10¹² typical)
    if Rmax < 1e6 or Rmax > 1e12:
        penalty += (np.log10(Rmax) - 9.0) ** 2 * 1e3

    # k_sat: allow broader range but discourage extreme values
    if k_sat < 0.5 or k_sat > 3.0:
        penalty += (k_sat - 1.5) ** 2 * 1e3
    # Gentle regularisation toward unity
    penalty += ((k_sat - 1.0) ** 2) * 1e-2

    return penalty


def _pbuf_objective_logparams(x, fixed_bg):
    """
    Objective function for PBUF optimization in log-parameter space.

    Parameters
    ----------
    x : array-like
        [log10(alpha), log10(Rmax), log10(k_sat)] parameters
    fixed_bg : dict
        Fixed background parameters (H0, Om0, Or0, Ok0)

    Returns
    -------
    float
        χ² value (or large penalty for unphysical parameters)
    """
    log_alpha, log_Rmax, log_k = x

    # Convert from log space to physical parameters
    alpha = 10.0 ** log_alpha
    Rmax = 10.0 ** log_Rmax
    k_sat = 10.0 ** log_k

    # Apply physical priors (includes both hard and soft constraints)
    penalty = _apply_physics_priors(alpha, Rmax, k_sat, fixed_bg)
    if penalty >= 1e20:
        return penalty  # skip unphysical region

    eps0 = float(fixed_bg.get("eps0", 0.7))
    n_alpha = float(fixed_bg.get("n_alpha", 0.0))
    n_eps = float(fixed_bg.get("n_eps", 0.0))
    n_R = float(fixed_bg.get("n_R", 0.0))

    try:
        model = _build_pbuf_model(
            fixed_bg["H0"], fixed_bg["Om0"], fixed_bg["Or0"], fixed_bg["Ok0"],
            alpha, Rmax, k_sat,
            eps0=eps0,
            n_alpha=n_alpha,
            n_eps=n_eps,
            n_R=n_R,
        )
        chi2 = chi_squared_cmb(model)
    except Exception:
        return 1e30  # Penalty for model construction failures

    return chi2 + penalty


def optimise_against_cmb(verbose: bool = True) -> dict:
    # Fixed background parameters (matching Planck-like baseline)
    fixed_bg = {
        "H0": 67.36,
        "Om0": 0.3153,
        "Or0": 9.2e-05,
        "Ok0": 0.0,
    }

    # Phase 1: coarse grid search in PHYSICALLY PLAUSIBLE parameter space (REDUCED FOR TESTING)
    alphas = np.logspace(-6, -1, 10)   # alpha: 1e-6 to 1e-1 (10 points for faster testing)
    Rmaxs = np.logspace(6, 12, 10)     # Rmax: 1e6 to 1e12 (10 points for faster testing)
    k_sats = np.linspace(0.5, 3.0, 10)   # k_sat: 0.5 to 3.0 (10 points for faster testing)

    candidates = []
    history = []

    for alpha in alphas:
        for Rmax in Rmaxs:
            for k_sat in k_sats:

                # Compute χ² for this parameter combination
                log_params = [log10(alpha), log10(Rmax), log10(k_sat)]
                chi2 = _pbuf_objective_logparams(log_params, fixed_bg)

                candidates.append((chi2, (alpha, Rmax, k_sat)))

                history.append({
                    "stage": "grid",
                    "params": {
                        "H0": fixed_bg["H0"],
                        "Om0": fixed_bg["Om0"],
                        "Or0": fixed_bg["Or0"],
                        "Ok0": fixed_bg["Ok0"],
                        "alpha": alpha,
                        "Rmax": Rmax,
                        "k_sat": k_sat,
                    },
                    "chi2": chi2,
                })

                if verbose:
                    print(f"[GRID] α={alpha:.3f} Rmax={Rmax:.3f} k_sat={k_sat:.3f} -> χ²={chi2:.6f}")

    # Sort by χ² and keep top candidates
    candidates.sort(key=lambda item: item[0])
    top_candidates = candidates[:10]  # Keep 10 best from grid

    # Phase 2: local refinement around each survivor in log-space
    bounds = [
        (-6.0, -1.0),              # log10(alpha): -6 to -1 (1e-6 to 1e-1)
        (6.0, 12.0),               # log10(Rmax): 6 to 12 (1e6 to 1e12)
        (np.log10(0.5), np.log10(3.0)),    # log10(k_sat): 0.5 to 3.0
    ]

    best_refined = None

    for chi2_seed, (alpha_seed, Rmax_seed, k_sat_seed) in top_candidates:
        x0 = np.array([log10(alpha_seed), log10(Rmax_seed), log10(k_sat_seed)], dtype=float)

        res = minimize(
            lambda x: _pbuf_objective_logparams(x, fixed_bg),
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
        )

        # Evaluate the refined solution
        chi2_fit = _pbuf_objective_logparams(res.x, fixed_bg)

        # Convert back to physical parameters
        alpha_fit = 10.0 ** res.x[0]
        Rmax_fit = 10.0 ** res.x[1]
        k_sat_fit = 10.0 ** res.x[2]

        history.append({
            "stage": "refine",
            "params": {
                "H0": fixed_bg["H0"],
                "Om0": fixed_bg["Om0"],
                "Or0": fixed_bg["Or0"],
                "Ok0": fixed_bg["Ok0"],
                "alpha": alpha_fit,
                "Rmax": Rmax_fit,
                "k_sat": k_sat_fit,
            },
            "chi2": chi2_fit,
        })

        if verbose:
            print(f"[REFINE] α={alpha_fit:.3e} Rmax={Rmax_fit:.3e} k_sat={k_sat_fit:.3f} χ²={chi2_fit:.6f}")

        # Track the best solution
        if (best_refined is None) or (chi2_fit < best_refined[0]):
            best_refined = (chi2_fit, alpha_fit, Rmax_fit, k_sat_fit)

    if best_refined is None:
        return {
            "success": False,
            "message": "No valid PBUF solution found.",
            "best_chi2": float('inf'),
            "best_params": {},
            "history": history,
        }

    best_chi2, alpha_best, Rmax_best, k_sat_best = best_refined

    return {
        "success": True,
        "message": "PBUF optimization complete.",
        "best_chi2": float(best_chi2),
        "best_params": {
            "H0": float(fixed_bg["H0"]),
            "Om0": float(fixed_bg["Om0"]),
            "Or0": float(fixed_bg["Or0"]),
            "Ok0": float(fixed_bg["Ok0"]),
            "alpha": float(alpha_best),
            "Rmax": float(Rmax_best),
            "k_sat": float(k_sat_best),
        },
        "history": history,
    }
