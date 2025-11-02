"""
Comprehensive CMB parameter optimizer.

This module provides unified optimization for both LCDM and PBUF models,
optimizing ALL parameters together (not just model-specific ones).
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.fits.cmb.chi2 import chi_squared_cmb


def _build_lcdm_model_full(H0, Om0, Or0, Ok0):
    """
    Build LCDM model from all physical parameters (including Or0).

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


def _build_pbuf_model_full(H0, Om0, Or0, Ok0, alpha, Rmax, k_sat,
                           eps0=0.7, n_alpha=0.0, n_eps=0.0, n_R=0.0):
    """
    Build PBUF model from all physical parameters.

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
        Saturation scale for elastic response
    k_sat : float
        Rigidity fraction (must be > 0)
    eps0 : float
        Baseline elastic stiffness value
    n_alpha, n_eps, n_R : float
        Power-law evolution indices for α(z), ε(z), and Rmax(z)

    Returns
    -------
    PBUF
        PBUF model instance
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


def _lcdm_objective_full(x):
    """
    Objective function for full LCDM optimization.

    Parameters
    ----------
    x : array-like
        [H0, Om0, Or0, Ok0] parameters

    Returns
    -------
    float
        χ² value (or large penalty for unphysical parameters)
    """
    H0, Om0, Or0, Ok0 = x
    Ol0 = 1.0 - (Om0 + Or0 + Ok0)

    # Physical constraints
    if Ol0 < 0:
        return 1e30  # Negative Omega_Lambda
    if Om0 <= 0 or Om0 > 1:
        return 1e30  # Unphysical matter density
    if Or0 <= 0 or Or0 > 0.1:
        return 1e30  # Unphysical radiation density
    if abs(Ok0) > 0.1:
        return 1e30  # Excessive curvature

    try:
        model = _build_lcdm_model_full(H0, Om0, Or0, Ok0)
        return chi_squared_cmb(model)
    except Exception:
        return 1e30


def _apply_physics_priors_full(H0, Om0, Or0, Ok0, alpha, Rmax, k_sat):
    """
    Apply physical plausibility priors for comprehensive optimization.
    Similar to PBUF priors but adapted for full parameter optimization.
    """
    penalty = 0.0

    # --- 1. Numerical sanity (hard guards) ---
    if (not np.isfinite(H0)) or (not np.isfinite(Om0)) or (not np.isfinite(Or0)) or \
       (not np.isfinite(Ok0)) or (not np.isfinite(alpha)) or (not np.isfinite(Rmax)) or \
       (not np.isfinite(k_sat)):
        return 1e30
    if H0 <= 0 or Om0 <= 0 or Or0 <= 0 or alpha < 0 or Rmax <= 0 or k_sat <= 0:
        return 1e30

    # --- 2. Physical plausibility (soft priors) ---

    # Hubble constant: Planck 2018 constraints ~67 ± 5
    if not (60 <= H0 <= 80):
        penalty += ((H0 - 67) / 5) ** 2 * 1e2

    # Matter density: Planck 2018 constraints ~0.3 ± 0.05
    if not (0.25 <= Om0 <= 0.35):
        penalty += ((Om0 - 0.3) / 0.05) ** 2 * 1e2

    # Curvature: must be nearly flat |Ωk| < 0.005
    if abs(Ok0) > 0.005:
        penalty += ((abs(Ok0) - 0.005) / 0.005) ** 2 * 1e3

    # Radiation density: should be fixed to ~9e-5
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
    penalty += ((k_sat - 1.0) ** 2) * 1e-2

    return penalty


def _pbuf_objective_full(x):
    """
    Objective function for full PBUF optimization.

    Parameters
    ----------
    x : array-like
        [H0, Om0, Or0, Ok0, alpha, Rmax, k_sat] parameters

    Returns
    -------
    float
        χ² value (or large penalty for unphysical parameters)
    """
    H0, Om0, Or0, Ok0, alpha, Rmax, k_sat = x

    # Apply PBUF-specific physical priors (includes both hard and soft constraints)
    penalty = _apply_physics_priors_full(H0, Om0, Or0, Ok0, alpha, Rmax, k_sat)
    if penalty >= 1e20:
        return penalty  # skip unphysical region

    try:
        model = _build_pbuf_model_full(H0, Om0, Or0, Ok0, alpha, Rmax, k_sat)
        chi2 = chi_squared_cmb(model)
    except Exception:
        return 1e30  # Penalty for model construction failures

    return chi2 + penalty


def _lcdm_objective_full(x):
    """
    Objective function for full LCDM optimization.

    Parameters
    ----------
    x : array-like
        [H0, Om0, Or0, Ok0] parameters

    Returns
    -------
    float
        χ² value (or large penalty for unphysical parameters)
    """
    H0, Om0, Or0, Ok0 = x
    Ol0 = 1.0 - (Om0 + Or0 + Ok0)

    # Apply LCDM-specific physical priors (includes both hard and soft constraints)
    penalty = _apply_lcdm_priors(H0, Om0, Or0, Ok0)
    if penalty >= 1e20:
        return penalty  # skip unphysical region

    try:
        model = _build_lcdm_model_full(H0, Om0, Or0, Ok0)
        chi2 = chi_squared_cmb(model)
    except Exception:
        return 1e30  # Penalty for model construction failures

    return chi2 + penalty


def _apply_lcdm_priors(H0, Om0, Or0, Ok0):
    """
    Apply physical plausibility priors for LCDM optimization.
    """
    penalty = 0.0

    # --- 1. Numerical sanity (hard guards) ---
    if (not np.isfinite(H0)) or (not np.isfinite(Om0)) or (not np.isfinite(Or0)) or (not np.isfinite(Ok0)):
        return 1e30
    if H0 <= 0 or Om0 <= 0 or Or0 <= 0:
        return 1e30

    # --- 2. Physical plausibility (soft priors) ---

    # Hubble constant: Planck 2018 constraints ~67 ± 5
    if not (60 <= H0 <= 80):
        penalty += ((H0 - 67) / 5) ** 2 * 1e2

    # Matter density: Planck 2018 constraints ~0.3 ± 0.05
    if not (0.25 <= Om0 <= 0.35):
        penalty += ((Om0 - 0.3) / 0.05) ** 2 * 1e2

    # Curvature: must be nearly flat |Ωk| < 0.005
    if abs(Ok0) > 0.005:
        penalty += ((abs(Ok0) - 0.005) / 0.005) ** 2 * 1e3

    # Radiation density: should be fixed to ~9e-5
    if abs(Or0 - 9.2e-5) / 9.2e-5 > 0.2:  # 20% deviation allowed
        penalty += ((Or0 - 9.2e-5) / 9.2e-5) ** 2 * 1e3

    return penalty


def optimise_lcdm_full(verbose: bool = True) -> dict:
    """
    Optimize ALL LCDM parameters against CMB distance priors.

    Optimizes: H0, Om0, Or0, Ok0 (with Ol0 derived from closure)

    Parameters
    ----------
    verbose : bool, optional
        Print progress information

    Returns
    -------
    dict
        Optimization results
    """
    print("🔬 Full LCDM optimization: optimizing (H0, Om0, Or0, Ok0)")

    # Initial guess from Planck 2018
    x0 = np.array([67.36, 0.3153, 9.2e-5, 0.0])

    # Bounds for all parameters
    bounds = [
        (60.0, 80.0),    # H0 [km/s/Mpc]
        (0.1, 0.6),      # Om0
        (1e-5, 1e-3),    # Or0
        (-0.1, 0.1),     # Ok0
    ]

    res = minimize(
        _lcdm_objective_full,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1000}
    )

    if not res.success:
        return {
            "success": False,
            "message": f"Optimization failed: {res.message}",
            "best_chi2": float(res.fun),
            "best_params": {},
        }

    H0_fit, Om0_fit, Or0_fit, Ok0_fit = res.x
    Ol0_fit = 1.0 - (Om0_fit + Or0_fit + Ok0_fit)

    if verbose:
        print(f"✅ Full LCDM fit: χ²={res.fun:.6f}")
        print(f"   H0={H0_fit:.2f}, Om0={Om0_fit:.4f}, Or0={Or0_fit:.2e}, Ok0={Ok0_fit:+.4f}, Ol0={Ol0_fit:.4f}")

    return {
        "success": True,
        "message": "Full LCDM optimization complete.",
        "best_chi2": float(res.fun),
        "best_params": {
            "H0": float(H0_fit),
            "Om0": float(Om0_fit),
            "Or0": float(Or0_fit),
            "Ok0": float(Ok0_fit),
            "Ol0": float(Ol0_fit),
        },
    }


def optimise_pbuf_full(verbose: bool = True) -> dict:
    """
    Optimize ALL PBUF parameters against CMB distance priors.

    Optimizes: H0, Om0, Or0, Ok0, alpha, Rmax, k_sat

    Parameters
    ----------
    verbose : bool, optional
        Print progress information

    Returns
    -------
    dict
        Optimization results
    """
    print("🔬 Full PBUF optimization: optimizing (H0, Om0, Or0, Ok0, alpha, Rmax, k_sat)")

    # Initial guess: Planck LCDM background + small elastic terms
    x0 = np.array([
        67.36,    # H0
        0.3153,   # Om0
        9.2e-5,   # Or0
        0.0,      # Ok0
        0.001,    # alpha (1e-3)
        1e9,      # Rmax (1e9)
        1.5       # k_sat
    ])

    # Bounds for all parameters
    bounds = [
        (60.0, 80.0),      # H0 [km/s/Mpc]
        (0.1, 0.6),        # Om0
        (1e-5, 1e-3),      # Or0
        (-0.1, 0.1),       # Ok0
        (1e-6, 1e-1),      # alpha (1e-6 to 1e-1)
        (1e6, 1e12),       # Rmax (1e6 to 1e12)
        (0.5, 3.0),        # k_sat: 0.5 to 3.0
    ]

    res = minimize(
        _pbuf_objective_full,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1000}
    )

    if not res.success:
        return {
            "success": False,
            "message": f"Optimization failed: {res.message}",
            "best_chi2": float(res.fun),
            "best_params": {},
        }

    H0_fit, Om0_fit, Or0_fit, Ok0_fit, alpha_fit, Rmax_fit, k_sat_fit = res.x

    if verbose:
        print(f"✅ Full PBUF fit: χ²={res.fun:.6f}")
        print(f"   H0={H0_fit:.2f}, Om0={Om0_fit:.4f}, Or0={Or0_fit:.2e}, Ok0={Ok0_fit:+.4f}")
        print(f"   alpha={alpha_fit:.3e}, Rmax={Rmax_fit:.3e}, k_sat={k_sat_fit:.3f}")

    return {
        "success": True,
        "message": "Full PBUF optimization complete.",
        "best_chi2": float(res.fun),
        "best_params": {
            "H0": float(H0_fit),
            "Om0": float(Om0_fit),
            "Or0": float(Or0_fit),
            "Ok0": float(Ok0_fit),
            "alpha": float(alpha_fit),
            "Rmax": float(Rmax_fit),
            "k_sat": float(k_sat_fit),
        },
    }


def compare_optimization_approaches():
    """
    Compare different optimization approaches using the new joint fitting system.
    """
    print("Comparing comprehensive optimization approaches with physics validation...")
    print("=" * 80)

    # 1. LCDM joint optimization
    print("\n1. LCDM (joint fit across all datasets):")
    try:
        from cosmos.fits.joint.optimizer import fit_joint_with_physics_validation
        result_lcdm_joint = fit_joint_with_physics_validation(
            model_type='lcdm',
            datasets=['cmb', 'sn', 'bao_iso', 'bao_aniso', 'cc', 'rsd'],
            verbose=False
        )
        if result_lcdm_joint["success"]:
            joint_result = result_lcdm_joint["joint_fit"]
            print(f"   χ² = {joint_result['chi2_total']:.6f}")
            print(f"   H0 = {joint_result['params']['H0']:.2f}")
            print(f"   Om0 = {joint_result['params']['Om0']:.4f}")
            print(f"   Validation: {result_lcdm_joint['validation']['summary']['passed_checks']}/{result_lcdm_joint['validation']['summary']['total_checks']} ✅")
        else:
            print(f"   FAILED validation")
    except Exception as e:
        print(f"   Error: {e}")

    # 2. PBUF joint optimization
    print("\n2. PBUF (joint fit across all datasets):")
    try:
        result_pbuf_joint = fit_joint_with_physics_validation(
            model_type='pbuf',
            datasets=['cmb', 'sn', 'bao_iso', 'bao_aniso', 'cc', 'rsd'],
            verbose=False
        )
        if result_pbuf_joint["success"]:
            joint_result = result_pbuf_joint["joint_fit"]
            print(f"   χ² = {joint_result['chi2_total']:.6f}")
            print(f"   H0 = {joint_result['params']['H0']:.2f}")
            print(f"   Om0 = {joint_result['params']['Om0']:.4f}")
            print(f"   alpha = {joint_result['params']['alpha']:.3e}")
            print(f"   Rmax = {joint_result['params']['Rmax']:.3e}")
            print(f"   k_sat = {joint_result['params']['k_sat']:.3f}")
            print(f"   Validation: {result_pbuf_joint['validation']['summary']['passed_checks']}/{result_pbuf_joint['validation']['summary']['total_checks']} ✅")
        else:
            print(f"   FAILED validation")
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Legacy CMB-only optimization (for comparison)
    print("\n3. CMB-only optimization (legacy approach):")
    try:
        from cosmos.pbuf.optimizer import optimise_against_cmb as optimise_pbuf_cmb
        from cosmos.lcdm.optimizer import optimise_against_cmb as optimise_lcdm_cmb

        result_pbuf_cmb = optimise_pbuf_cmb(verbose=False)
        if result_pbuf_cmb["success"]:
            print(f"   PBUF χ² = {result_pbuf_cmb['best_chi2']:.6f}")
            print(f"   PBUF α = {result_pbuf_cmb['best_params']['alpha']:.3e}")
            print(f"   PBUF Rmax = {result_pbuf_cmb['best_params']['Rmax']:.3e}")
            print(f"   PBUF k_sat = {result_pbuf_cmb['best_params']['k_sat']:.3f}")

        result_lcdm_cmb = optimise_lcdm_cmb(verbose=False)
        if result_lcdm_cmb["success"]:
            print(f"   LCDM χ² = {result_lcdm_cmb['best_chi2']:.6f}")
            print(f"   LCDM H0 = {result_lcdm_cmb['best_params']['H0']:.2f}")
            print(f"   LCDM Om0 = {result_lcdm_cmb['best_params']['Om0']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 80)
    print("Summary: Joint optimization provides comprehensive multi-dataset fitting")
    print("   with physics validation to ensure cosmologically meaningful results!")


if __name__ == "__main__":
    compare_optimization_approaches()
