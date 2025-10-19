#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance prior utilities for CMB calibration of the PBUF model.

The functions in this module evaluate background-only observables such
as the acoustic scale and shift parameter without invoking a full
Boltzmann solver. They rely on the shared model API exposed by
``core.gr_models`` and ``core.pbuf_models``.
"""

from __future__ import annotations
from typing import Dict, Mapping, Sequence
import numpy as np

from config.constants import C_LIGHT, KM_TO_M, NEFF, TCMB, omega_gamma_h2
from core import gr_models, pbuf_models

KM_PER_S = C_LIGHT / KM_TO_M
_PREPARED_KEY = "_cmb_background_prepared"

# ---------------------------------------------------------------------
#  Parameter preparation
# ---------------------------------------------------------------------
def prepare_background_params(params: Mapping[str, float]) -> Dict[str, float]:
    """Enrich parameter mapping with derived densities needed for CMB distances."""
    if isinstance(params, dict) and params.get(_PREPARED_KEY):
        return params  # already prepared

    work: Dict[str, float] = dict(params)
    if "Obh2" not in work:
        raise KeyError("Parameter 'Obh2' (Ω_b h^2) is required for CMB priors")

    hubble0 = float(work.get("H0", 70.0))
    if hubble0 <= 0.0:
        raise ValueError("H0 must be positive")
    h = hubble0 / 100.0
    work["H0"] = hubble0
    work["h"] = h
    work["Obh2"] = float(work["Obh2"])
    work["Ob0"] = work["Obh2"] / h**2

    tcmb = float(work.get("Tcmb", TCMB))
    neff = float(work.get("Neff", NEFF))
    omega_gamma_h2_today = omega_gamma_h2(tcmb)
    work["Omega_gamma_h2"] = omega_gamma_h2_today
    work["Omega_gamma"] = omega_gamma_h2_today / h**2
    work["Or0"] = work["Omega_gamma"] * (1.0 + 0.2271 * neff)

    work.setdefault("Om0", 0.3)
    work["Omh2"] = work["Om0"] * h**2
    ok0 = float(work.get("Ok0", 0.0))
    work["Ok0"] = ok0
    work.setdefault("Ol0", 1.0 - work["Om0"] - work["Or0"] - ok0)
    if "ns" in params:
        work["ns"] = float(params["ns"])
    work["recomb_method"] = str(params.get("recomb_method", "PLANCK18"))

    work[_PREPARED_KEY] = True
    return work

# ---------------------------------------------------------------------
#  Core background helpers
# ---------------------------------------------------------------------
def z_recombination(Obh2: float, Omh2: float, method: str = "PLANCK18") -> float:
    """
    Return the recombination redshift z* using one of several approximations:
      - "HS96": Hu & Sugiyama 1996 legacy fit (Planck Collaboration 2015, App. C)
      - "EH98": Eisenstein & Hu 1998 drag-epoch proxy (1.04 × z_d)
      - "PLANCK18": Planck 2018-calibrated fit (default)
    """
    method = method.upper()
    ob, om = float(Obh2), float(Omh2)

    if method == "HS96":
        g1 = 0.0783 * ob**-0.238 / (1.0 + 39.5 * ob**0.763)
        g2 = 0.560 / (1.0 + 21.1 * ob**1.81)
        return 1048.0 * (1.0 + 0.00124 * ob**-0.738) * (1.0 + g1 * om**g2)

    elif method == "EH98":
        b1 = 0.313 * om**-0.419 * (1 + 0.607 * om**0.674)
        b2 = 0.238 * om**0.223
        z_d = 1291 * om**0.251 / (1 + 0.659 * om**0.828) * (1 + b1 * ob**b2)
        return 1.04 * z_d  # good proxy for z_*

    elif method == "PLANCK18":
        # Empirical fit around Planck 2018 base cosmology
        z_star = 1089.80 * (ob / 0.02237)**(-0.004) * (om / 0.1424)**(0.010)
        return z_star

    else:
        raise ValueError(f"Unsupported recombination method '{method}'")


def z_drag(Obh2: float, Omh2: float) -> float:
    """Eisenstein & Hu (1998) baryon drag epoch z_d."""

    ob, om = float(Obh2), float(Omh2)
    b1 = 0.313 * om**-0.419 * (1 + 0.607 * om**0.674)
    b2 = 0.238 * om**0.223
    return 1291.0 * om**0.251 / (1 + 0.659 * om**0.828) * (1 + b1 * ob**b2)


def _sound_speed(z: np.ndarray, work: Mapping[str, float]) -> np.ndarray:
    rb_prefactor = 3.0 * work["Ob0"] / (4.0 * work["Omega_gamma"])
    return KM_PER_S / np.sqrt(3.0 * (1.0 + rb_prefactor / (1.0 + z)))


def _select_model(model):
    """Return the chosen model or default to PBUF for flexibility."""
    return model if model is not None else pbuf_models


def sound_horizon_a(z: float, params: Mapping[str, float], model=None, steps: int = 200_000) -> float:
    """
    Comoving sound horizon r_s(z) in Mpc using scale factor integration:

        r_s(z) = ∫_{0}^{a_*} c_s(a) / [a^2 H(a)] da ,  a_* = 1/(1+z).

    Uses a uniform grid in ln(a) for high-accuracy radiation-era convergence.
    """
    work = prepare_background_params(params)
    model = _select_model(model)

    # Define limits
    a_star = 1.0 / (1.0 + float(z))  # scale factor at recombination
    a_min = 1.0e-10                  # integrate deep into radiation era
    if a_star <= a_min:
        a_min = max(a_star * 1e-4, 1.0e-12)

    # Logarithmic grid in a
    u = np.linspace(np.log(a_min), np.log(a_star), steps)
    a = np.exp(u)
    da_du = a  # derivative da/du = a

    # Compute baryon-photon ratio R_b(a)
    rb_prefactor = 3.0 * work["Ob0"] / (4.0 * work["Omega_gamma"])
    Rb = rb_prefactor * a

    # Sound speed c_s(a)
    cs = KM_PER_S / np.sqrt(3.0 * (1.0 + Rb))  # km/s

    # Evaluate H(a) at z = 1/a - 1
    z_eval = (1.0 / a) - 1.0
    Ha = model.H(z_eval, work)  # km/s/Mpc

    # Integrand: c_s / (a^2 H(a)) * da
    integrand = (cs / (a * a) / Ha) * da_du  # Mpc

    trap = getattr(np, "trapezoid", np.trapz)
    rs = float(trap(integrand, u))  # Mpc
    return rs




def _frw_H_kms_per_Mpc(z: np.ndarray, work: Mapping[str, float]) -> np.ndarray:
    """Canonical flat-FRW H(z) using the prepared Ω’s (no hidden re-computation)."""
    H0 = float(work["H0"])
    Om = float(work["Om0"])
    Or = float(work["Or0"])
    Ok = float(work.get("Ok0", 0.0))
    Ol = float(work.get("Ol0", 1.0 - Om - Or - Ok))
    zp1 = 1.0 + z
    E2 = Om * zp1**3 + Or * zp1**4 + Ok * zp1**2 + Ol
    return H0 * np.sqrt(E2)


def _comoving_transverse_distance(z: float, params: Mapping[str, float], model=None, steps: int = 80_000) -> float:
    """Transverse comoving distance D_M(z) in Mpc using a uniform grid in ln(1+z)."""
    work = prepare_background_params(params)
    model = _select_model(model)
    if z <= 0.0:
        return 0.0
    u = np.linspace(0.0, np.log1p(z), steps)
    z_eval = np.expm1(u)
    dz_du = np.exp(u)                            # dz/du
    hz = model.H(z_eval, work)                   # km/s/Mpc
    integrand = (KM_PER_S / hz) * dz_du          # Mpc
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(integrand, u))             # Mpc

def D_A(z: float, params: Mapping[str, float], model=None, steps: int = 80_000) -> float:
    """Angular diameter distance via D_A(z) = D_M(z)/(1+z)."""
    dm = _comoving_transverse_distance(z, params, model=model, steps=steps)
    return float(dm / (1.0 + z))



# ---------------------------------------------------------------------
#  Observables
# ---------------------------------------------------------------------
def theta_star(params: Mapping[str, float], model=None) -> float:
    """Acoustic angular scale θ* = r_s(z*) / D_M(z*) with D_M=(1+z*)D_A."""
    work = prepare_background_params(params)
    method = str(work.get("recomb_method", "PLANCK18")).upper()
    z_star = work.get("z_star") or z_recombination(work["Obh2"], work["Omh2"], method=method)

    rs = sound_horizon_a(z_star, work, model=model)
    da = D_A(z_star, work, model=model)
    dm = da * (1.0 + z_star)
    theta = rs / dm
    # Diagnostics
    work.update({
        "z_star": z_star,
        "r_s_zstar": rs,
        "D_A_zstar": da,
        "D_M_zstar": dm,
        "theta_star": theta
    })
    return theta


def l_A(params: Mapping[str, float], model=None) -> float:
    """The acoustic scale l_A = π D_M(z*) / r_s(z*)."""
    work = prepare_background_params(params)
    method = str(work.get("recomb_method", "PLANCK18")).upper()
    z_star = work.get("z_star") or z_recombination(work["Obh2"], work["Omh2"], method=method)

    rs = sound_horizon_a(z_star, work, model=model)
    da = D_A(z_star, work, model=model)
    dm = da * (1.0 + z_star)
    return float(np.pi * dm / rs)


def shift_parameter(params: Mapping[str, float], model=None,
                    *, angular_diameter: float | None = None,
                    z_star: float | None = None) -> float:
    """CMB shift parameter R = sqrt(Ω_m)(H0/c)(1+z*)D_A(z*)."""
    work = prepare_background_params(params)
    method = str(work.get("recomb_method", "PLANCK18")).upper()
    z_use = z_star or work.get("z_star") or z_recombination(work["Obh2"], work["Omh2"], method=method)
    da = angular_diameter if angular_diameter is not None else work.get("D_A_zstar")
    if da is None:
        da = D_A(z_use, work, model=model)
    dm = float(da) * (1.0 + z_use)
    value = np.sqrt(work["Om0"]) * (work["H0"] / KM_PER_S) * dm
    return float(value)


def distance_priors(params: Mapping[str, float], model=None) -> Dict[str, float]:
    """
    Return CMB distance-prior predictions: R, lA, Ω_bh², n_s, and 100θ* (diagnostic).

    Notes
    -----
    - D_M(z*) is the transverse comoving distance, D_M = (1 + z*) · D_A(z*).
    - l_A = π · D_M(z*) / r_s(z*).
    - θ* = r_s(z*) / D_M(z*).
    - All distances are in Mpc; H(z) in km/s/Mpc.
    """
    # ------------------------------------------------------------------
    # 1. Prepare parameters and recombination redshift
    # ------------------------------------------------------------------
    work = prepare_background_params(params)
    method = str(work.get("recomb_method", "PLANCK18")).upper()
    z_star = work.get("z_star") or z_recombination(work["Obh2"], work["Omh2"], method=method)

    # ------------------------------------------------------------------
    # 2. Cross-check H(z) consistency (to catch radiation/unit issues)
    # ------------------------------------------------------------------
    z_chk = np.array([10.0, 100.0, z_star], dtype=float)
    H_model = _select_model(model).H(z_chk, work)       # what the chosen model returns
    H_frw = _frw_H_kms_per_Mpc(z_chk, work)             # canonical FRW from prepared Ω's
    ratio = H_model / H_frw
    print(
        f"[DBG H] z={z_chk.tolist()}  "
        f"H_model={H_model.tolist()}  "
        f"H_FRW={H_frw.tolist()}  ratio={ratio.tolist()}"
    )

    # ------------------------------------------------------------------
    # 3. Compute sound horizon and angular distances
    # ------------------------------------------------------------------
    rs = sound_horizon_a(z_star, work, model=model)     # high-accuracy a-space integral (Mpc)
    da = D_A(z_star, work, model=model)                 # Mpc
    dm = da * (1.0 + z_star)                            # transverse comoving distance (Mpc)

    # ------------------------------------------------------------------
    # 4. Acoustic quantities
    # ------------------------------------------------------------------
    theta = rs / dm                                     # dimensionless
    l_a = float(np.pi * dm / rs)                        # dimensionless

    # ------------------------------------------------------------------
    # 5. Diagnostics and reporting
    # ------------------------------------------------------------------
    baryon_physical = work["Obh2"]
    ns_value = float(work.get("ns", params.get("ns", 0.965)))

    print(
        f"[DBG] z*={z_star:.1f}  r_s={rs:.3f} Mpc  "
        f"D_M={dm:.3f} Mpc  D_A={da:.3f} Mpc  "
        f"100θ*={100.0 * theta:.4f}  l_A={l_a:.3f}"
    )

    # ------------------------------------------------------------------
    # 6. Return priors dictionary
    # ------------------------------------------------------------------
    priors = {
        "100theta_*": 100.0 * theta,  # diagnostic only
        "lA": l_a,
        "R": shift_parameter(work, model=model, angular_diameter=da, z_star=z_star),
        "Omegabh2": baryon_physical,
        "ns": ns_value,
        "z_star": z_star,
    }
    return priors



# ---------------------------------------------------------------------
#  χ² evaluation
# ---------------------------------------------------------------------
def chi2_cmb(pred: Mapping[str, float], priors: Mapping[str, object]) -> float:
    """
    Compute χ² between predicted and observed CMB distance priors.

    The labels order in priors["labels"] defines vector alignment.
    """
    labels: Sequence[str] = priors["labels"]  # e.g. ["R","lA","Omegabh2","ns"]
    means: Mapping[str, float] = priors["means"]
    cov = np.asarray(priors["cov"], dtype=float)
    cov = 0.5 * (cov + cov.T)
    delta = np.array([pred[label] - means[label] for label in labels], dtype=float)

    # Stability + positive definiteness
    eps = 1e-12 * np.eye(cov.shape[0])
    try:
        chol = np.linalg.cholesky(cov + eps)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-12, None)
        cov = (eigvecs * eigvals) @ eigvecs.T
        chol = np.linalg.cholesky(cov)

    y = np.linalg.solve(chol, delta)
    chi2 = float(y @ y)

    # Debug sanity print (temporary)
    dbg = " | ".join(f"{lab} Δ={delta[i]:+.3e}" for i, lab in enumerate(labels))
    print(f"[CMB χ² debug] χ²={chi2:.3f}  ({dbg})")
    print("[CMB Δ vector]", {lab: float(delta[i]) for i, lab in enumerate(labels)})
    print("[CMB covariance diag]", np.diag(cov).tolist())
    print("[CMB χ²]", chi2)

    return chi2
