"""Utility functions for per-event stiffness (E0) constraints."""

from __future__ import annotations

from typing import Callable

from .physics import c, MPC_TO_M


def fractional_speed_offset(
    dt_obs: float,
    dt_int_max: float,
    distance_mpc: float,
) -> float:
    """
    Compute ε_max = (|Δt_obs| + Δt_int_max) / (D / c).

    Parameters
    ----------
    dt_obs : float
        Observed arrival-time difference [s].
    dt_int_max : float
        Conservative bound on intrinsic emission lag [s] (must be non-negative).
    distance_mpc : float
        Luminosity distance to the source [Mpc]; must be positive.

    Returns
    -------
    float
        Fractional speed difference upper limit (dimensionless).
    """
    if distance_mpc <= 0.0:
        raise ValueError("distance_mpc must be positive")
    travel_time = (float(distance_mpc) * MPC_TO_M) / c
    if travel_time <= 0.0:
        raise ValueError("Effective travel time must be positive")
    lag_term = max(float(dt_int_max), 0.0)
    eps_max = (abs(float(dt_obs)) + lag_term) / travel_time
    return eps_max


def coupling_mass_energy(E: float, m: float, eta_m: float = 1.0) -> float:
    """A_k = η_m * (m^2 / E)."""
    if E <= 0.0:
        raise ValueError("Energy must be positive for coupling_mass_energy")
    return float(eta_m) * (float(m) ** 2 / float(E))


def coupling_energy(E: float, eta_E: float = 1.0) -> float:
    """A_k = η_E * E."""
    if E <= 0.0:
        raise ValueError("Energy must be positive for coupling_energy")
    return float(eta_E) * float(E)


def coupling_general(E: float, E_ref: float = 1.0, n: float = 1.0, eta: float = 1.0, **_: float) -> float:
    """A_k = η * (E / E_ref)^n."""
    if E <= 0.0:
        raise ValueError("Energy must be positive for coupling_general")
    if E_ref <= 0.0:
        raise ValueError("E_ref must be positive")
    return float(eta) * (float(E) / float(E_ref)) ** float(n)


def e0_lower_bound(A_i: float, A_j: float, eps_max: float) -> float:
    """
    Compute minimum stiffness scale:
        E0_min = |A_i - A_j| / ε_max
    """
    if eps_max <= 0.0:
        raise ValueError("eps_max must be positive")
    return abs(float(A_i) - float(A_j)) / float(eps_max)


def compute_E0_event(
    E_i: float,
    m_i: float,
    E_j: float,
    m_j: float,
    dt_obs: float,
    dt_int_max: float,
    distance_mpc: float,
    coupling: Callable[..., float] = coupling_general,
    **coupling_kwargs: float,
) -> float:
    """
    Full per-event stiffness constraint. Returns E0_min.
    """
    eps_max = fractional_speed_offset(dt_obs, dt_int_max, distance_mpc)
    A_i = coupling(E_i, m=m_i, **coupling_kwargs)
    A_j = coupling(E_j, m=m_j, **coupling_kwargs)
    return e0_lower_bound(A_i, A_j, eps_max)


__all__ = [
    "fractional_speed_offset",
    "coupling_mass_energy",
    "coupling_energy",
    "coupling_general",
    "e0_lower_bound",
    "compute_E0_event",
]
