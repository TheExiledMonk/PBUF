"""
ΛCDM cosmological model (no elasticity, no modifications).

This is the baseline model used for comparison and benchmarking.
"""

import numpy as np
from .equations import (
    H_lcdm_a,
    H_lcdm_z,
)
from .validators import (
    validate_lcdm_parameters,
    validate_scale_and_rate,
)
from ..helper.constants import T_CMB0
from ..helper.units import MPC_TO_M, RHO_CRIT_PREFAC
from ..helper.guards import check_scale_factor

def infer_omega_r(T_cmb, h):
    """
    Infer radiation density Ω_r from CMB temperature and standard neutrino physics.

    Using standard relation:
        Ω_r = Ω_γ (1 + 0.2271 * N_eff)
        Ω_γ = 2.469e-5 / h^2 * (T_cmb / 2.725)^4
    with N_eff = 3.046
    """
    T_ratio = T_cmb / 2.725
    omega_gamma = 2.469e-5 * T_ratio**4 / (h**2)
    N_eff = 3.046
    return omega_gamma * (1.0 + 0.2271 * N_eff)


class LCDM:
    """
    Standard ΛCDM cosmology.

    Parameters
    ----------
    omega_m : float
        Matter density today.
    omega_lambda : float
        Dark energy (cosmological constant) density today.
    h : float
        Dimensionless Hubble parameter (H0 = 100 h km/s/Mpc).
    omega_k : float
        Curvature density today.
    omega_r : float or None
        Radiation density today (inferred if None).
    omega_b : float
        Baryon density today (used for sound horizon, etc.)
    T_cmb : float or None
        CMB temperature [K]. Defaults to T_CMB0.
    """

    def __init__(
        self,
        omega_m,
        omega_lambda,
        h,
        omega_k=0.0,
        omega_r=9.2e-5,
        omega_b=0.022,
        T_cmb=None,
    ):
        self.omega_m = float(omega_m)
        self.omega_lambda = float(omega_lambda)
        self.omega_k = float(omega_k)
        self.omega_b = float(omega_b)
        self.h = float(h)
        self.h0 = 100.0 * self.h
        self.T_cmb = T_cmb if T_cmb is not None else T_CMB0

        if omega_r is None:
            self.omega_r = float(infer_omega_r(self.T_cmb, self.h))
        else:
            self.omega_r = float(omega_r)

        validate_lcdm_parameters(
            self.omega_m,
            self.omega_r,
            self.omega_lambda,
            self.omega_k,
            self.h,
        )

    def __repr__(self):
        return (
            "LCDM("
            f"omega_m={self.omega_m}, "
            f"omega_lambda={self.omega_lambda}, "
            f"omega_k={self.omega_k}, "
            f"omega_r={self.omega_r}, "
            f"h={self.h}"
            ")"
        )

    # -------------------------
    # Expansion
    # -------------------------

    def H(self, z):
        """
        Hubble parameter at redshift z [km/s/Mpc].
        """
        Hz = H_lcdm_z(
            z,
            self.h0,
            self.omega_m,
            self.omega_r,
            self.omega_k,
            self.omega_lambda,
        )
        z_arr = np.asarray(z, dtype=float)
        a_arr = 1.0 / (1.0 + z_arr)
        validate_scale_and_rate(a_arr, Hz, z_arr)
        return Hz

    def hubble_function(self, a):
        """
        Hubble parameter as function of scale factor a [km/s/Mpc].
        """
        check_scale_factor(a)
        Hz = H_lcdm_a(
            a,
            self.h0,
            self.omega_m,
            self.omega_r,
            self.omega_k,
            self.omega_lambda,
        )
        validate_scale_and_rate(a, Hz)
        return Hz

    # -------------------------
    # Density diagnostics
    # -------------------------

    def density_parameters_at_z(self, z):
        """
        Return fractional density components at redshift z.
        """
        a = 1.0 / (1.0 + z)
        Om_m = self.omega_m / (a**3)
        Om_r = self.omega_r / (a**4)
        Om_k = self.omega_k / (a**2)
        Om_l = self.omega_lambda
        total = Om_m + Om_r + Om_k + Om_l
        if np.any(total <= 0.0):
            raise ValueError(f"Unphysical total density at z={z}: {total}")

        return {
            "omega_m": Om_m / total,
            "omega_r": Om_r / total,
            "omega_k": Om_k / total,
            "omega_lambda": Om_l / total,
            "omega_total": 1.0,
        }

    def closure_today(self):
        """
        Return Ω_total(a=1).
        """
        return self.omega_m + self.omega_r + self.omega_k + self.omega_lambda

    def critical_density_today(self):
        """
        Critical density ρ_crit0 in kg/m^3.
        """
        H0_SI = (self.h0 * 1000.0) / MPC_TO_M
        return RHO_CRIT_PREFAC * (H0_SI**2)

    def parameters(self):
        """
        Return dict of model parameters.
        """
        return {
            "omega_m": self.omega_m,
            "omega_r": self.omega_r,
            "omega_k": self.omega_k,
            "omega_lambda": self.omega_lambda,
            "omega_b": self.omega_b,
            "h": self.h,
        }
