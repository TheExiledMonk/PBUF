"""
PBUF background cosmology model with curvature-based elastic sector.

This is the class interface used by higher-level code such as
cmb_observables(), distances, etc. It intentionally does NOT
inherit from LCDM. It represents a different physical model.

In this model, late-time acceleration is sourced by an elastic
spacetime sector whose effective contribution follows the restored
σ_eff · S + Δ_rad formulation from the PBUF v9 implementation.
"""

import numpy as np
from .equations import (
    omega_sigma_raw,
    omega_sigma_total,
    H_pbuf_a,
    H_pbuf_z,
    elastic_fraction,
)
from .validators import (
    validate_pbuf_parameters,
    validate_scale_and_rate,
)
from ..helper.constants import T_CMB0
from ..helper.guards import check_scale_factor, infer_omega_r

class PBUF:
    """
    Planck-Bound Unified Framework cosmology.

    Parameters
    ----------
    omega_m : float
        Matter density today (baryons + CDM)
    h : float
        Dimensionless Hubble parameter (H0 = 100 h km/s/Mpc)
    alpha : float
        Elastic amplitude parameter today
    Rmax : float
        Saturation scale factor for the elastic response today
    eps0 : float
        Baseline elastic stiffness parameter (dimensionless)
    k_sat : float
        Rigidity saturation control (k_sat > 0, values >1 allowed)
    n_alpha, n_eps, n_R : float
        Power-law evolution indices for α(z), ε(z), and Rmax(z)
    omega_k : float
        Curvature density today
    omega_r : float or None
        Radiation density today. If None, inferred from T_cmb.
    omega_b : float
        Baryon density today (used for sound speed / r_s, etc.)
    T_cmb : float or None
        CMB temperature in Kelvin. Defaults to T_CMB0.

    Physics
    -------
    H^2(a) = H0^2 [ Ω_m a^-3 + Ω_r a^-4 + Ω_k a^-2 + Ω_σ(a) ]
    Ω_σ(a) = k_sat · σ_eff(z) · S(z; k_sat) + Δ_rad(z; k_sat)

    No cosmological constant term is present.
    """

    def __init__(self,
                 omega_m,
                 h,
                 alpha,
                 Rmax,
                 k_sat=1.0,
                 eps0=0.7,
                 n_alpha=0.0,
                 n_eps=0.0,
                 n_R=0.0,
                 omega_k=0.0,
                 omega_r=9.2e-5,
                 omega_b=0.022,
                 T_cmb=None):
        self.omega_m = float(omega_m)
        self.h = float(h)
        self.h0 = 100.0 * self.h  # km/s/Mpc
        
        self.omega_k = float(omega_k)
        self.omega_b = float(omega_b)

        self.T_cmb = T_cmb if T_cmb is not None else T_CMB0
        if omega_r is None:
            # infer radiation density from T_cmb and h
            inferred = infer_omega_r(self.T_cmb, self.h)
            self.omega_r = float(inferred)
        else:
            self.omega_r = float(omega_r)

        # Elastic sector
        self.alpha = float(alpha)
        self.Rmax = float(Rmax)
        self.k_sat = float(k_sat)
        self.eps0 = float(eps0)
        self.n_alpha = float(n_alpha)
        self.n_eps = float(n_eps)
        self.n_R = float(n_R)

        # Validate physicality
        validate_pbuf_parameters(
            self.omega_m,
            self.omega_k,
            self.omega_r,
            self.alpha,
            self.Rmax,
            self.eps0,
            self.k_sat,
            self.n_alpha,
            self.n_eps,
            self.n_R,
            self.h
        )

    def __repr__(self):
        return (
            "PBUF("
            f"omega_m={self.omega_m}, "
            f"h={self.h}, "
            f"alpha={self.alpha}, "
            f"Rmax={self.Rmax}, "
            f"k_sat={self.k_sat}, "
            f"eps0={self.eps0}, "
            f"n_alpha={self.n_alpha}, "
            f"n_eps={self.n_eps}, "
            f"n_R={self.n_R}, "
            f"omega_k={self.omega_k}, "
            f"omega_r={self.omega_r}, "
            f"omega_b={self.omega_b}"
            ")"
        )

    @property
    def params(self):
        """
        Return a parameter dictionary compatible with the Path A helpers.
        """
        return {
            "H0": self.h0,
            "Om0": self.omega_m,
            "Or0": self.omega_r,
            "Ok0": self.omega_k,
            "alpha": self.alpha,
            "Rmax": self.Rmax,
            "k_sat": self.k_sat,
            "eps0": self.eps0,
            "n_alpha": self.n_alpha,
            "n_eps": self.n_eps,
            "n_R": self.n_R,
        }

    # -------------------------
    # Expansion history
    # -------------------------

    def H(self, z):
        """
        Hubble parameter at redshift z [km/s/Mpc].

        Uses the pure PBUF Friedmann equation (no Λ).
        """
        Hz = H_pbuf_z(z, self.params)
        # basic sanity check
        # (the validator needs scale factor, so derive a from z)
        z_arr = np.asarray(z, dtype=float)
        a_arr = 1.0 / (1.0 + z_arr)
        validate_scale_and_rate(a_arr, Hz, z_arr)
        return Hz

    def hubble_function(self, a):
        """
        Hubble parameter as a function of scale factor a (0 < a ≤ 1).
        """
        check_scale_factor(a)
        Hz = H_pbuf_a(a, self.params)
        validate_scale_and_rate(a, Hz)
        return Hz

    # -------------------------
    # Elastic sector
    # -------------------------

    def omega_sigma(self, a):
        """
        Effective elastic correction added to E²(z) at scale factor `a`.
        """
        return omega_sigma_total(a, self.params)

    def elastic_energy_density(self, z):
        """
        Physical elastic reservoir σ_eff·S(z) used by Phase 6a (dimensionless).
        """
        a = 1.0 / (1.0 + z)
        rho_el = omega_sigma_raw(a, self.params)
        return float(rho_el) if np.isfinite(rho_el) else np.nan

    def density_parameters_at_z(self, z):
        """
        Return the fractional density budget at redshift z.

        We include matter, radiation, curvature, and the elastic correction.
        """
        a = float(1.0 / (1.0 + z))

        params = self.params

        omega_sigma = float(omega_sigma_total(a, params))

        Om_m = self.omega_m / (a**3)
        Om_r = self.omega_r / (a**4)
        Om_k = self.omega_k / (a**2)
    
        total = Om_m + Om_r + Om_k + omega_sigma
        if total <= 0.0:
            raise ValueError(f"Unphysical total density at z={z}: {total}")

        return {
            "omega_m": float(Om_m / total),
            "omega_r": float(Om_r / total),
            "omega_k": float(Om_k / total),
            "omega_lambda": 0.0,
            "omega_sigma": float(omega_sigma / total),
            "omega_elastic_raw": float(omega_sigma),
            "omega_elastic_radfix": 0.0,
            "omega_total": 1.0,
        }

    def closure_today(self):
        """
        Return Ω_total(a=1) for PBUF model.

        With the restored elastic sector:
        Ω_total(a=1) = Ω_m + Ω_r + Ω_k + Ω_σ(a=1)
        """
        omega_sigma_today = omega_sigma_total(1.0, self.params)

        # Total density parameter today
        omega_total = (
            self.omega_m +
            self.omega_r +
            self.omega_k +
            omega_sigma_today
        )
        
        return omega_total

    def parameters(self):
        """
        Return dict of model parameters for logging / sampling.
        """
        param_dict = {
            **self.params,
            "omega_m": self.omega_m,
            "omega_k": self.omega_k,
            "omega_r": self.omega_r,
            "omega_b": self.omega_b,
            "h": self.h,
        }
        return param_dict
