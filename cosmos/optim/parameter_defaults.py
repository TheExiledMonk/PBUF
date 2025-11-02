"""
Shared default parameter dictionaries for optimization workflows.
"""

from __future__ import annotations

EPS0_DEFAULT: float = 0.7
SIGMA8_PLANCK: float = 0.811
SIGMA8_PBUF_BASELINE: float = 0.5582

"""

PBUF_PARAMETER_DEFAULTS = {
    "H0": 67.4431,
    "Om0": 0.281444,
    "Or0": 9.2e-05,
    "Ok0": 0.0,
    "alpha": 0.0382959,
    "Rmax": 7.21754e+06,
    "k_sat": 0.87876,
    "eps0": 0.7,
    "n_alpha": 0.0,
    "n_eps": 0.0,
    "n_R": 0.0,
}
"""


# cosmos/optim/parameter_defaults.py

PBUF_PARAMETER_DEFAULTS = {
    # --- Cosmological background ---
    "H0": 73.65,         # Hubble constant (km/s/Mpc) — tuned joint-fit value
    "Om0": 0.252,        # Matter density fraction today
    "Or0": 9.2e-05,      # Radiation density fraction (fixed physical value)
    "Ok0": 0.0,          # Curvature (flat baseline)
    
    # --- Elastic sector parameters (PBUF-specific) ---
    "alpha": 0.03,     # Elastic amplitude (sets late-time stress level)
    "Rmax": 14000000.0,    # Characteristic elastic scale / transition radius
    "k_sat": 0.991,     # Elastic saturation coupling (0.5–1 = partial–full response)
    "eps0": 0.909768,    # Elastic energy normalization (vacuum rigidity baseline)
    
    # --- Evolution exponents (scaling of elastic terms) ---
    "n_alpha": 0.8,      # Power-law evolution index for α(z)
    "n_eps": -0.5,       # Power-law evolution index for ε(z)
    "n_R": 0.0,          # Scale exponent for Rmax evolution
    
    # --- Derived / placeholder parameters ---
    "Ol0": 0.0,          # Λ term is absent in PBUF (replaced by elastic sector)
}



LCDM_PARAMETER_DEFAULTS = {
    "H0": 73.65,
    "Om0": 0.253,
    "Or0": 9.2e-05,
    "Ok0": 0.0,
}

__all__ = [
    "EPS0_DEFAULT",
    "SIGMA8_PLANCK",
    "SIGMA8_PBUF_BASELINE",
    "PBUF_PARAMETER_DEFAULTS",
    "LCDM_PARAMETER_DEFAULTS",
]
