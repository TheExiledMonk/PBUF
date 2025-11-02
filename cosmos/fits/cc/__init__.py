"""
Cosmic chronometer (CC) fitting interface.

Cosmic chronometers provide direct measurements of H(z) from
differential galaxy ages, independent of a cosmological model.

Exports:
- compute_cc_hubble_model  (predict H(z) from the cosmology)
- chi_squared_cc           (compute χ² against CC data)
- fit_cc                   (fit model params to CC data)
"""

from .observables import compute_cc_hubble_model, chi2_cc
from .chi2 import chi_squared_cc
from .optimizer import fit_cc

__all__ = [
    "compute_cc_hubble_model",
    "chi2_cc",
    "chi_squared_cc",
    "fit_cc",
]
