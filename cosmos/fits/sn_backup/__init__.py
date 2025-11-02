"""
Supernova (SN Ia) fitting interface for cosmological models.

This package provides functions to compute supernova observables and
perform parameter fitting using SN Ia distance modulus measurements.

Modules:
- data_loader : load supernova datasets (Pantheon+, SH0ES, etc.)
- observables : compute model-predicted distance modulus μ(z)
- chi2        : compute χ² between model and observed μ
- optimizer   : fit model parameters to minimize χ²
"""

from .observables import compute_sn_mu_model, chi2_sn
from .chi2 import chi_squared_sn
from .optimizer import fit_sn

__all__ = [
    "compute_sn_mu_model",
    "chi2_sn",
    "chi_squared_sn",
    "fit_sn",
]
