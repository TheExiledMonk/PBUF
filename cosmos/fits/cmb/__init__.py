"""
CMB fitting interface for cosmological models.

This package provides functions to compute CMB observables and
perform parameter fitting using Planck 2018 compressed distance priors.

Modules:
- data_loader : load priors and covariance
- observables : compute model predictions (R, l_A, theta_star, z_star)
- chi2        : compute chi-squared between model and Planck data
- optimizer   : fit model parameters to minimize χ²
"""

from .observables import redshift_star, redshift_drag, cmb_observables, chi_squared_cmb, PLANCK_2018_PRIORS, PLANCK_2018_COVARIANCE
from .optimizer import fit_cmb

__all__ = [
    "cmb_observables",
    "redshift_star",
    "redshift_drag",
    "chi_squared_cmb",
    "PLANCK_2018_PRIORS",
    "PLANCK_2018_COVARIANCE",
    "fit_cmb",
]
