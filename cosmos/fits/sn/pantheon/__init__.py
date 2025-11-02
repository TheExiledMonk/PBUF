"""
Pantheon Supernova Module

This module implements relative-distance supernova constraints (Pantheon-like)
for cosmological model evaluation. It constrains the shape of the expansion
history via distance moduli μ(z) but not the absolute scale.
"""

from .chi2 import chi2_sn_pantheon
from .loader import load_pantheon_data
from .observables import compute_pantheon_mu_model

__all__ = [
    "chi2_sn_pantheon",
    "load_pantheon_data",
    "compute_pantheon_mu_model",
]
