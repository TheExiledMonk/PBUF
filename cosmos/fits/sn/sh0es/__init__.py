"""
SH0ES Supernova Module

This module implements absolute H₀ calibration via local distance ladder SNe
(Cepheid-calibrated subset) or directly as a Gaussian prior on H₀.
"""

from .chi2 import chi2_sn_sh0es
from .loader import load_sh0es_data
from .observables import compute_sh0es_mu_model

__all__ = [
    "chi2_sn_sh0es",
    "load_sh0es_data",
    "compute_sh0es_mu_model",
]
