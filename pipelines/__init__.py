"""
PBUF Cosmology Pipeline Package

This package contains the unified cosmology fitting infrastructure for PBUF and ΛCDM models.
The refactored architecture centralizes parameter handling, likelihood computations, and 
statistical analysis while maintaining strict physical consistency across all cosmological blocks.
"""

__version__ = "2.0.0"
__author__ = "PBUF Cosmology Team"

# Core module imports
from .fit_core import engine, parameter, likelihoods, datasets, statistics, logging_utils, integrity

__all__ = [
    "engine",
    "parameter", 
    "likelihoods",
    "datasets",
    "statistics",
    "logging_utils",
    "integrity"
]