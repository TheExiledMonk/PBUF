"""
PBUF Data Preparation & Derivation Framework

This package provides a unified layer for transforming raw cosmological datasets
into standardized, analysis-ready formats with comprehensive validation and
provenance tracking.
"""

__version__ = "1.0.0"
__author__ = "PBUF Development Team"

from .core.schema import StandardDataset
from .core.interfaces import DerivationModule
from .engine.preparation_engine import DataPreparationFramework

__all__ = [
    "StandardDataset",
    "DerivationModule", 
    "DataPreparationFramework"
]