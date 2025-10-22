"""
Core fitting infrastructure for PBUF cosmology pipeline.

This module provides the unified architecture for cosmological parameter fitting,
including centralized parameter management, likelihood computations, and statistical analysis.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

# Type definitions for the unified fitting system
ParameterDict = Dict[str, Union[float, int, str]]
ResultsDict = Dict[str, Any]
DatasetDict = Dict[str, Union[np.ndarray, float, int]]
MetricsDict = Dict[str, float]
PredictionsDict = Dict[str, Union[float, np.ndarray]]

# Core module imports
from . import engine
from . import parameter
from . import likelihoods
from . import datasets
from . import statistics
from . import logging_utils
from . import integrity
from . import cmb_priors

__all__ = [
    # Type definitions
    "ParameterDict",
    "ResultsDict", 
    "DatasetDict",
    "MetricsDict",
    "PredictionsDict",
    # Core modules
    "engine",
    "parameter",
    "likelihoods", 
    "datasets",
    "statistics",
    "logging_utils",
    "integrity",
    "cmb_priors"
]