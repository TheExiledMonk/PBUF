"""
PBUF4 Optimization Module

This package hosts deterministic grid-based scoring utilities used to compare
LCDM and PBUF cosmologies across all observational datasets.
"""

from .grid_pipeline import (
    BAO_DATASETS,
    BASE_DATASETS,
    evaluate_cosmology,
    prepare_grid,
    run_dual_grid_search,
    run_grid_search,
)
from .physics_validator import validate_cosmology

__all__ = [
    "BAO_DATASETS",
    "BASE_DATASETS",
    "evaluate_cosmology",
    "prepare_grid",
    "run_dual_grid_search",
    "run_grid_search",
    "validate_cosmology",
]
