"""
Physical sanity checks for cosmological models.

This package provides lightweight physical consistency validation
for cosmological parameter grids, separate from the core cosmology math.
"""

from .phase6a import phase6a_passes

__all__ = ["phase6a_passes"]
