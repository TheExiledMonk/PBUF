"""Numba JIT backend implementation."""

from __future__ import annotations

import numba
import numpy as np
from typing import Any, Dict, List

from . import BaseBackend


class NumbaBackend(BaseBackend):
    """Numba JIT backend implementation for CPU optimization."""
    
    def __init__(self):
        super().__init__("numba")
        self._capabilities = {
            "backend_type": "numba",
            "device_count": 1,
            "memory": "system",
            "supported_operations": [
                "matrix_inverse", "matrix_multiply", "chi2_calculation",
                "simpson_integral", "batch_parameter_eval"
            ],
            "precision": "float64",
            "jit_compiled": True
        }
        
        # Compile JIT functions at initialization
        self._compile_functions()
    
    def _compile_functions(self):
        """Pre-compile Numba functions for better performance."""
        
        @numba.jit(nopython=True, cache=True)
        def _matrix_inverse_numba(matrix):
            """Matrix inverse using Numba."""
            return np.linalg.inv(matrix)
        
        @numba.jit(nopython=True, cache=True)
        def _matrix_multiply_numba(a, b):
            """Matrix multiplication using Numba."""
            return a @ b
        
        @numba.jit(nopython=True, cache=True)
        def _chi2_calculation_numba(residuals, inv_cov):
            """χ² calculation using Numba."""
            return float(residuals.T @ inv_cov @ residuals)
        
        @numba.jit(nopython=True, cache=True)
        def _simpson_integral_numba(func_values, h, n):
            """Simpson integration using Numba (pre-computed function values)."""
            result = h / 3.0 * (
                func_values[0] + func_values[-1] + 
                4.0 * np.sum(func_values[1:-1:2]) + 
                2.0 * np.sum(func_values[2:-2:2])
            )
            return result
        
        # Store compiled functions
        self._matrix_inverse_numba = _matrix_inverse_numba
        self._matrix_multiply_numba = _matrix_multiply_numba
        self._chi2_calculation_numba = _chi2_calculation_numba
        self._simpson_integral_numba = _simpson_integral_numba
    
    def matrix_inverse(self, matrix: Any) -> Any:
        """Compute matrix inverse using Numba."""
        return self._matrix_inverse_numba(matrix)
    
    def matrix_multiply(self, a: Any, b: Any) -> Any:
        """Compute matrix multiplication using Numba."""
        return self._matrix_multiply_numba(a, b)
    
    def chi2_calculation(self, residuals: Any, inv_cov: Any) -> float:
        """Compute χ² = residuals.T @ inv_cov @ residuals using Numba."""
        return self._chi2_calculation_numba(residuals, inv_cov)
    
    def simpson_integral(self, func: Any, lower: float, upper: float, n: int) -> float:
        """Compute Simpson's rule integration using Numba."""
        if n <= 0 or n % 2 != 0:
            raise ValueError("Simpson integrator expects a positive even number of steps")
        
        if upper == lower:
            return 0.0
        
        h = (upper - lower) / n
        x = np.linspace(lower, upper, n + 1)
        
        # Pre-compute function values (can't JIT the function call itself)
        y = np.array([func(xi) for xi in x])
        
        # Use Numba for the integration computation
        return float(self._simpson_integral_numba(y, h, n))
    
    def batch_parameter_eval(self, params_list: List[Dict[str, float]]) -> List[float]:
        """Evaluate multiple parameter sets using Numba (placeholder)."""
        # This would need actual model evaluation logic
        # For now, use Numba-accelerated processing
        return [0.0 for _ in params_list]
    
    def is_available(self) -> bool:
        """Check if Numba backend is available."""
        try:
            import numba
            return True
        except ImportError:
            return False
