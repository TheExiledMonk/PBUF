"""CPU backend implementation using NumPy."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List

from . import BaseBackend


class CPUBackend(BaseBackend):
    """CPU backend implementation using NumPy."""
    
    def __init__(self):
        super().__init__("cpu")
        self._capabilities = {
            "backend_type": "cpu",
            "device_count": 1,
            "memory": "system",
            "supported_operations": [
                "matrix_inverse", "matrix_multiply", "chi2_calculation",
                "simpson_integral", "batch_parameter_eval"
            ],
            "precision": "float64"
        }
    
    def matrix_inverse(self, matrix: Any) -> Any:
        """Compute matrix inverse using NumPy."""
        return np.linalg.inv(matrix)
    
    def matrix_multiply(self, a: Any, b: Any) -> Any:
        """Compute matrix multiplication using NumPy."""
        return np.matmul(a, b)
    
    def chi2_calculation(self, residuals: Any, inv_cov: Any) -> float:
        """Compute χ² = residuals.T @ inv_cov @ residuals."""
        return float(residuals.T @ inv_cov @ residuals)
    
    def simpson_integral(self, func: Any, lower: float, upper: float, n: int) -> float:
        """Compute Simpson's rule integration using NumPy."""
        if n <= 0 or n % 2 != 0:
            raise ValueError("Simpson integrator expects a positive even number of steps")
        
        if upper == lower:
            return 0.0
        
        h = (upper - lower) / n
        x = np.linspace(lower, upper, n + 1)
        y = np.array([func(xi) for xi in x])
        
        # Simpson's rule: h/3 * (y0 + yn + 4*sum(y_odd) + 2*sum(y_even_except_endpoints))
        result = h / 3.0 * (
            y[0] + y[-1] + 
            4.0 * np.sum(y[1:-1:2]) + 
            2.0 * np.sum(y[2:-2:2])
        )
        
        return float(result)
    
    def batch_parameter_eval(self, params_list: List[Dict[str, float]]) -> List[float]:
        """Evaluate multiple parameter sets sequentially (CPU fallback)."""
        # This is a placeholder - actual implementation would depend on the model
        # For now, just return placeholder values
        return [0.0 for _ in params_list]
    
    def is_available(self) -> bool:
        """Check if CPU backend is available (always true)."""
        return True
