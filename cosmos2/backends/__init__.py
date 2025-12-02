"""Backend system for cosmos2 GPU optimization."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod


class BackendInterface(Protocol):
    """Protocol defining the interface for all computation backends."""
    
    def matrix_inverse(self, matrix: Any) -> Any:
        """Compute matrix inverse."""
        ...
    
    def matrix_multiply(self, a: Any, b: Any) -> Any:
        """Compute matrix multiplication a @ b."""
        ...
    
    def chi2_calculation(self, residuals: Any, inv_cov: Any) -> float:
        """Compute χ² = residuals.T @ inv_cov @ residuals."""
        ...
    
    def simpson_integral(self, func: Any, lower: float, upper: float, n: int) -> float:
        """Compute Simpson's rule integration."""
        ...
    
    def batch_parameter_eval(self, params_list: List[Dict[str, float]]) -> List[float]:
        """Evaluate multiple parameter sets in batch."""
        ...


class BaseBackend(ABC):
    """Abstract base class for all backends."""
    
    def __init__(self, name: str):
        self.name = name
        self._capabilities: Dict[str, Any] = {}
    
    @abstractmethod
    def matrix_inverse(self, matrix: Any) -> Any:
        """Compute matrix inverse."""
        pass
    
    @abstractmethod
    def matrix_multiply(self, a: Any, b: Any) -> Any:
        """Compute matrix multiplication a @ b."""
        pass
    
    @abstractmethod
    def chi2_calculation(self, residuals: Any, inv_cov: Any) -> float:
        """Compute χ² = residuals.T @ inv_cov @ residuals."""
        pass
    
    @abstractmethod
    def simpson_integral(self, func: Any, lower: float, upper: float, n: int) -> float:
        """Compute Simpson's rule integration."""
        pass
    
    @abstractmethod
    def batch_parameter_eval(self, params_list: List[Dict[str, float]]) -> List[float]:
        """Evaluate multiple parameter sets in batch."""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get backend capabilities."""
        return self._capabilities.copy()
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        return True


def _is_backend_available(name: str) -> bool:
    """Test if a specific backend is available and functional."""
    if name == "cpu":
        return True  # Always available
    
    elif name == "numba":
        try:
            import numba
            # Test JIT compilation
            @numba.jit
            def test_func(x):
                return x * 2
            _ = test_func(5)
            return True
        except Exception:
            return False
    
    elif name == "hip":
        try:
            import ctypes
            import numpy as np
            
            # Set up ROCm environment
            os.environ['ROCM_PATH'] = '/opt/rocm-6.4.2'
            os.environ['HIP_PATH'] = '/opt/rocm-6.4.2'
            
            # Try to load HIP library
            hip_lib = ctypes.CDLL('/opt/rocm-6.4.2/lib/libamdhip64.so')
            
            # Test basic GPU operations
            hip_lib.hipGetDeviceCount.restype = ctypes.c_int
            hip_lib.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
            
            count = ctypes.c_int()
            result = hip_lib.hipGetDeviceCount(ctypes.byref(count))
            
            return result == 0 and count.value > 0
        except Exception:
            return False
    
    # Future backends
    elif name == "cuda":
        try:
            import cupy as cp
            _ = cp.array([1, 2, 3])  # Test basic operation
            return True
        except Exception:
            return False
    
    elif name == "opencl":
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            return len(platforms) > 0
        except Exception:
            return False
    
    elif name == "metal":
        try:
            import metalpy as mp
            return True
        except Exception:
            return False
    
    return False


def _create_backend(name: str) -> BaseBackend:
    """Create a backend instance."""
    if name == "cpu":
        from .cpu_backend import CPUBackend
        return CPUBackend()
    
    elif name == "numba":
        from .numba_backend import NumbaBackend
        return NumbaBackend()
    
    elif name == "hip":
        from .rocm_backend import ROCmBackend
        return ROCmBackend()
    
    # Future backends
    elif name == "cuda":
        from .cuda_backend import CUDABackend
        return CUDABackend()
    
    elif name == "opencl":
        from .opencl_backend import OpenCLBackend
        return OpenCLBackend()
    
    elif name == "metal":
        from .metal_backend import MetalBackend
        return MetalBackend()
    
    else:
        raise ValueError(f"Unknown backend: {name}")


def select_backend(*available_backends: str) -> BackendInterface:
    """
    Auto-select best available backend in priority order.
    
    Args:
        *available_backends: Backend names in priority order (first = highest priority)
        
    Returns:
        Backend instance for the best available backend
        
    Raises:
        RuntimeError: If no backends are available
    """
    for backend_name in available_backends:
        if _is_backend_available(backend_name):
            try:
                backend = _create_backend(backend_name)
                print(f"[backend] Selected backend: {backend_name}")
                return backend
            except Exception as e:
                print(f"[backend] Failed to create {backend_name} backend: {e}")
                continue
    
    raise RuntimeError("No available backends found")


def get_available_backends() -> List[str]:
    """Get list of all available backends."""
    all_backends = ["cpu", "numba", "hip", "cuda", "opencl", "metal"]
    available = []
    
    for backend_name in all_backends:
        if _is_backend_available(backend_name):
            available.append(backend_name)
    
    return available


def benchmark_backends() -> Dict[str, Dict[str, float]]:
    """
    Benchmark all available backends.
    
    Returns:
        Dictionary with timing results for each backend
    """
    import time
    import numpy as np
    
    results = {}
    available = get_available_backends()
    
    # Test data
    test_matrix = np.random.rand(100, 100)
    test_residuals = np.random.rand(100)
    test_cov = np.random.rand(100, 100)
    
    for backend_name in available:
        try:
            backend = _create_backend(backend_name)
            
            # Benchmark matrix inverse
            start = time.time()
            _ = backend.matrix_inverse(test_matrix)
            inv_time = time.time() - start
            
            # Benchmark χ² calculation
            start = time.time()
            _ = backend.chi2_calculation(test_residuals, test_cov)
            chi2_time = time.time() - start
            
            results[backend_name] = {
                "matrix_inverse": inv_time,
                "chi2_calculation": chi2_time
            }
            
        except Exception as e:
            print(f"[backend] Benchmark failed for {backend_name}: {e}")
            results[backend_name] = {"error": str(e)}
    
    return results


def get_backend():
    """Get the best available backend with smart selection."""
    # Try Smart backend first (auto-selects optimal backend for each operation)
    try:
        from .smart_backend import SmartBackend
        backend = SmartBackend()
        if backend.is_available():
            print("[Backend] Using Smart backend (auto-optimized)")
            return backend
    except Exception:
        pass
    
    # Fall back to ROCm
    try:
        from .rocm_backend import ROCmBackend
        backend = ROCmBackend()
        if backend.is_available():
            print("[Backend] Using ROCm backend")
            return backend
    except Exception:
        pass
    
    # Fall back to Numba
    try:
        from .numba_backend import NumbaBackend
        backend = NumbaBackend()
        print("[Backend] Using Numba backend")
        return backend
    except Exception as e:
        print(f"[Backend] No backend available: {e}")
        raise RuntimeError("No available computation backend")
