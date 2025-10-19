"""
Validators for dataset inputs.
"""

from __future__ import annotations

import numpy as np


class ValidationError(ValueError):
    """Raised when dataset validation fails."""


def validate_vector(vec: np.ndarray, name: str) -> None:
    if vec.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional")
    if np.any(~np.isfinite(vec)):
        raise ValidationError(f"{name} contains non-finite values")


def validate_covariance(cov: np.ndarray) -> None:
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValidationError("Covariance matrix must be square")
    if np.any(~np.isfinite(cov)):
        raise ValidationError("Covariance matrix contains non-finite values")
    if not np.allclose(cov, cov.T, atol=1e-10):
        cov[:] = 0.5 * (cov + cov.T)


def validate_lengths(*arrays: np.ndarray) -> None:
    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) > 1:
        raise ValidationError("Input vectors must share the same length")
