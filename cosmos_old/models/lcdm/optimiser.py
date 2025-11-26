"""Model-specific optimiser for LCDM."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

AllowedParam = str
Bounds = Dict[AllowedParam, Tuple[float, float]]


@dataclass
class OptimiserResult:
    params: Dict[str, float]
    chi2: float
    evaluations: int


class LCDMOptimiser:
    TUNABLE = ("H0", "Omega_m0", "Omega_b0", "Omega_lambda0")

    def __init__(self, data, chi2_callback: Callable[[Dict[str, float]], float], seed: int | None = None):
        self._data = data
        self._chi2 = chi2_callback
        self._rng = random.Random(seed)

    def search(self, bounds: Bounds, iterations: int = 200) -> OptimiserResult:
        self._validate_bounds(bounds)

        best = None
        best_chi2 = float("inf")
        evals = 0

        for _ in range(iterations):
            candidate = {key: self._rng.uniform(*bounds[key]) for key in self.TUNABLE if key in bounds}
            chi2 = self._chi2(candidate)
            evals += 1

            if chi2 < best_chi2:
                best_chi2 = chi2
                best = candidate

        if best is None:
            raise RuntimeError("No LCDM optimiser evaluations were performed")

        return OptimiserResult(params=best, chi2=best_chi2, evaluations=evals)

    def _validate_bounds(self, bounds: Bounds) -> None:
        missing = [key for key in self.TUNABLE if key not in bounds]
        if missing:
            raise ValueError(f"Missing bounds for parameters: {missing}")
