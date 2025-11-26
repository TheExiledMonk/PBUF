"""Basic grid/random search engine for optimisation."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, Sequence, Tuple

ParamDict = Dict[str, float]
EvalFn = Callable[[ParamDict], float]
SanityFn = Callable[[ParamDict], Tuple[bool, str | None]]


def _sample_parameters(bounds: Dict[str, Tuple[float, float]], rng: random.Random) -> ParamDict:
    params: ParamDict = {}
    for name, (lower, upper) in bounds.items():
        if lower == upper:
            params[name] = lower
        else:
            params[name] = rng.uniform(lower, upper)
    return params


def run_grid_search(
    *,
    evaluate: EvalFn,
    bounds: Dict[str, Tuple[float, float]],
    n_samples: int = 500,
    phase7a: SanityFn | None = None,
    rng_seed: int | None = None,
    model_name: str | None = None,
    dataset_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Randomly sample the provided bounds and report the best χ² value."""

    rng = random.Random(rng_seed)
    best_chi2 = math.inf
    best_params: ParamDict | None = None
    evaluations = 0

    for _ in range(max(1, n_samples)):
        params = _sample_parameters(bounds, rng)
        if phase7a:
            ok, _ = phase7a(params)
            if not ok:
                continue
        try:
            chi2 = evaluate(params)
        except Exception:
            continue
        if not math.isfinite(chi2):
            continue
        evaluations += 1
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_params = dict(params)

    return {
        "engine": "grid_search",
        "model": model_name,
        "datasets": list(dataset_names or []),
        "best_chi2": best_chi2,
        "best_parameters": best_params or {},
        "evaluations": evaluations,
        "n_samples": n_samples,
    }
