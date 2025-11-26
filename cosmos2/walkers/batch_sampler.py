"""Batch sampler utilities for parameter proposals."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from cosmos2.utils.batch_utils import clamp_to_bounds


def generate_batch(params_state: Dict, batch_size: int) -> List[Dict[str, float]]:
    """
    Generate a batch of parameter dictionaries by uniform sampling within bounds.

    params_state is expected to provide:
      - "bounds": mapping of param -> (low, high)
      - optional "anchor": dict of starting values to jitter around
      - optional "rng": np.random.Generator
    """
    bounds: Dict[str, Tuple[float, float]] = params_state.get("bounds", {})
    anchor: Dict[str, float] = params_state.get("anchor", {})
    rng: np.random.Generator = params_state.get("rng") or np.random.default_rng()

    keys = list(bounds.keys()) or list(anchor.keys())
    batch: List[Dict[str, float]] = []
    for _ in range(int(batch_size)):
        sample: Dict[str, float] = {}
        for key in keys:
            if key in bounds:
                low, high = bounds[key]
                sample[key] = float(rng.uniform(low, high))
            elif key in anchor:
                sample[key] = float(anchor[key])
        batch.append(clamp_to_bounds(sample, bounds))
    return batch


def update_after_results(params_state: Dict, results: Iterable[Tuple[Dict[str, float], float]]) -> Dict:
    """
    Update the sampler anchor to the best-performing set in the provided results.
    """
    best_pair = min(results, key=lambda pair: pair[1], default=None)
    if best_pair is None:
        return params_state
    best_params, _ = best_pair
    params_state["anchor"] = dict(best_params)
    return params_state
