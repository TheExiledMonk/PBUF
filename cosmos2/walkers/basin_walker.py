"""BasinWalker orchestrates sampling for a single model."""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from cosmos2.utils.batch_utils import clamp_to_bounds
from cosmos2.walkers.batch_sampler import generate_batch, update_after_results


class BasinWalker:
    """
    Basin sampler with optional Latin hypercube scatter, local island jitters,
    and coarse grid seeding. Keeps compatibility with the simple interface used
    by model threads.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        evaluator: Callable[[Dict[str, float]], float],
        *,
        batch_size: int = 32,
        rng=None,
        n_scatter: int = 0,
        scatter_scale: float = 0.05,
        island_fraction: float = 0.5,
        grid_points: int | None = None,
    ):
        self.bounds = dict(bounds)
        self.evaluator = evaluator
        self.batch_size = int(batch_size)
        self.rng: np.random.Generator = rng or np.random.default_rng()

        self.state: Dict = {"bounds": self.bounds, "rng": self.rng}
        self.best_params: Dict[str, float] | None = None
        self.best_chi2: float = float("inf")

        # Scatter pools
        self._scatter_pool: List[Dict[str, float]] = self._latin_hypercube_samples(n_scatter)
        self._grid_pool: List[Dict[str, float]] = self._coarse_grid_pool(grid_points)
        self._islands: List[Tuple[Dict[str, float], float]] = []
        self.scatter_scale = float(max(scatter_scale, 0.0))
        self.island_fraction = float(min(max(island_fraction, 0.0), 1.0))

    # --------------------------
    # Proposal generation
    # --------------------------
    def _latin_hypercube_samples(self, n: int) -> List[Dict[str, float]]:
        if n <= 0 or not self.bounds:
            return []
        keys = list(self.bounds.keys())
        dim = len(keys)
        points = np.zeros((n, dim), dtype=float)
        for j, key in enumerate(keys):
            low, high = self.bounds[key]
            # Latin hypercube in [0,1] then scale to bounds
            cut = np.linspace(0, 1, n + 1)
            u = self.rng.uniform(size=n)
            points[:, j] = cut[:n] + u * (1.0 / n)
            self.rng.shuffle(points[:, j])
            points[:, j] = low + points[:, j] * (high - low)
        samples = []
        for i in range(n):
            sample = {k: float(points[i, idx]) for idx, k in enumerate(keys)}
            samples.append(clamp_to_bounds(sample, self.bounds))
        return samples

    def _coarse_grid_pool(self, grid_points: int | None) -> List[Dict[str, float]]:
        if grid_points is None or grid_points <= 1 or not self.bounds:
            return []
        full_keys = list(self.bounds.keys())
        keys = list(full_keys)
        # Limit grid dimensionality to avoid explosion but keep essential parameters (e.g., Rmax for PBUF).
        max_dim = 3
        if len(keys) > max_dim:
            truncated = keys[:max_dim]
            if "Rmax" in keys and "Rmax" not in truncated:
                truncated[-1] = "Rmax"
            keys = truncated
        linspaces = [np.linspace(lo, hi, grid_points) for lo, hi in (self.bounds[k] for k in keys)]
        mesh = np.stack(np.meshgrid(*linspaces, indexing="ij"), axis=-1).reshape(-1, len(keys))
        pool: List[Dict[str, float]] = []
        for row in mesh:
            sample = {k: float(val) for k, val in zip(keys, row)}
            # Fill any truncated dimensions with midpoints so required params remain present.
            for key in full_keys:
                if key not in sample:
                    lo, hi = self.bounds[key]
                    sample[key] = 0.5 * (lo + hi)
            pool.append(clamp_to_bounds(sample, self.bounds))
        self.rng.shuffle(pool)
        return pool

    def _jitter(self, anchor: Dict[str, float]) -> Dict[str, float]:
        jittered = {}
        for key, value in anchor.items():
            if key not in self.bounds:
                jittered[key] = value
                continue
            lo, hi = self.bounds[key]
            span = hi - lo
            step = span * self.scatter_scale
            if step <= 0.0:
                jittered[key] = value
                continue
            jittered[key] = value + self.rng.normal(scale=step)
        return clamp_to_bounds(jittered, self.bounds)

    def _propose_batch(self) -> List[Dict[str, float]]:
        proposals: List[Dict[str, float]] = []

        # Prefer coarse grid seeds if available.
        while self._grid_pool and len(proposals) < self.batch_size:
            proposals.append(self._grid_pool.pop())

        # Then use precomputed Latin hypercube scatter.
        while self._scatter_pool and len(proposals) < self.batch_size:
            proposals.append(self._scatter_pool.pop())

        remaining = self.batch_size - len(proposals)
        if remaining <= 0:
            return proposals

        # Mix local island jitters and global uniform samples.
        local_count = 0
        if self._islands:
            local_count = int(math.ceil(self.island_fraction * remaining))
            local_count = min(local_count, remaining)
        global_count = remaining - local_count

        for _ in range(local_count):
            idx = self.rng.integers(low=0, high=len(self._islands))
            anchor, _ = self._islands[int(idx)]
            proposals.append(self._jitter(anchor))

        if global_count > 0:
            # Reuse uniform sampler for the remainder.
            self.state["anchor"] = {}
            proposals.extend(generate_batch(self.state, global_count))

        return proposals

    # --------------------------
    # Public API
    # --------------------------
    def run_batch(self, map_fn=None) -> List[Tuple[Dict[str, float], float]]:
        """Generate a batch, evaluate chi2, update best, and return results."""
        batch = self._propose_batch()
        results: List[Tuple[Dict[str, float], float]] = []
        if map_fn is not None:
            try:
                mapped = map_fn(batch)
            except Exception:
                mapped = []
            for params, chi2 in mapped:
                chi2_val = float(chi2)
                if not math.isfinite(chi2_val):
                    chi2_val = float("inf")
                results.append((params, chi2_val))
                if chi2_val < self.best_chi2:
                    self.best_chi2 = chi2_val
                    self.best_params = dict(params)
        else:
            for params in batch:
                chi2 = float(self.evaluator(params))
                if not math.isfinite(chi2):
                    chi2 = float("inf")
                results.append((params, chi2))
                if chi2 < self.best_chi2:
                    self.best_chi2 = chi2
                    self.best_params = dict(params)
        self._update_islands(results)
        self.state = update_after_results(self.state, results)
        return results

    # --------------------------
    # Internal book-keeping
    # --------------------------
    def _update_islands(self, results: Iterable[Tuple[Dict[str, float], float]], max_islands: int = 8) -> None:
        for params, chi2 in results:
            if not math.isfinite(chi2):
                continue
            self._islands.append((params, chi2))
        if not self._islands:
            return
        self._islands = sorted(self._islands, key=lambda pair: pair[1])[:max_islands]
