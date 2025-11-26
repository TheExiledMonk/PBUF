"""Executor helpers for parallel batch evaluation."""

from __future__ import annotations

import multiprocessing as mp
from functools import partial
from typing import Callable, Sequence, Tuple


def _eval_wrapper(evaluator: Callable[[dict[str, float]], float], params: dict[str, float]) -> Tuple[dict[str, float], float]:
    return params, float(evaluator(params))


def make_process_pool(
    evaluator: Callable[[dict[str, float]], float], n_workers: int
) -> tuple[Callable[[Sequence[dict[str, float]]], list[tuple[dict[str, float], float]]], Callable[[], None]]:
    """
    Build a simple process pool executor for batch evaluation.

    Returns (map_fn, shutdown) where:
      - map_fn(params_list) -> list[(params, chi2)]
      - shutdown() tears down the pool
    """
    workers = max(1, int(n_workers))
    pool = mp.Pool(processes=workers)
    func = partial(_eval_wrapper, evaluator)

    def map_fn(params_list: Sequence[dict[str, float]]) -> list[tuple[dict[str, float], float]]:
        try:
            return pool.map(func, params_list)
        except Exception:
            # Fallback to in-process evaluation when pickling fails or pool errors.
            map_fn.fallback_hits += 1  # type: ignore[attr-defined]
            return [_eval_wrapper(evaluator, params) for params in params_list]

    def shutdown() -> None:
        pool.terminate()
        pool.join()

    map_fn.fallback_hits = 0  # type: ignore[attr-defined]
    map_fn.workers = workers  # type: ignore[attr-defined]
    return map_fn, shutdown
