"""
Parallel execution helpers for batch cosmology evaluations.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, List, Sequence, Tuple


def map_tasks(func: Callable[[object], object], tasks: Sequence[object], max_workers: int | None = None) -> List[object]:
    """
    Evaluate `func` over `tasks` using a shared process pool.

    Falls back to serial execution for short task lists.
    """

    if len(tasks) < 2:
        return [func(task) for task in tasks]

    results: List[object] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(func, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    return results
