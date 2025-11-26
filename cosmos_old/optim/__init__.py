"""Namespace for optimisation helpers, engines, and the sanity layer."""

from typing import Callable

from cosmos.optim.engines.basin import run_basin
from cosmos.optim.engines.grid_search import run_grid_search
from cosmos.optim.sanity import (
    HUGE_CHI2,
    evaluate_candidate,
    run_dataset_sanity,
    run_model_sanity,
)
from cosmos.optim.sanity_base import SanityResult
from cosmos.optim.sanity_utils import make_a_grid

ENGINE_REGISTRY = {
    "grid_search": run_grid_search,
    "basin": run_basin,
}


def get_engine(name: str) -> Callable[..., dict]:
    try:
        return ENGINE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown optimisation engine: {name}") from exc


__all__ = [
    "HUGE_CHI2",
    "SanityResult",
    "evaluate_candidate",
    "run_model_sanity",
    "run_dataset_sanity",
    "make_a_grid",
    "run_grid_search",
    "run_basin",
    "get_engine",
]
