"""Factory helpers for models and optimisation engines."""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

from cosmos.optim.engines import run_basin, run_grid_search
from cosmos.models.factory import make_model_factory as _make_model_factory

ParamDict = Dict[str, float]
EvalFn = Callable[[ParamDict], float]
Phase7aFn = Callable[[ParamDict], Tuple[bool, str | None]]


def make_model_factory(model_name: str, *, datasets: Sequence[str] | None = None) -> Callable[[ParamDict], Any]:
    return _make_model_factory(model_name, datasets=datasets)


class BaseEngine:
    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = dict(settings)

    def optimise(
        self,
        *,
        objective: EvalFn,
        bounds: Dict[str, Tuple[float, float]],
        fixed_parameters: Dict[str, float],
        initial_parameters: Dict[str, float],
        phase7a: Phase7aFn,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class BasinEngine(BaseEngine):
    _key_map = {
        "threads": "n_threads",
        "seed": "rng_seed",
        "steps": "n_refine",
    }
    _allowed = {"n_scatter", "n_seeds", "n_refine", "n_threads", "rng_seed"}

    def optimise(
        self,
        *,
        objective,
        bounds,
        fixed_parameters,
        initial_parameters,
        phase7a,
    ):
        kw: Dict[str, Any] = {}
        for key, value in self.settings.items():
            mapped = self._key_map.get(key, key)
            if mapped in self._allowed:
                kw[mapped] = value
        result = run_basin(
            evaluate=objective,
            bounds=bounds,
            phase7a=phase7a,
            **kw,
        )
        return {
            "engine": "basin",
            "best_params": result.get("best_params", {}),
            "best_chi2": result.get("best_chi2"),
            "trace": result.get("islands", []),
            "metadata": result,
        }


class GridEngine(BaseEngine):
    _key_map = {"samples": "n_samples", "seed": "rng_seed"}
    _allowed = {"n_samples", "phase7a", "rng_seed"}

    def optimise(
        self,
        *,
        objective,
        bounds,
        fixed_parameters,
        initial_parameters,
        phase7a,
    ):
        kw: Dict[str, Any] = {}
        for key, value in self.settings.items():
            mapped = self._key_map.get(key, key)
            if mapped in self._allowed:
                kw[mapped] = value
        result = run_grid_search(
            evaluate=objective,
            bounds=bounds,
            phase7a=phase7a,
            **kw,
        )
        return {
            "engine": "grid",
            "best_params": result.get("best_parameters", {}),
            "best_chi2": result.get("best_chi2"),
            "trace": [],
            "metadata": result,
        }


class GriSearchEngine(BaseEngine):
    def optimise(self, **_: Any) -> Dict[str, Any]:
        raise NotImplementedError("GriSearch engine is not available in this build.")


def make_engine(engine_name: str, settings: Dict[str, Any]) -> BaseEngine:
    normalized = engine_name.strip().lower()
    if normalized == "basin":
        return BasinEngine(settings)
    if normalized == "grid":
        return GridEngine(settings)
    if normalized == "gri_search":
        return GriSearchEngine(settings)
    raise ValueError(f"Unknown engine '{engine_name}'.")
