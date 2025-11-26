"""Shared basin-walker implementation used by the model-specific wrappers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from cosmos.optim.sanity import evaluate_candidate


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    prior: str = "flat"

    @property
    def span(self) -> float:
        return float(self.upper) - float(self.lower)


@dataclass
class BasinConfig:
    global_samples: int = 500
    best_seeds: int = 20
    local_steps: int = 400
    step_scale: float = 0.1
    max_iter: int = 3000
    rng_seed: int = 1234
    datasets: Sequence[str] | None = None
    threads: int = 1


@dataclass
class WalkerCandidate:
    params: Dict[str, float]
    chi2: float


BoundRange = Tuple[float, float]


class BasinWalkerBase:
    """
    Implements the three-stage basin-walking search described in the developer
    specification. Model-specific wrappers inject their module so the walker
    can call evaluate_chi2 directly.
    """

    _checkpoint_interval = 100
    _convergence_patience = 8
    _convergence_threshold = 1.0e-6
    _deep_convergence_patience = 25

    def __init__(self, model_module: Any, config: BasinConfig, bounds: Dict[str, Any] | None = None):
        self.model = model_module
        self.config = config
        self._bounds_payload = bounds or {}
        self.system_bounds: Dict[str, BoundRange] = self._prepare_parameter_bounds(self._bounds_payload)
        self.param_specs: List[ParameterSpec] = self._load_param_specs(model_module)
        dataset_names = config.datasets or ["cmb"]
        self.datasets = [name.lower() for name in dataset_names]
        self.rng = random.Random(config.rng_seed)
        self._history: List[Dict[str, Any]] = []
        self._acceptance: Dict[str, Dict[str, int]] = {}
        self._iterations: Dict[str, int] = {}
        self._evaluations = 0
        module_name = getattr(model_module, "__name__", "")
        self.model_name = module_name.split(".")[-1] if module_name else ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        seeds, best_seed = self._stage_global()
        if self.config.local_steps > 0:
            best_refined = self._stage_local(seeds, best_seed)
        else:
            best_refined = best_seed
        best_final = self._stage_deep(best_refined)
        return {
            "best_chi2": best_final.chi2,
            "best_params": dict(best_final.params),
            "history": self._history,
            "evaluations": self._evaluations,
        }

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage_global(self) -> Tuple[List[WalkerCandidate], WalkerCandidate]:
        seeds: List[WalkerCandidate] = []
        best: WalkerCandidate | None = None
        for _ in range(max(1, self.config.global_samples)):
            params = self._sample_uniform()
            chi2 = self._evaluate(params)
            candidate = WalkerCandidate(params, chi2)
            seeds.append(candidate)

            accepted = False
            if best is None or chi2 < best.chi2:
                best = WalkerCandidate(dict(params), chi2)
                accepted = True

            self._record_history("global", chi2, self.config.step_scale, accepted, params, best)

        seeds.sort(key=lambda cand: cand.chi2)
        keep = min(len(seeds), max(1, self.config.best_seeds))
        return seeds[:keep], best or seeds[0]

    def _stage_local(self, seeds: List[WalkerCandidate], global_best: WalkerCandidate) -> WalkerCandidate:
        best = WalkerCandidate(dict(global_best.params), global_best.chi2)
        for seed in seeds:
            current = WalkerCandidate(dict(seed.params), seed.chi2)
            tiny_progress = 0
            stagnant = 0
            step_scale = max(self.config.step_scale, 1.0e-3)

            for _ in range(self.config.local_steps):
                proposal = self._propose_nearby(current.params, step_scale)
                chi2 = self._evaluate(proposal)
                improved = chi2 < current.chi2 - 1.0e-9
                self._record_history("local", chi2, step_scale, improved, proposal if improved else None, best)

                if improved:
                    delta = current.chi2 - chi2
                    current = WalkerCandidate(proposal, chi2)
                    best = self._maybe_upgrade(best, current)
                    stagnant = 0
                    if delta < self._convergence_threshold:
                        tiny_progress += 1
                    else:
                        tiny_progress = 0
                else:
                    stagnant += 1

                if stagnant >= self._convergence_patience:
                    step_scale = max(step_scale * 0.5, 1.0e-4)
                    stagnant = 0

                if tiny_progress >= self._convergence_patience:
                    break
        return best

    def _stage_deep(self, candidate: WalkerCandidate) -> WalkerCandidate:
        best = WalkerCandidate(dict(candidate.params), candidate.chi2)
        step_scale = max(self.config.step_scale * 0.5, 1.0e-3)
        stagnant = 0

        for _ in range(max(1, self.config.max_iter)):
            step_scale = max(step_scale * 0.95, 1.0e-5)
            proposal = self._propose_nearby(best.params, step_scale)
            chi2 = self._evaluate(proposal)
            improved = chi2 < best.chi2 - 1.0e-9
            self._record_history("deep", chi2, step_scale, improved, proposal if improved else None, best)

            if improved:
                best = WalkerCandidate(proposal, chi2)
                stagnant = 0
            else:
                stagnant += 1

            if stagnant >= self._deep_convergence_patience:
                step_scale = max(step_scale * 0.5, 1.0e-5)
                stagnant = 0

        return best

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _load_param_specs(self, model_module: Any) -> List[ParameterSpec]:
        specs_raw = model_module.get_optimisable_parameters()
        specs: List[ParameterSpec] = []
        for raw in specs_raw:
            name = raw["name"]
            lower = float(raw["lower"])
            upper = float(raw["upper"])
            if upper < lower:
                raise ValueError(f"Upper bound smaller than lower bound for {name}.")
            bound = self.system_bounds.get(name)
            if bound is not None:
                lower = max(lower, bound[0])
                upper = min(upper, bound[1])
                if upper < lower:
                    raise ValueError(f"Bounds for {name} are inconsistent with optimisable limits.")
            specs.append(ParameterSpec(name=name, lower=lower, upper=upper, prior=str(raw.get("prior", "flat"))))
        if not specs:
            raise ValueError("Model did not declare any optimisable parameters.")
        return specs

    def _prepare_parameter_bounds(self, payload: Dict[str, Any] | None) -> Dict[str, BoundRange]:
        normalized: Dict[str, BoundRange] = {}
        raw_parameters = (payload or {}).get("parameters")
        if not isinstance(raw_parameters, dict):
            return normalized
        for name, interval in raw_parameters.items():
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(f"Bounds for {name} must be a two-element list.")
            lower = float(interval[0])
            upper = float(interval[1])
            if upper < lower:
                raise ValueError(f"Upper bound smaller than lower bound for {name}.")
            normalized[name] = (lower, upper)
        return normalized

    def _sample_uniform(self) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for spec in self.param_specs:
            if spec.span <= 0.0:
                params[spec.name] = spec.lower
                continue
            params[spec.name] = self.rng.uniform(spec.lower, spec.upper)
        self._clip_to_system_bounds(params)
        return params

    def _propose_nearby(self, base: Dict[str, float], step_scale: float) -> Dict[str, float]:
        proposal: Dict[str, float] = {}
        for spec in self.param_specs:
            value = base.get(spec.name, spec.lower)
            if spec.span <= 0.0:
                proposal[spec.name] = spec.lower
                continue
            delta = self.rng.uniform(-1.0, 1.0) * spec.span * step_scale
            proposal[spec.name] = min(max(value + delta, spec.lower), spec.upper)
        self._clip_to_system_bounds(proposal)
        return proposal

    def _evaluate(self, params: Dict[str, float]) -> float:
        self._clip_to_system_bounds(params)
        value, extras = evaluate_candidate(self.model_name, params, self.datasets)
        if not math.isfinite(value):
            raise ValueError("Model returned a non-finite χ² value.")
        self._evaluations += 1
        return value

    def _clip_to_system_bounds(self, params: Dict[str, float]) -> None:
        if not self.system_bounds:
            return
        for name, (lower, upper) in self.system_bounds.items():
            value = params.get(name)
            if value is None:
                continue
            clipped = min(max(float(value), lower), upper)
            params[name] = clipped

    def _maybe_upgrade(self, best: WalkerCandidate, candidate: WalkerCandidate) -> WalkerCandidate:
        if candidate.chi2 < best.chi2:
            return WalkerCandidate(dict(candidate.params), candidate.chi2)
        return best

    def _record_history(
        self,
        stage: str,
        chi2: float,
        step_scale: float,
        accepted: bool,
        params: Dict[str, float] | None,
        best: WalkerCandidate | None,
    ) -> None:
        stats = self._acceptance.setdefault(stage, {"accepted": 0, "total": 0})
        stats["total"] += 1
        if accepted:
            stats["accepted"] += 1
        ratio = stats["accepted"] / max(stats["total"], 1)

        iteration = self._iterations.get(stage, 0)
        self._iterations[stage] = iteration + 1
        checkpoint = ((iteration + 1) % self._checkpoint_interval == 0)

        entry: Dict[str, Any] = {
            "stage": stage,
            "iteration": iteration,
            "chi2": chi2,
            "step_scale": step_scale,
            "accepted": bool(accepted),
            "acceptance_ratio": ratio,
        }

        if params is not None:
            entry["params"] = dict(params)
        elif checkpoint and best is not None:
            entry["params"] = dict(best.params)
            entry["checkpoint"] = True
            entry["best_chi2"] = best.chi2

        self._history.append(entry)


__all__ = ["BasinConfig", "BasinWalkerBase", "ParameterSpec", "WalkerCandidate"]
