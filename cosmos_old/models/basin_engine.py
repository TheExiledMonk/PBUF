"""Shared basin optimization engine used by model-specific basin managers."""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

HUGE_CHI2 = 1.0e12


@dataclass(frozen=True)
class OptimisationConfig:
    initial_samples: int = 60
    max_iterations: int = 300
    main_step_scale: float = 0.2
    threads: int = 8
    dataset_weights: Dict[str, float] = field(default_factory=lambda: {"cmb": 1.0})

    @classmethod
    def from_file(cls, path: Path | str) -> "OptimisationConfig":
        raw = json.loads(Path(path).read_text())
        weights = raw.get("dataset_weights") or {}
        normalized_weights = {key.lower(): float(value) for key, value in weights.items()}
        return cls(
            initial_samples=int(raw.get("initial_samples", cls.initial_samples)),
            max_iterations=int(raw.get("max_iterations", cls.max_iterations)),
            main_step_scale=float(raw.get("main_step_scale", cls.main_step_scale)),
            threads=int(raw.get("threads", cls.threads)),
            dataset_weights=normalized_weights or dict(cls.dataset_weights),
        )

    def with_overrides(
        self,
        *,
        initial_samples: int | None = None,
        max_iterations: int | None = None,
        main_step_scale: float | None = None,
        threads: int | None = None,
    ) -> "OptimisationConfig":
        return OptimisationConfig(
            initial_samples=initial_samples if initial_samples is not None else self.initial_samples,
            max_iterations=max_iterations if max_iterations is not None else self.max_iterations,
            main_step_scale=main_step_scale if main_step_scale is not None else self.main_step_scale,
            threads=threads if threads is not None else self.threads,
            dataset_weights=dict(self.dataset_weights),
        )


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    prior: str = "flat"

    @property
    def span(self) -> float:
        return float(self.upper) - float(self.lower)


class ParameterSpace:
    def __init__(self, specs: Sequence[ParameterSpec], system_bounds: Dict[str, Sequence[float]] | None = None):
        self.param_specs = list(specs)
        self.system_bounds: Dict[str, tuple[float, float]] = {}
        if system_bounds:
            for name, interval in system_bounds.items():
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    continue
                lower, upper = float(interval[0]), float(interval[1])
                if upper < lower:
                    continue
                self.system_bounds[name] = (lower, upper)

    def _clip_value(self, name: str, value: float, lower: float, upper: float) -> float:
        bounds = self.system_bounds.get(name)
        if bounds:
            lower = max(lower, bounds[0])
            upper = min(upper, bounds[1])
        return min(max(value, lower), upper)

    def clip(self, params: Dict[str, float]) -> None:
        for spec in self.param_specs:
            value = params.get(spec.name)
            if value is None:
                continue
            params[spec.name] = self._clip_value(spec.name, float(value), spec.lower, spec.upper)

    def bounds_ok(self, params: Dict[str, float]) -> bool:
        for spec in self.param_specs:
            value = params.get(spec.name)
            if value is None:
                return False
            value = float(value)
            lower = spec.lower
            upper = spec.upper
            bounds = self.system_bounds.get(spec.name)
            if bounds:
                lower = max(lower, bounds[0])
                upper = min(upper, bounds[1])
            if not (lower <= value <= upper):
                return False
        return True


class LatinHypercubeSampler:
    def __init__(self, space: ParameterSpace, samples: int, rng: random.Random):
        self.space = space
        self.samples = max(1, samples)
        self.rng = rng
        self._orders: Dict[str, List[int]] = {}
        for spec in space.param_specs:
            indices = list(range(self.samples))
            self.rng.shuffle(indices)
            self._orders[spec.name] = indices

    def sample(self, iteration: int) -> Dict[str, float] | None:
        if iteration >= self.samples:
            return None
        params: Dict[str, float] = {}
        for spec in self.space.param_specs:
            order = self._orders[spec.name]
            slot = order[iteration]
            step = (slot + self.rng.random()) / self.samples
            value = spec.lower + max(0.0, min(1.0, step)) * spec.span
            params[spec.name] = value
        self.space.clip(params)
        return params


@dataclass
class WorkerResult:
    dataset_name: str
    chi2: float
    status: str
    reasons: List[str] = field(default_factory=list)


class BaseWorker:
    """Abstract worker that manages model building, sanity, and chi²."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def evaluate(self, params: Dict[str, float]) -> WorkerResult:
        sanitized = {key: float(value) for key, value in params.items()}
        try:
            model = self.build_model(sanitized)
        except Exception as exc:
            return WorkerResult(dataset_name=self.dataset_name, chi2=HUGE_CHI2, status="build_failure", reasons=[str(exc)])

        sanity_ok, sanity_reasons = self.run_model_sanity(sanitized, model)
        if not sanity_ok:
            return WorkerResult(dataset_name=self.dataset_name, chi2=HUGE_CHI2, status="sanity_failed", reasons=sanity_reasons)

        try:
            chi2 = self.compute_chi2(model)
        except Exception as exc:
            return WorkerResult(dataset_name=self.dataset_name, chi2=HUGE_CHI2, status="chi2_failure", reasons=[str(exc)])

        return WorkerResult(dataset_name=self.dataset_name, chi2=float(chi2), status="ok", reasons=[])

    def build_model(self, params: Dict[str, float]) -> object:
        raise NotImplementedError()

    def run_model_sanity(self, params: Dict[str, float], model: object) -> tuple[bool, List[str]]:
        raise NotImplementedError()

    def compute_chi2(self, model: object) -> float:
        raise NotImplementedError()


class WorkerPool:
    def __init__(self, workers: Iterable[BaseWorker], max_workers: int):
        self.workers = list(workers)
        self.executor = ThreadPoolExecutor(max_workers=max_workers or 1)

    def evaluate(self, params: Dict[str, float]) -> Dict[str, WorkerResult]:
        futures = {self.executor.submit(worker.evaluate, params): worker for worker in self.workers}
        results: Dict[str, WorkerResult] = {}
        for future, worker in futures.items():
            try:
                result = future.result()
            except Exception as exc:
                result = WorkerResult(dataset_name=worker.dataset_name, chi2=HUGE_CHI2, status="failure", reasons=[str(exc)])
            results[result.dataset_name] = result
        return results

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)


@dataclass
class FinderOutcome:
    params: Dict[str, float]
    chi2: float
    dataset_results: Dict[str, WorkerResult]
    phase6a_passed: bool
    timestamp: str


class MainFinder:
    def __init__(self, space: ParameterSpace, dataset_names: Sequence[str], config: OptimisationConfig):
        self.space = space
        self.config = config
        self.dataset_names = [name.lower() for name in dataset_names]
        self.dataset_weights = config.dataset_weights
        self.rng = random.Random()
        self.sampler = LatinHypercubeSampler(space, config.initial_samples, self.rng)
        self.best_outcome: FinderOutcome | None = None

    def search(self, pool: WorkerPool) -> FinderOutcome | None:
        for iteration in range(self.config.max_iterations):
            params = self._propose(iteration)
            if params is None:
                continue
            if not self.space.bounds_ok(params):
                continue
            results = pool.evaluate(params)
            outcome = self._summarize_results(params, results)
            if outcome is None:
                continue
            if outcome.phase6a_passed and (self.best_outcome is None or outcome.chi2 < self.best_outcome.chi2):
                self.best_outcome = outcome
        return self.best_outcome

    def _propose(self, iteration: int) -> Dict[str, float] | None:
        if iteration < self.config.initial_samples:
            return self.sampler.sample(iteration)
        base = dict(self.best_outcome.params) if self.best_outcome else self.sampler.sample(iteration % self.sampler.samples)
        if base is None:
            return None
        scale = max(self.config.main_step_scale * (0.96 ** (iteration - self.config.initial_samples)), 1.0e-4)
        proposal = {}
        for spec in self.space.param_specs:
            if spec.span <= 0.0:
                proposal[spec.name] = spec.lower
                continue
            delta = self.rng.gauss(0.0, 1.0) * spec.span * scale
            base_value = base.get(spec.name, spec.lower)
            proposal[spec.name] = base_value + delta
        self.space.clip(proposal)
        return proposal

    def _summarize_results(self, params: Dict[str, float], results: Dict[str, WorkerResult]) -> FinderOutcome | None:
        total = 0.0
        complete = True
        for name in self.dataset_names:
            result = results.get(name)
            if result is None:
                return None
            weight = self.dataset_weights.get(name, 1.0)
            total += result.chi2 * weight
            if result.status != "ok":
                complete = False
        timestamp = datetime.now(timezone.utc).isoformat()
        return FinderOutcome(
            params=dict(params),
            chi2=total,
            dataset_results=results,
            phase6a_passed=complete,
            timestamp=timestamp,
        )


class BasinManager:
    def __init__(
        self,
        *,
        model_name: str,
        parameter_specs: Sequence[ParameterSpec],
        bounds_payload: Dict[str, Sequence[float]],
        dataset_names: Sequence[str],
        config: OptimisationConfig,
        worker_factory: Callable[[], Iterable[BaseWorker]],
        quantum_runner: Callable[[], Dict[str, object]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.parameter_space = ParameterSpace(parameter_specs, bounds_payload.get("parameters"))
        self.config = config
        self.dataset_names = [name.lower() for name in dataset_names]
        self.worker_factory = worker_factory
        self.quantum_runner = quantum_runner
        self._bootstrap_metadata: Dict[str, object] = {}
        self.bounds_payload = bounds_payload

    def run(self) -> Dict[str, object]:
        if self.quantum_runner:
            self._bootstrap_metadata = self.quantum_runner()
        workers = list(self.worker_factory())
        pool = WorkerPool(workers, max_workers=self.config.threads)
        finder = MainFinder(self.parameter_space, self.dataset_names, self.config)
        best = finder.search(pool)
        pool.shutdown()
        if best is None:
            raise RuntimeError("Basin finder did not return a best candidate.")
        return {
            "best_chi2": best.chi2,
            "best_parameters": best.params,
            "dataset_results": {name: result.__dict__ for name, result in best.dataset_results.items()},
            "phase6a_passed": best.phase6a_passed,
            "timestamp": best.timestamp,
            "quantum_metadata": self._bootstrap_metadata,
            "model": self.model_name,
            "datasets": self.dataset_names,
            "boundaries_used": self.bounds_payload.get("parameters", {}),
        }
