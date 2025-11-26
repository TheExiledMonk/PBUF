"""Basin walker sampler for CLI optimisation."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any, Callable, Dict, List, Sequence, Tuple

import math
import numpy as np
from collections import Counter

ParamDict = Dict[str, float]
EvalFn = Callable[[ParamDict], float]
SanityFn = Callable[[ParamDict], Tuple[bool, str | None]]


class _EvaluationTracker:
    def __init__(self, target: EvalFn, on_eval: Callable[[ParamDict, float], None] | None = None) -> None:
        self._target = target
        self._lock = threading.Lock()
        self.count = 0
        self._on_eval = on_eval

    def __call__(self, params: ParamDict) -> float:
        value = self._target(params)
        if self._on_eval:
            self._on_eval(dict(params), value)
        with self._lock:
            self.count += 1
        return value


def _random_param(bounds: Dict[str, Tuple[float, float]], rng: np.random.Generator) -> ParamDict:
    return {
        name: float(lower if lower == upper else rng.uniform(lower, upper))
        for name, (lower, upper) in bounds.items()
    }


def _normalized_vector(params: ParamDict, bounds: Dict[str, Tuple[float, float]], keys: Sequence[str]) -> np.ndarray:
    vector = []
    for key in keys:
        lower, upper = bounds[key]
        value = params.get(key, lower)
        if upper <= lower:
            vector.append(0.0)
            continue
        vector.append((value - lower) / (upper - lower))
    return np.array(vector, dtype=float)


def _scatter_samples(
    tracker: _EvaluationTracker,
    bounds: Dict[str, Tuple[float, float]],
    phase7a: SanityFn,
    n_scatter: int,
    rng: np.random.Generator,
    n_threads: int,
    ) -> tuple[List[Dict[str, Any]], list[str]]:
    candidates: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, n_threads)) as executor:
        futures: Dict[Future, ParamDict] = {}
        failure_reasons: list[str] = []
        for _ in range(max(1, n_scatter)):
            params = _random_param(bounds, rng)
            ok, reason = phase7a(params)
            if not ok:
                failure_reasons.append(reason or "unspecified")
                continue
            futures[executor.submit(tracker, dict(params))] = dict(params)

        for future, params in futures.items():
            try:
                chi2 = future.result()
            except Exception as exc:
                failure_reasons.append(f"evaluation_exception:{type(exc).__name__}")
                continue
            if not math.isfinite(chi2):
                failure_reasons.append("nonfinite_chi2")
                continue
            candidates.append({"params": params, "chi2": chi2})
    return candidates, failure_reasons


def _extract_seeds(
    candidates: List[Dict[str, Any]],
    bounds: Dict[str, Tuple[float, float]],
    n_seeds: int,
    eps: float = 0.02,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    sorted_candidates = sorted(candidates, key=lambda item: item["chi2"])
    param_keys = list(bounds.keys())
    seeds: List[Dict[str, Any]] = []
    centers: List[np.ndarray] = []
    for candidate in sorted_candidates:
        if len(seeds) >= max(1, n_seeds):
            break
        normalized = _normalized_vector(candidate["params"], bounds, param_keys)
        if any(np.linalg.norm(normalized - center) < eps for center in centers):
            continue
        seeds.append(candidate)
        centers.append(normalized)
    return seeds


def _local_descent(
    seed_params: ParamDict,
    seed_chi2: float,
    tracker: _EvaluationTracker,
    phase7a: SanityFn,
    bounds: Dict[str, Tuple[float, float]],
    n_steps: int,
) -> Tuple[ParamDict, float]:
    params = dict(seed_params)
    best_chi2 = seed_chi2
    step = {name: 0.1 * (bounds[name][1] - bounds[name][0]) for name in params}

    for _ in range(max(1, n_steps)):
        improved = False
        for name in params:
            header = bounds[name]
            lower, upper = header
            for direction in (+1.0, -1.0):
                trial = dict(params)
                trial[name] = float(
                    np.clip(trial[name] + direction * step[name], lower, upper)
                )
                ok, _ = phase7a(trial)
                if not ok:
                    continue
                try:
                    chi2 = tracker(trial)
                except Exception:
                    continue
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    params = trial
                    improved = True
        if not improved:
            for key in step:
                step[key] *= 0.5
            if all(
                step[key] < 1e-4 * max(1.0, bounds[key][1] - bounds[key][0])
                for key in step
            ):
                break
    return _clip_params(params, bounds), best_chi2


def _run_local_descent(
    seeds: List[Dict[str, Any]],
    tracker: _EvaluationTracker,
    phase7a: SanityFn,
    bounds: Dict[str, Tuple[float, float]],
    n_steps: int,
    n_threads: int,
) -> List[Dict[str, Any]]:
    if not seeds:
        return []
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, n_threads)) as executor:
        futures = {
            executor.submit(
                _local_descent,
                dict(seed["params"]),
                float(seed["chi2"]),
                tracker,
                phase7a,
                bounds,
                n_steps,
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            try:
                params, chi2 = future.result()
            except Exception:
                continue
            results.append({"params": _clip_params(params, bounds), "chi2": chi2})
    return results


def _cluster_islands(
    minima: List[Dict[str, Any]],
    bounds: Dict[str, Tuple[float, float]],
    eps: float = 0.1,
) -> List[Dict[str, Any]]:
    if not minima:
        return []
    keys = list(bounds.keys())
    islands: List[Dict[str, Any]] = []
    for candidate in minima:
        normalized = _normalized_vector(candidate["params"], bounds, keys)
        assigned = False
        for island in islands:
            distance = np.linalg.norm(normalized - island["center_norm"])
            if distance < eps:
                island["minima"].append(
                    {"params": dict(candidate["params"]), "chi2": candidate["chi2"]}
                )
                if candidate["chi2"] < island["best_chi2"]:
                    island["best_chi2"] = candidate["chi2"]
                    island["center_params"] = dict(candidate["params"])
                    island["center_norm"] = normalized
                assigned = True
                break
        if not assigned:
            islands.append(
                {
                    "center_params": dict(candidate["params"]),
                    "center_norm": normalized,
                    "best_chi2": candidate["chi2"],
                    "minima": [
                        {"params": dict(candidate["params"]), "chi2": candidate["chi2"]}
                    ],
                }
            )
    final_islands: List[Dict[str, Any]] = []
    for island in islands:
        final_islands.append(
            {
                "center_params": island["center_params"],
                "minima": island["minima"],
                "best_chi2": island["best_chi2"],
                "n_minima": len(island["minima"]),
            }
        )
    final_islands.sort(key=lambda entry: entry["best_chi2"])
    return final_islands


def _clip_params(params: ParamDict, bounds: Dict[str, Tuple[float, float]]) -> ParamDict:
    clipped: ParamDict = {}
    for name, value in params.items():
        if name in bounds:
            lower, upper = bounds[name]
            clipped[name] = float(np.clip(value, lower, upper))
        else:
            clipped[name] = float(value)
    return clipped


def run_basin(
    *,
    evaluate: EvalFn,
    bounds: Dict[str, Tuple[float, float]],
    phase7a: SanityFn,
    n_scatter: int = 200,
    n_seeds: int = 10,
    n_refine: int = 50,
    n_threads: int = 4,
    rng_seed: int | None = None,
) -> Dict[str, Any]:
    global_best: dict[str, Any] = {"chi2": math.inf, "params": {}}
    global_best_lock = threading.Lock()

    def _record_global_best(params: ParamDict, chi2: float) -> None:
        if not math.isfinite(chi2):
            return
        with global_best_lock:
            if chi2 < global_best["chi2"]:
                global_best["chi2"] = chi2
                global_best["params"] = dict(params)

    tracker = _EvaluationTracker(evaluate, on_eval=_record_global_best)
    rng = np.random.default_rng(rng_seed)

    candidates, failure_reasons = _scatter_samples(tracker, bounds, phase7a, n_scatter, rng, n_threads)
    if not candidates:
        summary = "; ".join(dict.fromkeys(failure_reasons[:5]))
        evt = Counter(failure_reasons)
        details = "; ".join(
            f"{reason or 'unspecified'}={count}"
            for reason, count in evt.most_common(5)
        )
        raise RuntimeError(
            "Basin scatter stage found no Phase-7a-safe candidates."
            + (f" sample reasons: {summary}" if summary else "")
            + (f" :: breakdown: {details}" if evt else "")
        )

    seeds = _extract_seeds(candidates, bounds, n_seeds)
    if not seeds:
        raise RuntimeError("Basin stage could not select any distinct seeds.")

    minima = _run_local_descent(seeds, tracker, phase7a, bounds, n_refine, n_threads)
    if not minima:
        minima = seeds

    islands = _cluster_islands(minima, bounds)
    best_candidate = min(minima, key=lambda entry: entry["chi2"]) if minima else None

    if global_best["chi2"] < math.inf and (
        best_candidate is None or global_best["chi2"] < best_candidate["chi2"]
    ):
        best_candidate = {
            "params": dict(global_best["params"]),
            "chi2": global_best["chi2"],
        }

    if best_candidate is None:
        raise RuntimeError("No basin candidate survived the optimisation.")

    best_candidate["params"] = _clip_params(best_candidate["params"], bounds)
    return {
        "engine": "basin",
        "best_params": dict(best_candidate["params"]),
        "best_chi2": float(best_candidate["chi2"]),
        "islands": islands,
        "n_scatter": max(1, n_scatter),
        "n_refine": max(1, n_refine),
        "n_evals": tracker.count,
    }
