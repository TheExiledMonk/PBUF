from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from itertools import product
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from cosmos.optim.dataset_evaluators import list_available_datasets
from cosmos.optim.parameter_defaults import (
    LCDM_PARAMETER_DEFAULTS,
    PBUF_PARAMETER_DEFAULTS,
)
from cosmos.optim.grid_pipeline import evaluate_cosmology
from cosmos.optim.chi2_targets import Chi2TargetRegistry
from cosmos.phase.validation import prior_violation_reason, record_prior_violation
from cosmos.optim.coord_optimizer.observers import BasinWalkObserver

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm unavailable
    tqdm = None

CPU_COUNT = os.cpu_count() or 1
MAX_WORKERS = max(1, min(CPU_COUNT, 16))
EDGE_EXPANSION_MAX = 5
EDGE_TOLERANCE_FACTOR = 0.5
EDGE_TOLERANCE_EPS = 1.0e-6
EDGE_EXPANSION_LIMIT = 6
MAX_EQ_PASSES = 3

DEFAULT_REFERENCES: Dict[str, Dict[str, float]] = {
    "pbuf": dict(PBUF_PARAMETER_DEFAULTS),
    "lcdm": dict(LCDM_PARAMETER_DEFAULTS),
}

DEFAULT_SCAN_PRESETS: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "pbuf": {
        "H0": {
            "coarse": {"type": "linear", "start": 66.0, "stop": 74.0, "step": 0.1},
            "refine": {
                "type": "linear_relative",
                "radius": 0.5,
                "step": 0.05,
                "clip_min": 60.0,
                "clip_max": 80.0,
            },
        },
        "Om0": {
            "coarse": {"type": "linear", "start": 0.250, "stop": 0.350, "step": 0.002},
            "refine": {
                "type": "linear_relative",
                "radius": 0.01,
                "step": 0.001,
                "clip_min": 0.20,
                "clip_max": 0.40,
            },
        },
        "alpha": {
            "coarse": {
                "type": "list",
                "values": [0.003, 0.005, 0.008, 0.012, 0.016, 0.020, 0.024, 0.030],
            },
            "refine": {
                "type": "linear_relative",
                "radius": 0.005,
                "step": 0.001,
                "clip_min": 5.0e-4,
                "clip_max": 0.05,
            },
        },
        "k_sat": {
            "coarse": {"type": "linear", "start": 0.90, "stop": 0.995, "step": 0.005},
            "refine": {
                "type": "linear_relative",
                "radius": 0.01,
                "step": 0.002,
                "clip_min": 0.80,
                "clip_max": 0.999,
            },
        },
        "Rmax": {
            "coarse": {
                "type": "list",
                "values": [1.0e7, 2.0e7, 4.0e7, 7.0e7, 1.0e8],
            },
            "refine": {
                "type": "log_relative",
                "factors": [0.5, 0.7, 1.0, 1.3, 1.6],
                "clip_min": 5.0e6,
                "clip_max": 5.0e8,
            },
        },
    },
    "lcdm": {
        "H0": {
            "coarse": {"type": "linear", "start": 66.0, "stop": 74.0, "step": 0.1},
            "refine": {
                "type": "linear_relative",
                "radius": 0.5,
                "step": 0.05,
                "clip_min": 60.0,
                "clip_max": 80.0,
            },
        },
        "Om0": {
            "coarse": {"type": "linear", "start": 0.250, "stop": 0.350, "step": 0.002},
            "refine": {
                "type": "linear_relative",
                "radius": 0.01,
                "step": 0.001,
                "clip_min": 0.2,
                "clip_max": 0.4,
            },
        },
    },
}

DEFAULT_PARAM_ORDER: Dict[str, Tuple[str, ...]] = {
    "pbuf": ("H0", "Om0", "alpha", "k_sat", "Rmax"),
    "lcdm": ("H0", "Om0"),
}

DEFAULT_SECOND_PASS_PARAMS: Dict[str, Tuple[str, ...]] = {
    "pbuf": ("H0", "Om0", "k_sat", "alpha", "Rmax"),
    "lcdm": ("H0", "Om0"),
}

DEFAULT_COUPLED_GROUPS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "pbuf": (
        ("H0", "Om0"),
        ("k_sat", "alpha"),
    ),
    "lcdm": (
        ("H0", "Om0"),
    ),
}

# Convenience aliases for callers that only work with a single model family
DEFAULT_PBUF_REFERENCE: Dict[str, float] = DEFAULT_REFERENCES["pbuf"]
DEFAULT_LCDM_REFERENCE: Dict[str, float] = DEFAULT_REFERENCES["lcdm"]

# Backwards compatibility export (legacy callers may import this name directly)
SECOND_PASS_PARAMS: Tuple[str, ...] = DEFAULT_SECOND_PASS_PARAMS["pbuf"]
OPTIMIZER_VERSION = "coord-opt-v2"
DEFAULT_DELTA_CHI2 = 20.0


@dataclass(frozen=True)
class ScanCurvePoint:
    value: float
    chi2: Optional[float]
    valid: bool
    passes_phase6a: bool
    score: Optional[float] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "value": float(self.value),
            "valid": bool(self.valid),
            "passes_phase6a": bool(self.passes_phase6a),
        }
        if self.chi2 is not None:
            payload["chi2"] = float(self.chi2)
        else:
            payload["chi2"] = None
        if self.score is not None:
            payload["score"] = float(self.score)
        if self.rejection_reason is not None:
            payload["rejection_reason"] = self.rejection_reason
        return payload


def _linear_range(start: float, stop: float, step: float) -> List[float]:
    if step <= 0.0:
        raise ValueError("Step must be positive for linear range.")
    if stop < start:
        start, stop = stop, start
    values = np.arange(start, stop + 0.5 * step, step)
    return [float(v) for v in values]


def _linear_relative(center: float, radius: float, step: float, clip_min: Optional[float], clip_max: Optional[float]) -> List[float]:
    if radius <= 0.0 or step <= 0.0:
        return [float(center)]
    start = center - radius
    stop = center + radius
    if clip_min is not None:
        start = max(start, clip_min)
    if clip_max is not None:
        stop = min(stop, clip_max)
    if start > stop:
        start = stop = center
    return _linear_range(start, stop, step)


def _log_relative(center: float, factors: Sequence[float], clip_min: Optional[float], clip_max: Optional[float]) -> List[float]:
    if center <= 0.0:
        raise ValueError("Center must be positive for log-relative range.")
    values = []
    for factor in factors:
        candidate = center * factor
        if clip_min is not None:
            candidate = max(candidate, clip_min)
        if clip_max is not None:
            candidate = min(candidate, clip_max)
        values.append(candidate)
    return _unique_sorted(values)


def _unique_sorted(values: Iterable[float]) -> List[float]:
    seen: set[float] = set()
    ordered: List[float] = []
    for value in sorted(values):
        key = round(float(value), 12)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(float(value))
    return ordered


def _normalize_float_dict(values: Mapping[str, float]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in values.items():
        normalized[key] = float(value)
    return normalized


def _params_key(params: Mapping[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((key, round(float(value), 12)) for key, value in params.items()))


def _normalize_datasets(datasets: Sequence[str]) -> List[str]:
    if not datasets:
        raise ValueError("At least one dataset must be provided.")
    available = set(list_available_datasets())
    normalized: List[str] = []
    seen: set[str] = set()
    for raw in datasets:
        name = raw.strip().lower()
        if not name:
            continue
        if name not in available:
            raise ValueError(f"Unknown dataset '{raw}'. Available: {sorted(available)}")
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


class CoordinateBasinWalker:
    """
    Coordinate descent optimizer that maps 1D χ² basins for cosmological parameters.
    """

    def __init__(
        self,
        model_type: str,
        datasets: Sequence[str],
        *,
        enforce_phase6a: bool = True,
        delta_chi2: float = DEFAULT_DELTA_CHI2,
        reference_params: Optional[Mapping[str, float]] = None,
        scan_presets: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
        param_order: Optional[Sequence[str]] = None,
        second_pass_params: Optional[Sequence[str]] = None,
        verbose: bool = False,
        progress: bool = False,
        debug_rejections: bool = True,
        max_workers: Optional[int] = None,
        improvement_tol: float = 1.0e-2,
        param_shift_tol: float = 1.0e-3,
        max_cycles: int = 6,
        chi2_targets: Optional[Chi2TargetRegistry] = None,
        priors: Optional[Mapping[str, Mapping[str, Any]]] = None,
        walker_settings: Optional[Mapping[str, Any]] = None,
        observers: Optional[Sequence[BasinWalkObserver]] = None,
    ) -> None:
        self.model_type = model_type.lower()
        if self.model_type not in DEFAULT_REFERENCES:
            raise ValueError(
                f"Unsupported model '{model_type}'. "
                f"Supported models: {', '.join(sorted(DEFAULT_REFERENCES))}"
            )
        self.datasets = _normalize_datasets(datasets)
        self.enforce_phase6a = bool(enforce_phase6a)
        self.delta_chi2 = float(delta_chi2)
        self.reference_params = dict(reference_params or DEFAULT_REFERENCES[self.model_type])
        self.scan_presets = dict(scan_presets or DEFAULT_SCAN_PRESETS[self.model_type])
        default_order = DEFAULT_PARAM_ORDER[self.model_type]
        self.param_order = tuple(param_order) if param_order is not None else default_order
        if second_pass_params is None:
            self.second_pass_params = DEFAULT_SECOND_PASS_PARAMS[self.model_type]
        else:
            self.second_pass_params = tuple(second_pass_params)
        self.verbose = bool(verbose)
        self._progress_requested = bool(progress)
        self.progress = bool(progress and (tqdm is not None))
        self.debug_rejections = bool(debug_rejections)
        self._cache: Dict[Tuple[Tuple[str, float], ...], Dict[str, Any]] = {}
        self._max_workers_requested = max_workers
        self.max_workers = self._resolve_worker_count(max_workers)
        self.parallel = bool(self.max_workers > 1)
        self.improvement_tol = self._sanitize_improvement_tol(improvement_tol)
        self.param_shift_tol = self._sanitize_param_shift_tol(param_shift_tol)
        self.max_cycles = max(1, int(max_cycles))
        self._last_axis_scans: List[Dict[str, Any]] = []
        self._last_result: Optional[Dict[str, Any]] = None
        if len(self.param_order) > 1:
            self.primary_rescan_params = tuple(self.param_order[:2])
        else:
            self.primary_rescan_params = tuple()
        self._chi2_registry = chi2_targets if chi2_targets and not chi2_targets.is_empty() else None
        self._score_delta = float(self._chi2_registry.delta) if self._chi2_registry else float(self.delta_chi2)
        self.priors = {key: dict(value) for key, value in (priors or {}).items()}
        self._walker_settings = dict(walker_settings or {})
        self._force_convergence = bool(self._walker_settings.get("converge"))
        self._max_rebalances = self._sanitize_rebalance_limit(self._walker_settings.get("max_rebalances"))
        self._rebalance_count = 0
        self._reseed_on_plateau = bool(self._walker_settings.get("reseed_on_plateau"))
        self._plateau_delta = float(self._walker_settings.get("plateau_delta", 1.0) or 1.0)
        window = self._walker_settings.get("plateau_window", 3)
        try:
            self._plateau_window = max(1, int(window))
        except (TypeError, ValueError):
            self._plateau_window = 3
        self._plateau_rng = random.Random(self._walker_settings.get("plateau_seed"))
        self._convergence_forced = False
        self._adaptive_windows: Dict[str, Dict[str, Any]] = {}
        self._scan_limits: Dict[Tuple[str, int], Tuple[Optional[float], Optional[float]]] = {}
        self._observers: List[BasinWalkObserver] = []
        self._active_run_id: Optional[str] = None
        coupled_override = self._walker_settings.get("coupled_params")
        if isinstance(coupled_override, Sequence) and coupled_override:
            normalized_pairs: List[Tuple[str, ...]] = []
            for entry in coupled_override:
                if isinstance(entry, Sequence) and entry:
                    normalized_pairs.append(tuple(str(item) for item in entry))
            self._coupled_pairs: Tuple[Tuple[str, ...], ...] = tuple(normalized_pairs) or DEFAULT_COUPLED_GROUPS.get(self.model_type, tuple())
        else:
            self._coupled_pairs = DEFAULT_COUPLED_GROUPS.get(self.model_type, tuple())
        self._max_edge_expansions = max(1, int(self._walker_settings.get("edge_expansion_limit", EDGE_EXPANSION_LIMIT)))
        plateau_samples = self._walker_settings.get("plateau_samples", 5)
        try:
            self._plateau_samples = max(0, int(plateau_samples))
        except (TypeError, ValueError):
            self._plateau_samples = 5
        adaptive_decay = self._walker_settings.get("adaptive_decay", 0.5)
        try:
            self._adaptive_decay = self._clamp(float(adaptive_decay), 0.0, 1.0)
        except (TypeError, ValueError):
            self._adaptive_decay = 0.5
        if observers:
            for observer in observers:
                self.add_observer(observer)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _debug(self, message: str) -> None:
        if self.debug_rejections:
            print(message)

    def add_observer(self, observer: BasinWalkObserver) -> None:
        if observer in self._observers:
            return
        self._observers.append(observer)
        observer.on_attach(self)

    def remove_observer(self, observer: BasinWalkObserver) -> None:
        if observer not in self._observers:
            return
        self._observers.remove(observer)
        observer.on_detach(self)

    def _notify_run_started(self, mode: str) -> None:
        if not self._observers:
            return
        self._active_run_id = uuid.uuid4().hex
        context = {
            "run_id": self._active_run_id,
            "mode": mode,
            "model": self.model_type,
            "datasets": list(self.datasets),
            "enforce_phase6a": self.enforce_phase6a,
            "delta_chi2": self.delta_chi2,
            "timestamp": datetime.now(UTC).isoformat(),
            "walker_settings": dict(self._walker_settings),
            "reference_params": dict(self.reference_params),
            "param_order": list(self.param_order),
            "second_pass_params": list(self.second_pass_params),
            "max_workers": self.max_workers,
        }
        for observer in self._observers:
            observer.on_run_started(self, context)

    def _notify_scan_completed(self, summary: Dict[str, Any]) -> None:
        if not self._observers:
            return
        for observer in self._observers:
            observer.on_scan_completed(self, summary)

    def _notify_coupled_update(self, summary: Dict[str, Any]) -> None:
        if not self._observers:
            return
        for observer in self._observers:
            observer.on_coupled_update(self, summary)

    def _notify_plateau_reseed(self, summary: Dict[str, Any]) -> None:
        if not self._observers:
            return
        for observer in self._observers:
            observer.on_plateau_reseed(self, summary)

    def _notify_island_center(self, payload: Dict[str, Any]) -> None:
        if not self._observers:
            return
        for observer in self._observers:
            observer.on_island_center(self, payload)

    def _notify_run_completed(self, result: Dict[str, Any]) -> None:
        if not self._observers:
            return
        for observer in self._observers:
            observer.on_run_completed(self, result)
        self._active_run_id = None

    def _resolve_worker_count(self, requested: Optional[int]) -> int:
        if requested is None:
            return MAX_WORKERS
        try:
            value = int(requested)
        except (TypeError, ValueError):
            return MAX_WORKERS
        if value <= 1:
            return 1
        return max(1, min(value, MAX_WORKERS))

    @staticmethod
    def _sanitize_improvement_tol(tol: float) -> float:
        try:
            value = float(tol)
        except (TypeError, ValueError):
            return 1.0e-2
        if value <= 0.0:
            return 1.0e-2
        return value

    @staticmethod
    def _sanitize_param_shift_tol(tol: float) -> float:
        try:
            value = float(tol)
        except (TypeError, ValueError):
            return 1.0e-3
        if value <= 0.0:
            return 1.0e-3
        return value

    @staticmethod
    def _sanitize_rebalance_limit(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            result = int(value)
        except (TypeError, ValueError):
            return None
        if result <= 0:
            return None
        return result

    def _iter_with_progress(self, values: Sequence[float], param: str, pass_id: int):
        if not self.progress or not values:
            for value in values:
                yield value
            return
        desc = f"⚙️ {param} (pass {pass_id})"
        for value in tqdm(values, desc=desc, leave=False):
            yield value

    def _diagnose_rejection(
        self,
        evaluation: Mapping[str, Any],
        *,
        passes_phase6a: bool,
    ) -> str:
        if not isinstance(evaluation, Mapping):
            return f"non-mapping evaluation: {type(evaluation).__name__}"

        reasons: List[str] = []
        status = evaluation.get("status")
        if status and status != "valid":
            reasons.append(f"status={status}")

        chi2_total = evaluation.get("chi2_total")
        if chi2_total is None:
            reasons.append("chi2_total=None")
        elif not isinstance(chi2_total, (int, float)) or not math.isfinite(float(chi2_total)):
            reasons.append(f"chi2_total={chi2_total}")

        if self.enforce_phase6a and self.model_type == "pbuf" and not passes_phase6a:
            reasons.append("phase6a_failed")

        validation = evaluation.get("validation")
        if isinstance(validation, Mapping) and not validation.get("valid", True):
            reason = validation.get("reason")
            if reason:
                reasons.append(f"validation={reason}")
            else:
                reasons.append("validation_failed")

        dataset_errors = evaluation.get("dataset_errors")
        if isinstance(dataset_errors, Mapping) and dataset_errors:
            details = ", ".join(f"{k}: {dataset_errors[k]}" for k in list(dataset_errors)[:3])
            if len(dataset_errors) > 3:
                details += ", …"
            reasons.append(f"dataset_errors={details}")

        metadata = evaluation.get("metadata")
        if isinstance(metadata, Mapping):
            meta_error = metadata.get("error")
            if meta_error:
                reasons.append(f"metadata_error={meta_error}")

        error = evaluation.get("error")
        if error and error not in reasons:
            reasons.append(f"error={error}")

        if not reasons:
            return "rejected_without_detail"
        return "; ".join(reasons)

    def _evaluate_point(
        self,
        trial_params: Mapping[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, Any], bool, Optional[float], bool]:
        params_dict = {key: float(value) for key, value in trial_params.items()}
        try:
            evaluation = self._evaluate(params_dict)
        except Exception as exc:  # pragma: no cover - defensive guard
            evaluation = {
                "status": "error",
                "error": repr(exc),
            }
        chi2_total = evaluation.get("chi2_total")
        chi2_value: Optional[float] = None
        if isinstance(chi2_total, (int, float)):
            chi2_value = float(chi2_total)
            if not math.isfinite(chi2_value):
                chi2_value = None
        passes_phase6a = bool(evaluation.get("passes_phase6a", True))
        status = evaluation.get("status")
        valid = bool(
            status == "valid"
            and chi2_value is not None
        )
        if self.enforce_phase6a and self.model_type == "pbuf":
            valid = valid and passes_phase6a
        return params_dict, evaluation, valid, chi2_value, passes_phase6a

    def _score_evaluation(self, evaluation: Mapping[str, Any], chi2_value: Optional[float]) -> Optional[float]:
        if self._chi2_registry is None:
            return chi2_value
        return self._chi2_registry.score(evaluation, chi2_value)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        if value < low:
            return low
        if value > high:
            return high
        return value

    @staticmethod
    def _sample_k_sat(rng: random.Random, left: float, right: float) -> float:
        if right <= left:
            return left
        # Prefer the SH0ES-anchored region near unity, restricting to [0.95, 1.0].
        target_left = max(0.95, left)
        target_right = min(1.0, right)
        if target_right > target_left:
            base = rng.betavariate(6.0, 1.0)
            return target_left + (target_right - target_left) * base
        return rng.uniform(left, right)

    def _sample_rmax(self, rng: random.Random, left: float, right: float) -> float:
        if right <= left:
            return left
        safe_left = max(left, 1e-12)
        safe_right = max(right, safe_left * (1.0 + 1e-9))
        log_left = math.log10(safe_left)
        log_right = math.log10(safe_right)
        mean = 6.9
        sigma = 0.2
        for _ in range(64):
            draw = rng.gauss(mean, sigma)
            if log_left <= draw <= log_right:
                value = 10.0 ** draw
                if left <= value <= right:
                    return value
        fallback = 10.0 ** mean
        return self._clamp(fallback, left, right)

    def _sample_parameter_value(self, param: str, rng: random.Random, left: float, right: float) -> float:
        if right <= left:
            return left
        param_lower = param.lower()
        if param_lower == "k_sat":
            return self._sample_k_sat(rng, left, right)
        if param_lower == "rmax":
            return self._sample_rmax(rng, left, right)
        return rng.uniform(left, right)

    def _extract_parameter_ranges(
        self,
        axis_scans: Sequence[Mapping[str, Any]],
        fiducial_params: Mapping[str, float],
    ) -> Dict[str, Dict[str, float]]:
        ranges: Dict[str, Dict[str, float]] = {}
        for entry in axis_scans:
            param = entry.get("param")
            if not isinstance(param, str):
                continue
            best = entry.get("best")
            left_edge = entry.get("left_edge")
            right_edge = entry.get("right_edge")
            if best is None or left_edge is None or right_edge is None:
                continue
            try:
                best_f = float(best)
                left_f = float(left_edge)
                right_f = float(right_edge)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(best_f) or not math.isfinite(left_f) or not math.isfinite(right_f):
                continue
            if right_f <= left_f:
                continue
            delta = 0.5 * (right_f - left_f)
            if delta <= 0.0:
                continue
            ranges[param] = {
                "best": best_f,
                "left": left_f,
                "right": right_f,
                "delta": delta,
                "pass": entry.get("pass"),
                "cycle": entry.get("cycle"),
            }
        # Ensure every fiducial parameter we scanned has a fallback range.
        tracked_params = set(self.param_order) | set(getattr(self, "second_pass_params", ()))
        for param, value in fiducial_params.items():
            if param in ranges:
                continue
            if param not in tracked_params:
                continue
            try:
                center = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(center):
                continue
            span = max(abs(center), 1.0) * 0.05 or 0.05
            ranges[param] = {
                "best": center,
                "left": center - span,
                "right": center + span,
                "delta": span,
                "pass": None,
                "cycle": None,
                "fallback": True,
            }
        return ranges

    @staticmethod
    def _scaled_distance(
        a: Mapping[str, float],
        b: Mapping[str, float],
        scales: Mapping[str, float],
    ) -> float:
        total = 0.0
        for param, scale in scales.items():
            denom = max(scale, 1e-12)
            av = float(a.get(param, 0.0))
            bv = float(b.get(param, 0.0))
            total += ((av - bv) / denom) ** 2
        return math.sqrt(total)

    def _relative_param_shift(
        self,
        previous: Mapping[str, float],
        current: Mapping[str, float],
    ) -> float:
        combined_order = tuple(
            dict.fromkeys(
                list(self.param_order)
                + list(getattr(self, "second_pass_params", ()))
            )
        )
        if not combined_order:
            combined_order = tuple(sorted(set(previous) | set(current)))
        total_shift = 0.0
        for param in combined_order:
            prev_val = float(previous.get(param, 0.0))
            curr_val = float(current.get(param, prev_val))
            if not math.isfinite(prev_val):
                prev_val = 0.0
            if not math.isfinite(curr_val):
                curr_val = prev_val
            denom = max(abs(prev_val), 1.0e-9)
            total_shift += abs(curr_val - prev_val) / denom
        return total_shift

    def _should_run_rebalance(self) -> bool:
        if self._max_rebalances is None:
            return True
        if self._rebalance_count < self._max_rebalances:
            self._rebalance_count += 1
            return True
        return False

    def _apply_plateau_reseed(
        self,
        params: MutableMapping[str, float],
        axis_scans: List[Dict[str, Any]],
        cycle_index: int,
    ) -> bool:
        ranges = self._extract_parameter_ranges(axis_scans, params)
        if not ranges:
            return False
        rng = self._plateau_rng or random.Random()
        sample_count = max(1, self._plateau_samples)

        base_trial = dict(params)
        (
            base_normalized,
            base_evaluation,
            base_valid,
            base_chi2,
            base_passes_phase6a,
        ) = self._evaluate_point(base_trial)
        self._cache[_params_key(base_normalized)] = base_evaluation

        base_score = None
        if base_valid and base_chi2 is not None:
            base_score = self._score_evaluation(base_evaluation, base_chi2)
            if base_score is None:
                base_score = base_chi2

        candidates: List[Dict[str, Any]] = []

        for _ in range(sample_count):
            candidate: Dict[str, float] = dict(params)
            for param_name, bounds in ranges.items():
                left = float(bounds.get("left", candidate.get(param_name, 0.0)))
                right = float(bounds.get("right", candidate.get(param_name, left)))
                if not math.isfinite(left) or not math.isfinite(right):
                    continue
                sampled = self._sample_parameter_value(param_name, rng, left, right)
                if math.isfinite(sampled):
                    candidate[param_name] = float(sampled)
            candidates.append(candidate)

        best_candidate = dict(params)
        best_eval = base_evaluation
        best_score = base_score
        best_chi2 = base_chi2
        best_valid = base_valid and base_chi2 is not None
        best_passes = base_passes_phase6a

        for candidate in candidates:
            normalized, evaluation, valid, chi2_value, passes_phase6a = self._evaluate_point(candidate)
            self._cache[_params_key(normalized)] = evaluation
            if not valid or chi2_value is None:
                continue
            score_value = self._score_evaluation(evaluation, chi2_value)
            if score_value is None:
                score_value = chi2_value
            if best_score is None or score_value < best_score:
                best_candidate = dict(candidate)
                best_eval = evaluation
                best_score = float(score_value)
                best_chi2 = float(chi2_value)
                best_valid = True
                best_passes = bool(passes_phase6a)

        improved = best_valid and best_score is not None and (base_score is None or best_score + 1e-6 < base_score)
        if improved:
            params.update(best_candidate)

        reseed_summary = {
            "type": "plateau_reseed",
            "cycle": cycle_index,
            "num_samples": len(candidates),
            "improved": improved,
            "best_score": best_score,
            "base_score": base_score,
            "passes_phase6a": best_passes,
        }
        if best_chi2 is not None:
            reseed_summary["best_chi2"] = best_chi2
        if base_chi2 is not None:
            reseed_summary["base_chi2"] = base_chi2
        axis_scans.append(reseed_summary)
        self._notify_plateau_reseed(reseed_summary)

        return improved

    def find_island_center(
        self,
        run_result: Mapping[str, Any],
        *,
        num_samples: int = 200,
        chi2_delta: float = 20.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive to search for an island center.")
        try:
            chi2_delta = float(chi2_delta)
        except (TypeError, ValueError):
            raise ValueError("chi2_delta must be numeric.") from None
        if chi2_delta <= 0.0:
            raise ValueError("chi2_delta must be positive.")

        axis_scans = list(run_result.get("axis_scans") or [])
        if not axis_scans:
            raise ValueError("No axis scan data available to derive island bounds.")

        fiducial_params = dict(run_result.get("fiducial_params") or self.reference_params)
        if not fiducial_params:
            raise ValueError("No fiducial parameters available for island search.")

        parameter_ranges = self._extract_parameter_ranges(axis_scans, fiducial_params)
        if not parameter_ranges:
            raise ValueError("Unable to derive parameter ranges for island sampling.")

        param_order = [param for param in self.param_order if param in parameter_ranges]
        if not param_order:
            param_order = list(parameter_ranges.keys())

        rng = random.Random()
        if seed is not None:
            rng.seed(int(seed))

        base_template = dict(self.reference_params)
        base_template.update(fiducial_params)

        samples: List[Dict[str, float]] = []
        samples.append({param: float(base_template.get(param, parameter_ranges[param]["best"])) for param in parameter_ranges})

        while len(samples) < num_samples:
            candidate: Dict[str, float] = {}
            for param, bounds in parameter_ranges.items():
                left = float(bounds["left"])
                right = float(bounds["right"])
                candidate[param] = self._sample_parameter_value(param, rng, left, right)
            samples.append(candidate)

        # Evaluate samples (including fiducial anchor)
        evaluated: List[Dict[str, Any]] = []
        trials: List[Dict[str, float]] = []
        for coords in samples:
            trial = dict(base_template)
            trial.update(coords)
            trials.append(trial)

        def _evaluate_trial(trial_params: Mapping[str, float]):
            normalized_params, evaluation, valid, chi2_value, passes_phase6a = self._evaluate_point(trial_params)
            self._cache[_params_key(normalized_params)] = evaluation
            record = {
                "params": normalized_params,
                "valid": bool(valid),
                "passes_phase6a": bool(passes_phase6a),
                "chi2": chi2_value if isinstance(chi2_value, (int, float)) else None,
                "status": evaluation.get("status") if isinstance(evaluation, Mapping) else None,
            }
            return record

        if self.parallel and len(trials) > 1:
            futures: Dict[Any, int] = {}
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for idx, trial in enumerate(trials):
                    futures[executor.submit(self._evaluate_point, trial)] = idx
                progress_bar = None
                if self.progress:
                    progress_bar = tqdm(total=len(futures), desc="⚙️ island sampling", leave=False)
                try:
                    for future in as_completed(futures):
                        idx = futures[future]
                        normalized, evaluation, valid, chi2_value, passes_phase6a = future.result()
                        if progress_bar is not None:
                            progress_bar.update(1)
                        self._cache[_params_key(normalized)] = evaluation
                        evaluated.append(
                            {
                                "params": normalized,
                                "valid": bool(valid),
                                "passes_phase6a": bool(passes_phase6a),
                                "chi2": chi2_value if isinstance(chi2_value, (int, float)) else None,
                                "status": evaluation.get("status") if isinstance(evaluation, Mapping) else None,
                            }
                        )
                finally:
                    if progress_bar is not None:
                        progress_bar.close()
        else:
            for trial in trials:
                evaluated.append(_evaluate_trial(trial))

        viable = [
            sample
            for sample in evaluated
            if sample["valid"] and isinstance(sample["chi2"], (int, float)) and math.isfinite(sample["chi2"])
        ]
        if not viable:
            raise ValueError("Island sampling produced no valid points.")

        chi2_min = min(sample["chi2"] for sample in viable)
        chi2_threshold = chi2_min + chi2_delta

        core = [sample for sample in viable if sample["chi2"] <= chi2_threshold]
        if not core:
            raise ValueError("No core samples within the requested Δχ² threshold.")

        scales = {param: max(parameter_ranges[param]["delta"], 1e-8) for param in param_order}

        if len(core) == 1:
            core[0]["avg_distance"] = 0.0
        else:
            for sample in core:
                distance_sum = 0.0
                for other in core:
                    if other is sample:
                        continue
                    distance_sum += self._scaled_distance(sample["params"], other["params"], scales)
                sample["avg_distance"] = distance_sum / (len(core) - 1)

        core.sort(key=lambda sample: (sample.get("avg_distance", float("inf")), sample["chi2"]))

        def _verify_candidate(candidate: Mapping[str, Any]) -> bool:
            candidate_params = dict(base_template)
            candidate_params.update(candidate["params"])
            candidate_valid = candidate["valid"] and isinstance(candidate.get("chi2"), (int, float))
            if not candidate_valid:
                return False
            for param in param_order:
                delta_half = 0.5 * scales[param]
                if delta_half <= 0.0:
                    continue
                for direction in (-1.0, 1.0):
                    perturbed = dict(candidate_params)
                    target = perturbed[param] + direction * delta_half
                    bounds = parameter_ranges[param]
                    target = self._clamp(target, bounds["left"], bounds["right"])
                    if target == perturbed[param]:
                        continue
                    perturbed[param] = target
                    normalized, evaluation, valid, chi2_value, passes_phase6a = self._evaluate_point(perturbed)
                    self._cache[_params_key(normalized)] = evaluation
                    if not valid or chi2_value is None or not math.isfinite(chi2_value):
                        return False
                    if chi2_value > chi2_threshold:
                        return False
                    if self.enforce_phase6a and self.model_type == "pbuf" and not passes_phase6a:
                        return False
            return True

        chosen: Optional[Dict[str, Any]] = None
        for candidate in core:
            if _verify_candidate(candidate):
                chosen = candidate
                break
        if chosen is None:
            chosen = core[0]

        ranges_payload = {
            param: {
                "left": float(bounds["left"]),
                "right": float(bounds["right"]),
                "delta": float(bounds["delta"]),
            }
            for param, bounds in parameter_ranges.items()
        }

        core_chi2_values = [sample["chi2"] for sample in core]
        center_params = dict(base_template)
        center_params.update(chosen["params"])

        island_payload: Dict[str, Any] = {
            "method": "most_central",
            "seed": seed,
            "num_samples": len(evaluated),
            "num_viable": len(viable),
            "num_core": len(core),
            "chi2_min": chi2_min,
            "chi2_threshold": chi2_threshold,
            "center_params": {key: float(value) for key, value in center_params.items()},
            "center_chi2": float(chosen["chi2"]),
            "center_avg_distance": float(chosen.get("avg_distance", 0.0)),
            "ranges": ranges_payload,
            "core_stats": {
                "chi2_min": float(min(core_chi2_values)),
                "chi2_max": float(max(core_chi2_values)),
                "chi2_mean": float(sum(core_chi2_values) / len(core_chi2_values)),
            },
        }

        self._notify_island_center(island_payload)
        return island_payload

    def run(self) -> Dict[str, Any]:
        """
        Execute the basin walk and return a JSON-serialisable payload.
        """
        if self._force_convergence and not self._convergence_forced:
            self._convergence_forced = True
            try:
                return self.run_with_convergence()
            finally:
                self._convergence_forced = False

        params: Dict[str, float] = _normalize_float_dict(self.reference_params)
        axis_scans: List[Dict[str, Any]] = []
        self._adaptive_windows.clear()
        self._scan_limits.clear()

        self._log("▶️  Starting coordinate basin walk")
        self._log(f"   Model: {self.model_type}")
        self._log(f"   Datasets: {', '.join(self.datasets)}")
        if self._progress_requested and not self.progress:
            self._log("   (Requested progress bars but tqdm is unavailable; falling back to plain iteration)")

        self._notify_run_started("run")

        # First pass
        for param in self.param_order:
            scan_values = self._make_scan_range(param, params, pass_id=1)
            self._log(
                f"→ Pass 1: scanning {param} over {len(scan_values)} points "
                f"[range {scan_values[0]:.6g} .. {scan_values[-1]:.6g}]"
            )
            scan_reports = self._scan_axis_with_edge_rescan(param, params, scan_values, pass_id=1)
            for report in scan_reports:
                axis_scans.append(report)
                self._notify_scan_completed(report)
            final_report = scan_reports[-1]
            if final_report.get("best") is not None:
                params[param] = float(final_report["best"])
                score_min = final_report.get("score_min")
                score_suffix = ""
                if score_min is not None and self._chi2_registry is not None:
                    score_suffix = f", score={float(score_min):.3f}"
                self._log(
                    f"   ↳ best {param} = {final_report['best']:.6g}, "
                    f"χ² = {final_report.get('chi2_min', float('nan')):.3f}{score_suffix}"
                )
            else:
                self._log(f"   ↳ no valid points for {param}")

        # Optional refinement
        for param in self.second_pass_params:
            scan_values = self._make_scan_range(param, params, pass_id=2)
            self._log(
                f"→ Pass 2: tightening {param} with {len(scan_values)} points "
                f"[range {scan_values[0]:.6g} .. {scan_values[-1]:.6g}]"
            )
            scan_reports = self._scan_axis_with_edge_rescan(param, params, scan_values, pass_id=2)
            for report in scan_reports:
                axis_scans.append(report)
                self._notify_scan_completed(report)
            final_report = scan_reports[-1]
            if final_report.get("best") is not None:
                params[param] = float(final_report["best"])
                score_min = final_report.get("score_min")
                score_suffix = ""
                if score_min is not None and self._chi2_registry is not None:
                    score_suffix = f", score={float(score_min):.3f}"
                self._log(
                    f"   ↳ best {param} = {final_report['best']:.6g}, "
                    f"χ² = {final_report.get('chi2_min', float('nan')):.3f}{score_suffix}"
                )
            else:
                self._log(f"   ↳ no valid points for {param}")

        # Final rebalancing of primary parameters after secondary tweaks
        if getattr(self, "primary_rescan_params", ()) and self._should_run_rebalance():
            for param in self.primary_rescan_params:
                scan_values = self._make_scan_range(param, params, pass_id=3)
                self._log(
                    f"→ Pass 3: rebalancing {param} over {len(scan_values)} points "
                    f"[range {scan_values[0]:.6g} .. {scan_values[-1]:.6g}]"
                )
                scan_reports = self._scan_axis_with_edge_rescan(param, params, scan_values, pass_id=3)
                for report in scan_reports:
                    axis_scans.append(report)
                    self._notify_scan_completed(report)
                final_report = scan_reports[-1]
                if final_report.get("best") is not None:
                    params[param] = float(final_report["best"])
                    score_min = final_report.get("score_min")
                    score_suffix = ""
                    if score_min is not None and self._chi2_registry is not None:
                        score_suffix = f", score={float(score_min):.3f}"
                    self._log(
                        f"   ↳ tuned {param} = {final_report['best']:.6g}, "
                        f"χ² = {final_report.get('chi2_min', float('nan')):.3f}{score_suffix}"
                    )
            else:
                self._log(f"   ↳ no valid points for {param}")

        if self._coupled_pairs:
            self._run_coupled_updates(params, axis_scans, cycle=None)

        fiducial = _normalize_float_dict(params)
        fiducial_eval = self._evaluate(fiducial)
        fiducial_chi2: Optional[float] = None
        fiducial_passes = None
        if fiducial_eval.get("status") == "valid" and math.isfinite(fiducial_eval.get("chi2_total", math.inf)):
            fiducial_chi2 = float(fiducial_eval["chi2_total"])
            fiducial_passes = bool(fiducial_eval.get("passes_phase6a", True))
            self._log(f"✔️  Fiducial χ²_total = {fiducial_chi2:.3f}")
        else:
            self._log("⚠️  Fiducial evaluation invalid or non-finite χ²")

        payload: Dict[str, Any] = {
            "version": OPTIMIZER_VERSION,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "model_type": self.model_type,
            "datasets_used": list(self.datasets),
            "phase6a_enforced": self.enforce_phase6a,
            "delta_chi2_tolerance": self.delta_chi2,
            "fiducial_params": fiducial,
            "axis_scans": axis_scans,
        }

        if fiducial_chi2 is not None:
            payload["fiducial_chi2"] = fiducial_chi2
        if fiducial_passes is not None:
            payload["fiducial_passes_phase6a"] = fiducial_passes
        if "chi2_breakdown" in fiducial_eval:
            payload["fiducial_breakdown"] = {
                name: float(value) for name, value in fiducial_eval["chi2_breakdown"].items()
            }
        if "validation" in fiducial_eval:
            payload["fiducial_validation"] = fiducial_eval["validation"]
        if self._chi2_registry is not None:
            payload["chi2_targets"] = self._chi2_registry.describe()

        self._last_axis_scans = [dict(scan) for scan in axis_scans]
        self._last_result = json.loads(json.dumps(payload))
        self._notify_run_completed(payload)

        return payload

    def run_with_convergence(self) -> Dict[str, Any]:
        """
        Execute iterative coordinate sweeps until the χ² improvement stalls or the cycle
        budget is exhausted.
        """
        params: Dict[str, float] = _normalize_float_dict(self.reference_params)
        axis_scans: List[Dict[str, Any]] = []
        cycle_summaries: List[Dict[str, Any]] = []
        chi2_history: List[Optional[float]] = []
        param_shift_history: List[Optional[float]] = []
        self._adaptive_windows.clear()
        self._scan_limits.clear()
        self._notify_run_started("convergence")

        last_valid_chi2 = math.inf
        converged = False

        final_fiducial: Dict[str, float] = dict(params)
        final_eval: Dict[str, Any] = {}
        final_chi2: Optional[float] = None
        final_passes: Optional[bool] = None
        previous_cycle_params: Dict[str, float] = dict(params)

        refine_params: Tuple[str, ...] = (
            tuple(self.second_pass_params) if self.second_pass_params else tuple(self.param_order)
        )
        stagnation_buffer: List[float] = []

        for cycle_index in range(self.max_cycles):
            self._log(f"🔁 Cycle {cycle_index + 1}/{self.max_cycles}")
            stage_records: List[Dict[str, Any]] = []

            def run_stage(param_list: Sequence[str], stage_label: str, *, pass_id_override: Optional[int] = None) -> Dict[str, Any]:
                stage_pass_id = pass_id_override
                if stage_pass_id is None:
                    stage_pass_id = 1 if stage_label == "coarse" and cycle_index == 0 else 2
                if not param_list:
                    return {
                        "stage": stage_label,
                        "pass_id": stage_pass_id,
                        "scan_indices": [],
                        "best_parameters": {},
                    }
                scan_indices: List[int] = []
                best_parameters: Dict[str, Optional[float]] = {}
                for param in param_list:
                    scan_values = self._make_scan_range(param, params, pass_id=stage_pass_id)
                    if scan_values:
                        self._log(
                            f"→ Cycle {cycle_index} [{stage_label}] scanning {param} over {len(scan_values)} points "
                            f"[range {scan_values[0]:.6g} .. {scan_values[-1]:.6g}]"
                        )
                    scan_reports = self._scan_axis_with_edge_rescan(
                        param,
                        params,
                        scan_values,
                        pass_id=stage_pass_id,
                        cycle=cycle_index,
                    )
                    for report in scan_reports:
                        axis_scans.append(report)
                        self._notify_scan_completed(report)
                        scan_indices.append(len(axis_scans) - 1)
                    final_report = scan_reports[-1]
                    best = final_report.get("best")
                    if best is not None:
                        params[param] = float(best)
                        best_parameters[param] = float(best)
                        score_min = final_report.get("score_min")
                        score_suffix = ""
                        if score_min is not None and self._chi2_registry is not None:
                            score_suffix = f", score={float(score_min):.3f}"
                        self._log(
                            f"   ↳ best {param} = {float(best):.6g}, "
                            f"χ² = {final_report.get('chi2_min', float('nan')):.3f}{score_suffix}"
                        )
                    else:
                        best_parameters[param] = None
                        self._log(f"   ↳ no valid points for {param}")
                return {
                    "stage": stage_label,
                    "pass_id": stage_pass_id,
                    "scan_indices": scan_indices,
                    "best_parameters": best_parameters,
                }

            if cycle_index == 0:
                stage_records.append(run_stage(self.param_order, "coarse"))
                stage_records.append(run_stage(refine_params, "refine"))
            else:
                stage_records.append(run_stage(refine_params, "refine"))

            if getattr(self, "primary_rescan_params", ()) and self._should_run_rebalance():
                stage_records.append(
                    run_stage(self.primary_rescan_params, "primary", pass_id_override=3)
                )

            if self._coupled_pairs:
                coupled_summaries = self._run_coupled_updates(params, axis_scans, cycle=cycle_index)
                if coupled_summaries:
                    stage_records.append(
                        {
                            "stage": "coupled",
                            "pass_id": None,
                            "scan_indices": list(range(len(axis_scans) - len(coupled_summaries), len(axis_scans))),
                            "best_parameters": {
                                ",".join(summary["parameters"]): summary.get("best_params")
                                for summary in coupled_summaries
                                if summary.get("best_params")
                            },
                        }
                    )

            self._log("↻ starting coupled equilibrium loop")
            equilibrium_info = self._full_equilibrium_rescan(params, axis_scans)

            fiducial = _normalize_float_dict(params)
            evaluation = self._evaluate(fiducial)
            fiducial_passes = bool(evaluation.get("passes_phase6a", True))

            fiducial_chi2: Optional[float] = None
            chi2_total = evaluation.get("chi2_total")
            if evaluation.get("status") == "valid" and isinstance(chi2_total, (int, float)) and math.isfinite(float(chi2_total)):
                fiducial_chi2 = float(chi2_total)
                self._log(f"   Cycle {cycle_index} χ²_total = {fiducial_chi2:.3f}")
            else:
                self._log(f"⚠️  Cycle {cycle_index} produced invalid or non-finite χ²")

            improvement: Optional[float] = None
            if fiducial_chi2 is not None:
                if math.isfinite(last_valid_chi2):
                    improvement = last_valid_chi2 - fiducial_chi2
                    self._log(f"   Δχ² improvement = {improvement:.6g}")
                last_valid_chi2 = fiducial_chi2
                chi2_history.append(fiducial_chi2)
            else:
                chi2_history.append(None)
                improvement = None

            param_shift: Optional[float] = None
            if previous_cycle_params:
                param_shift = self._relative_param_shift(previous_cycle_params, fiducial)
            param_shift_history.append(param_shift)

            cycle_record: Dict[str, Any] = {
                "cycle": cycle_index,
                "stages": stage_records,
                "fiducial_params": dict(fiducial),
                "passes_phase6a": fiducial_passes,
            }
            if fiducial_chi2 is not None:
                cycle_record["fiducial_chi2"] = fiducial_chi2
            if improvement is not None:
                cycle_record["delta_chi2"] = improvement
            if param_shift is not None:
                cycle_record["param_shift"] = param_shift
            cycle_record["equilibrium"] = equilibrium_info
            if (
                improvement is not None
                and param_shift is not None
                and improvement >= 0.0
                and improvement < self.improvement_tol
                and param_shift < self.param_shift_tol
                and not equilibrium_info.get("edges_pending", False)
            ):
                self._log("   ↳ convergence tolerance reached; stopping.")
                converged = True
            cycle_summaries.append(cycle_record)

            if self._reseed_on_plateau:
                if improvement is None:
                    stagnation_buffer.clear()
                else:
                    stagnation_buffer.append(abs(improvement))
                    if len(stagnation_buffer) > self._plateau_window:
                        stagnation_buffer.pop(0)
                    if (
                        len(stagnation_buffer) == self._plateau_window
                        and all(delta < self._plateau_delta for delta in stagnation_buffer)
                    ):
                        if self._apply_plateau_reseed(params, axis_scans, cycle_index):
                            self._log("   ♻️ plateau detected; reseeding parameters within island.")
                            stagnation_buffer.clear()
                            last_valid_chi2 = math.inf
                            previous_cycle_params = dict(params)
                            converged = False
                            continue

            final_fiducial = fiducial
            final_eval = evaluation
            final_chi2 = fiducial_chi2
            final_passes = fiducial_passes
            previous_cycle_params = dict(fiducial)

            if converged:
                break

        payload: Dict[str, Any] = {
            "version": OPTIMIZER_VERSION,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "model_type": self.model_type,
            "datasets_used": list(self.datasets),
            "phase6a_enforced": self.enforce_phase6a,
            "delta_chi2_tolerance": self.delta_chi2,
            "fiducial_params": final_fiducial,
            "axis_scans": axis_scans,
            "convergence": {
                "max_cycles": self.max_cycles,
                "improvement_tol": self.improvement_tol,
                "chi2_history": chi2_history,
                "param_shift_tol": self.param_shift_tol,
                "param_shift_history": param_shift_history,
                "cycles": cycle_summaries,
                "converged": converged,
            },
        }

        if final_chi2 is not None:
            payload["fiducial_chi2"] = final_chi2
        if final_passes is not None:
            payload["fiducial_passes_phase6a"] = final_passes
        if isinstance(final_eval, Mapping) and "chi2_breakdown" in final_eval:
            breakdown = final_eval["chi2_breakdown"]
            if isinstance(breakdown, Mapping):
                payload["fiducial_breakdown"] = {
                    name: float(value) if isinstance(value, (int, float)) else value
                    for name, value in breakdown.items()
                }
        if isinstance(final_eval, Mapping) and "validation" in final_eval:
            payload["fiducial_validation"] = final_eval["validation"]

        payload["convergence"]["cycles_completed"] = len(cycle_summaries)

        if self._chi2_registry is not None:
            payload["chi2_targets"] = self._chi2_registry.describe()

        self._last_axis_scans = [dict(scan) for scan in axis_scans]
        self._last_result = json.loads(json.dumps(payload))
        self._notify_run_completed(payload)

        return payload

    def run_and_save(self, output_path: Path | str) -> Dict[str, Any]:
        """
        Convenience helper to execute the walk and serialise the payload to disk.
        """
        result = self.run()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        return result

    def _make_scan_range(self, param: str, params: Mapping[str, float], *, pass_id: int) -> List[float]:
        if param not in self.scan_presets:
            raise KeyError(f"No scan preset defined for parameter '{param}'")

        stage = "coarse" if pass_id == 1 else "refine"
        if stage not in self.scan_presets[param]:
            raise KeyError(f"Scan preset for '{param}' missing '{stage}' definition.")
        spec = dict(self.scan_presets[param][stage])

        center = float(params.get(param, self.reference_params.get(param, 0.0)))
        limits: Tuple[Optional[float], Optional[float]] = (None, None)

        if stage == "refine":
            spec = self._prepare_refine_range(param, spec, center, pass_id)

        values: List[float]

        if spec["type"] == "linear":
            start = float(spec["start"])
            stop = float(spec["stop"])
            step = float(spec["step"])
            values = _linear_range(start, stop, step)
            limits = (min(start, stop), max(start, stop))
        elif spec["type"] == "linear_relative":
            clip_min = float(spec["clip_min"]) if "clip_min" in spec else None
            clip_max = float(spec["clip_max"]) if "clip_max" in spec else None
            values = _linear_relative(
                center=center,
                radius=float(spec.get("radius", 0.0)),
                step=float(spec.get("step", 0.0)),
                clip_min=clip_min,
                clip_max=clip_max,
            )
            limits = (clip_min, clip_max)
        elif spec["type"] == "log_relative":
            clip_min = float(spec["clip_min"]) if "clip_min" in spec else None
            clip_max = float(spec["clip_max"]) if "clip_max" in spec else None
            values = _log_relative(
                center=max(center, 1e-12),
                factors=spec.get("factors", []),
                clip_min=clip_min,
                clip_max=clip_max,
            )
            limits = (clip_min, clip_max)
        elif spec["type"] == "list":
            values = [float(v) for v in spec.get("values", [])]
            if values:
                limits = (min(values), max(values))
        elif spec["type"] == "logspace":
            start_exp = float(spec["start_exp"])
            stop_exp = float(spec["stop_exp"])
            num = int(spec["num"])
            values = [float(v) for v in np.logspace(start_exp, stop_exp, num)]
            if values:
                limits = (min(values), max(values))
        else:
            raise ValueError(f"Unsupported scan preset type '{spec['type']}' for parameter '{param}'")

        if not values:
            return [center]

        self._scan_limits[(param, pass_id)] = limits
        return _unique_sorted(values)

    def _prepare_refine_range(
        self,
        param: str,
        spec: Mapping[str, Any],
        center: float,
        pass_id: int,
    ) -> Dict[str, Any]:
        prepared = dict(spec)
        info = self._adaptive_windows.get(param)
        if not info:
            return prepared

        if prepared.get("type") == "linear_relative":
            radius = float(prepared.get("radius", 0.0))
            half_width = info.get("half_width")
            if half_width is not None and half_width > 0.0:
                radius = self._adaptive_blend(radius, half_width)
            min_step = info.get("min_step")
            if min_step is not None and min_step > 0.0:
                radius = max(radius, 2.0 * float(min_step))
            if info.get("edge_bias"):
                radius *= 1.5
            prepared["radius"] = radius

            step_raw = float(prepared.get("step", 0.0)) or max(radius / 10.0, 1e-6)
            if min_step is not None and min_step > 0.0:
                target_step = min(radius, float(min_step))
                step_raw = self._adaptive_blend(step_raw, target_step)
            prepared["step"] = max(step_raw, 1e-6)

            clip_min = prepared.get("clip_min")
            clip_max = prepared.get("clip_max")
            left_edge = info.get("left_edge")
            right_edge = info.get("right_edge")
            if clip_min is not None and left_edge is not None:
                prepared["clip_min"] = float(min(float(clip_min), float(left_edge)))
            if clip_max is not None and right_edge is not None:
                prepared["clip_max"] = float(max(float(clip_max), float(right_edge)))

        elif prepared.get("type") == "log_relative":
            factors = list(prepared.get("factors", []))
            if info.get("edge_bias"):
                factors.extend([0.3, 2.5])
            prepared["factors"] = sorted(set(float(f) for f in factors))
            clip_min = prepared.get("clip_min")
            clip_max = prepared.get("clip_max")
            left_edge = info.get("left_edge")
            right_edge = info.get("right_edge")
            if clip_min is not None and left_edge is not None:
                prepared["clip_min"] = float(min(float(clip_min), float(left_edge)))
            if clip_max is not None and right_edge is not None:
                prepared["clip_max"] = float(max(float(clip_max), float(right_edge)))

        elif prepared.get("type") == "linear":
            start = float(prepared.get("start", center))
            stop = float(prepared.get("stop", center))
            span = abs(stop - start)
            half_width = info.get("half_width")
            if half_width is not None and half_width > 0.0:
                expansion = max(half_width * 1.5, span)
                start = center - expansion
                stop = center + expansion
            prepared["start"] = start
            prepared["stop"] = stop

        return prepared

    def _adaptive_blend(self, previous: Optional[float], new: Optional[float]) -> float:
        if new is None:
            return float(previous) if previous is not None else 0.0
        if previous is None:
            return float(new)
        if not math.isfinite(previous):
            return float(new)
        if not math.isfinite(new):
            return float(previous)
        weight = float(self._adaptive_decay)
        return (1.0 - weight) * float(previous) + weight * float(new)

    def _register_scan_summary(self, summary: Mapping[str, Any]) -> None:
        param = summary.get("param")
        if not isinstance(param, str):
            return
        best = summary.get("best")
        left = summary.get("left_edge")
        right = summary.get("right_edge")
        if best is None or left is None or right is None:
            return
        try:
            best_f = float(best)
            left_f = float(left)
            right_f = float(right)
        except (TypeError, ValueError):
            return
        if not (math.isfinite(best_f) and math.isfinite(left_f) and math.isfinite(right_f)):
            return
        if right_f <= left_f:
            return
        half_width = 0.5 * (right_f - left_f)
        min_step = summary.get("min_step")
        edge_bias = bool(summary.get("edge_hit") or summary.get("edge_rescan"))
        info = {
            "center": best_f,
            "left_edge": left_f,
            "right_edge": right_f,
            "half_width": half_width,
            "min_step": float(min_step) if isinstance(min_step, (int, float)) else None,
            "edge_bias": edge_bias,
            "num_points": summary.get("num_points"),
            "pass_id": summary.get("pass"),
        }
        existing = self._adaptive_windows.get(param)
        if existing:
            blended: Dict[str, Any] = dict(existing)
            for key in ("center", "left_edge", "right_edge", "half_width"):
                if key in info and info[key] is not None:
                    blended[key] = self._adaptive_blend(existing.get(key, info[key]), info[key])
            if info.get("min_step") is not None:
                blended["min_step"] = self._adaptive_blend(existing.get("min_step", info["min_step"]), info["min_step"])
            blended["edge_bias"] = existing.get("edge_bias", False) or info["edge_bias"]
            blended["num_points"] = info.get("num_points", existing.get("num_points"))
            blended["pass_id"] = info.get("pass_id", existing.get("pass_id"))
            self._adaptive_windows[param] = blended
        else:
            self._adaptive_windows[param] = info

    def _refine_axis_minimum(
        self,
        param: str,
        params: Mapping[str, float],
        entries: Sequence[Tuple[float, float, float]],
        best_entry: Tuple[float, float, float],
        pass_id: int,
        cycle: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if len(entries) < 3:
            return None
        ordered = sorted(entries, key=lambda item: item[0])
        best_value = best_entry[0]
        best_score = best_entry[2]
        try:
            best_index = next(i for i, item in enumerate(ordered) if math.isclose(item[0], best_value, rel_tol=1e-12, abs_tol=1e-12))
        except StopIteration:
            return None
        if best_index == 0 or best_index == len(ordered) - 1:
            return None
        left = ordered[best_index - 1]
        right = ordered[best_index + 1]
        x0, f0 = left[0], left[1]
        x1, f1 = best_entry[0], best_entry[1]
        x2, f2 = right[0], right[1]
        denom = (x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0)
        if denom == 0.0:
            return None
        numerator = ((x1 - x0) ** 2) * (f1 - f2) - ((x1 - x2) ** 2) * (f1 - f0)
        candidate_value = x1 - 0.5 * numerator / denom
        if not math.isfinite(candidate_value):
            return None
        left_edge = min(x0, x2)
        right_edge = max(x0, x2)
        if not (left_edge < candidate_value < right_edge):
            return None
        trial_params = dict(params)
        trial_params[param] = float(candidate_value)
        normalized, evaluation, valid, chi2_value, passes_phase6a = self._evaluate_point(trial_params)
        self._cache[_params_key(normalized)] = evaluation
        if not valid or chi2_value is None:
            return None
        score_value = self._score_evaluation(evaluation, chi2_value)
        if score_value is None:
            score_value = chi2_value
        if score_value >= best_score:
            return None
        return {
            "value": float(candidate_value),
            "chi2": float(chi2_value),
            "score": float(score_value),
            "evaluation": evaluation,
            "passes_phase6a": bool(passes_phase6a),
            "pass": pass_id,
            "cycle": cycle,
        }

    def _scan_axis(
        self,
        param: str,
        params: MutableMapping[str, float],
        scan_values: Sequence[float],
        *,
        pass_id: int,
        cycle: Optional[int] = None,
    ) -> Dict[str, Any]:
        total_points = len(scan_values)
        points_buffer: Dict[int, ScanCurvePoint] = {}
        valid_entries: List[Tuple[float, float, float]] = []
        scan_values_sequence = [float(value) for value in scan_values]
        if len(scan_values_sequence) >= 2 and scan_values_sequence[1] < scan_values_sequence[0]:
            scan_values_sequence.sort()
        step_samples: List[float] = []
        for i in range(len(scan_values_sequence) - 1):
            diff = abs(scan_values_sequence[i + 1] - scan_values_sequence[i])
            if diff > 0.0:
                step_samples.append(diff)

        def _register_point(
            index: int,
            trial_value: float,
            chi2_value: Optional[float],
            valid: bool,
            passes_phase6a: bool,
            evaluation: Mapping[str, Any],
        ) -> None:
            rejection_reason: Optional[str] = None
            if not valid:
                rejection_reason = self._diagnose_rejection(evaluation, passes_phase6a=passes_phase6a)
                metadata = evaluation.get("metadata") if isinstance(evaluation, Mapping) else None
                physics_reason = "?"
                if isinstance(metadata, Mapping):
                    physics_reason = metadata.get("reason", physics_reason)
                    if metadata.get("error") == "prior_violation":
                        metadata_prior_violation = True
                    else:
                        metadata_prior_violation = False
                else:
                    metadata_prior_violation = False
                if rejection_reason:
                    rejection_reason = f"{rejection_reason}; physics_reason={physics_reason}"
                else:
                    rejection_reason = f"physics_reason={physics_reason}"
                if self.debug_rejections and not metadata_prior_violation:
                    chi2_txt = "nan" if chi2_value is None else f"{chi2_value:.3f}"
                    self._debug(
                        f"   ✖ {param}={trial_value:.6g} (pass {pass_id}) rejected "
                        f"[χ²={chi2_txt}]: {rejection_reason}"
                    )

            score_value: Optional[float] = None
            if valid and chi2_value is not None:
                score_value = self._score_evaluation(evaluation, chi2_value)
                if score_value is None:
                    score_value = float(chi2_value)
                valid_entries.append((trial_value, float(chi2_value), float(score_value)))

            points_buffer[index] = ScanCurvePoint(
                value=trial_value,
                chi2=chi2_value,
                valid=bool(valid),
                passes_phase6a=passes_phase6a,
                score=float(score_value) if score_value is not None else None,
                rejection_reason=rejection_reason,
            )

        use_parallel = self.parallel and total_points > 1
        if use_parallel:
            futures: Dict[Any, Tuple[int, float]] = {}
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for index, value in enumerate(scan_values):
                    trial_params = dict(params)
                    trial_params[param] = float(value)
                    futures[executor.submit(self._evaluate_point, trial_params)] = (index, float(value))
                progress_bar = None
                if self.progress:
                    desc = f"⚙️ {param} (pass {pass_id})"
                    progress_bar = tqdm(total=len(futures), desc=desc, leave=False)
                try:
                    for future in as_completed(futures):
                        index, scheduled_value = futures[future]
                        trial_params, evaluation, valid, chi2_value, passes_phase6a = future.result()
                        if progress_bar is not None:
                            progress_bar.update(1)
                        self._cache[_params_key(trial_params)] = evaluation
                        trial_value = float(trial_params.get(param, scheduled_value))
                        _register_point(index, trial_value, chi2_value, valid, passes_phase6a, evaluation)
                finally:
                    if progress_bar is not None:
                        progress_bar.close()
        else:
            for index, value in enumerate(self._iter_with_progress(scan_values, param, pass_id)):
                trial_params = dict(params)
                trial_params[param] = float(value)
                (
                    normalized_params,
                    evaluation,
                    valid,
                    chi2_value,
                    passes_phase6a,
                ) = self._evaluate_point(trial_params)
                self._cache[_params_key(normalized_params)] = evaluation
                trial_value = float(normalized_params.get(param, value))
                _register_point(index, trial_value, chi2_value, valid, passes_phase6a, evaluation)

        points: List[ScanCurvePoint] = [
            points_buffer[i] for i in range(total_points)
        ]

        best_value: Optional[float] = None
        chi2_min: Optional[float] = None
        score_min: Optional[float] = None
        left_edge: Optional[float] = None
        right_edge: Optional[float] = None
        refined_payload: Optional[Dict[str, Any]] = None

        if valid_entries:
            valid_entries.sort(key=lambda item: item[2])
            best_value, chi2_min, score_min = valid_entries[0]
            best_entry = valid_entries[0]
            threshold = float(score_min) + self._score_delta
            allowed = [
                (val, chi2, score)
                for val, chi2, score in valid_entries
                if score <= threshold
            ]
            if allowed:
                left_edge = min(val for val, _, _ in allowed)
                right_edge = max(val for val, _, _ in allowed)
            else:
                left_edge = right_edge = best_value

            refined_payload = self._refine_axis_minimum(
                param,
                params,
                valid_entries,
                best_entry,
                pass_id,
                cycle,
            )
            if refined_payload is not None:
                best_value = refined_payload["value"]
                chi2_min = refined_payload["chi2"]
                score_min = refined_payload["score"]
                left_edge = min(left_edge, best_value) if left_edge is not None else best_value
                right_edge = max(right_edge, best_value) if right_edge is not None else best_value

        min_step = min(step_samples) if step_samples else None

        summary: Dict[str, Any] = {
            "param": param,
            "pass": pass_id,
            "best": best_value,
            "chi2_min": chi2_min,
            "score_min": float(score_min) if score_min is not None else None,
            "left_edge": left_edge,
            "right_edge": right_edge,
            "curve": [point.to_dict() for point in points],
            "num_points": len(scan_values),
            "num_valid": len(valid_entries),
        }
        limits = self._scan_limits.get((param, pass_id))
        if limits is not None:
            summary["limits"] = {
                "min": limits[0],
                "max": limits[1],
            }

        if cycle is not None:
            summary["cycle"] = cycle

        if not valid_entries:
            summary["note"] = "no_valid_points"

        summary["scan_values"] = _unique_sorted(scan_values_sequence)
        if min_step is not None:
            summary["min_step"] = float(min_step)
        if refined_payload is not None:
            summary["refined"] = {
                "value": refined_payload["value"],
                "chi2": refined_payload["chi2"],
                "score": refined_payload["score"],
            }

        self._register_scan_summary(summary)
        return summary

    def _scan_axis_with_edge_rescan(
        self,
        param: str,
        params: MutableMapping[str, float],
        scan_values: Sequence[float],
        *,
        pass_id: int,
        cycle: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        current_values = list(scan_values)
        previous_values = list(current_values)
        limits = self._scan_limits.get((param, pass_id))
        pending_edge_rescan: Optional[str] = None

        for expansion in range(self._max_edge_expansions):
            summary = self._scan_axis(param, params, current_values, pass_id=pass_id, cycle=cycle)
            summary["edge_iteration"] = expansion
            if pending_edge_rescan is not None:
                summary["edge_rescan"] = pending_edge_rescan
                summary["edge_origin_points"] = len(previous_values)
            summaries.append(summary)

            edge_direction = self._detect_edge_hit(summary)
            if edge_direction is None:
                break

            summary["edge_hit"] = edge_direction
            original_scan_values = summary.get("scan_values")
            if not original_scan_values:
                break

            expanded_values = self._expand_scan_values(
                original_scan_values,
                edge_direction,
                limits=limits,
                expansion=expansion,
            )
            if not expanded_values:
                break

            if (
                len(expanded_values) == len(original_scan_values)
                and expanded_values == list(original_scan_values)
            ):
                break

            self._log(
                f"   ↺ {param} edge detected on {edge_direction}; extending range to "
                f"[{expanded_values[0]:.6g}, {expanded_values[-1]:.6g}]"
            )

            summary["scan_points_added"] = len(expanded_values) - len(original_scan_values)
            summary["edge_origin_points"] = len(original_scan_values)
            current_values = expanded_values
            previous_values = list(original_scan_values)
            pending_edge_rescan = edge_direction

        return summaries

    def _detect_edge_hit(self, summary: Mapping[str, Any]) -> Optional[str]:
        if summary.get("note") == "no_valid_points":
            return None
        if summary.get("edge_rescan") is not None:
            return None
        best = summary.get("best")
        if best is None or not isinstance(best, (int, float)) or not math.isfinite(float(best)):
            return None
        scan_values = summary.get("scan_values")
        if not scan_values or len(scan_values) < 2:
            return None
        best_f = float(best)
        first = float(scan_values[0])
        last = float(scan_values[-1])
        min_step = summary.get("min_step")
        scale = max(abs(best_f), abs(first), abs(last), 1.0)
        tol = max(
            EDGE_TOLERANCE_EPS * scale,
            float(min_step) * EDGE_TOLERANCE_FACTOR if min_step else 0.0,
        )
        if abs(best_f - first) <= tol:
            return "left"
        if abs(best_f - last) <= tol:
            return "right"
        return None

    @staticmethod
    def _edge_adjust_point_count(values: Sequence[float]) -> int:
        length = len(values)
        if length <= 2:
            return 1
        guess = max(1, length // 5)
        return min(EDGE_EXPANSION_MAX, guess)

    def _expand_scan_values(
        self,
        values: Sequence[float],
        direction: str,
        *,
        limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        expansion: int = 0,
    ) -> Optional[List[float]]:
        ordered = list(values)
        if len(ordered) < 1:
            return None
        if len(ordered) == 1:
            step = max(1e-3, 0.05 * max(abs(ordered[0]), 1.0))
            expanded = [ordered[0] - step, ordered[0], ordered[0] + step]
            return _unique_sorted(expanded)

        if ordered[1] < ordered[0]:
            ordered.sort()

        min_step = None
        for i in range(len(ordered) - 1):
            delta = abs(ordered[i + 1] - ordered[i])
            if delta > 0.0:
                if min_step is None or delta < min_step:
                    min_step = delta

        count = self._edge_adjust_point_count(ordered)
        if count <= 0:
            count = 1

        growth = 1.5 ** max(1, expansion + 1)
        if direction == "left":
            step = ordered[1] - ordered[0]
            if step == 0.0:
                step = max(1e-3, 0.05 * max(abs(ordered[0]), 1.0))
            step *= growth
            new_points = [ordered[0] - step * i for i in range(count, 0, -1)]
            trim = min(count, len(ordered) - 1)
            retained = ordered[: len(ordered) - trim]
            if not retained:
                retained = ordered[:1]
            candidate = new_points + retained

            # Prevent expansion to non-physical values for Rmax
            if hasattr(self, 'param_order') and 'Rmax' in self.param_order:
                candidate = [max(x, 1e6) for x in candidate]  # Rmax must be > 0

        elif direction == "right":
            step = ordered[-1] - ordered[-2]
            if step == 0.0:
                step = max(1e-3, 0.05 * max(abs(ordered[-1]), 1.0))
            step *= growth
            new_points = [ordered[-1] + step * i for i in range(1, count + 1)]
            trim = min(count, len(ordered) - 1)
            retained = ordered[trim:]
            if not retained:
                retained = ordered[-1:]
            candidate = retained + new_points
        else:
            return None

        expanded = _unique_sorted(candidate)
        tolerance = max(EDGE_TOLERANCE_EPS * max(abs(ordered[0]), abs(ordered[-1]), 1.0), (min_step or 0.0) * EDGE_TOLERANCE_FACTOR)
        if direction == "left" and not (expanded[0] < ordered[0] - tolerance or len(expanded) > len(ordered)):
            return None
        if direction == "right" and not (expanded[-1] > ordered[-1] + tolerance or len(expanded) > len(ordered)):
            return None

        clip_min = limits[0] if limits else None
        clip_max = limits[1] if limits else None
        if clip_min is not None:
            expanded = [max(value, float(clip_min)) for value in expanded]
        if clip_max is not None:
            expanded = [min(value, float(clip_max)) for value in expanded]

        return expanded

    def _build_coupled_axis_values(
        self,
        param: str,
        range_info: Mapping[str, Any],
        current_value: float,
    ) -> List[float]:
        left = float(range_info.get("left", current_value))
        right = float(range_info.get("right", current_value))
        best = float(range_info.get("best", current_value))
        delta = float(range_info.get("delta", abs(right - left) * 0.5 or 0.0))
        if not math.isfinite(delta) or delta <= 0.0:
            delta = max(abs(best), 1.0) * 0.05
        candidates = {
            self._clamp(best, left, right),
            self._clamp(current_value, left, right),
        }
        for scale in (0.5, 1.0, 1.5):
            offset = delta * scale
            candidates.add(self._clamp(best - offset, left, right))
            candidates.add(self._clamp(best + offset, left, right))
        unique_values = sorted(set(float(v) for v in candidates if math.isfinite(v)))
        if len(unique_values) > 5:
            mid = len(unique_values) // 2
            return unique_values[mid - 2 : mid + 3]
        return unique_values

    def _run_coupled_updates(
        self,
        params: MutableMapping[str, float],
        axis_scans: List[Dict[str, Any]],
        *,
        cycle: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self._coupled_pairs:
            return []

        summaries: List[Dict[str, Any]] = []
        base_eval = None
        base_chi2 = None
        base_score = None

        for group in self._coupled_pairs:
            ranges = self._extract_parameter_ranges(axis_scans, params)
            base_state = _normalize_float_dict(params)
            base_eval = self._evaluate(base_state)
            base_chi2 = base_eval.get("chi2_total")
            base_score = None
            if isinstance(base_chi2, (int, float)) and math.isfinite(float(base_chi2)):
                base_chi2 = float(base_chi2)
                base_score = self._score_evaluation(base_eval, base_chi2)
            if base_score is None and isinstance(base_chi2, (int, float)):
                base_score = float(base_chi2)

            normalized_group: Tuple[str, ...] = tuple(str(p) for p in group if str(p) in ranges)
            if len(normalized_group) < 2:
                continue

            grid_axes: List[List[float]] = []
            for param_name in normalized_group:
                current_value = float(params.get(param_name, ranges[param_name]["best"]))
                axis_values = self._build_coupled_axis_values(param_name, ranges[param_name], current_value)
                if len(axis_values) == 1:
                    continue
                grid_axes.append(axis_values)

            if len(grid_axes) != len(normalized_group):
                continue

            trials: List[Dict[str, Any]] = []
            for combo in product(*grid_axes):
                trial_params = dict(params)
                for idx, name in enumerate(normalized_group):
                    trial_params[name] = float(combo[idx])
                trials.append(trial_params)

            if not trials:
                continue

            best_trial: Optional[Dict[str, Any]] = None
            best_trial_score: Optional[float] = None
            best_trial_chi2: Optional[float] = None
            best_trial_eval: Optional[Dict[str, Any]] = None
            best_trial_valid = False
            valid_count = 0

            for trial in trials:
                normalized, evaluation, valid, chi2_value, passes_phase6a = self._evaluate_point(trial)
                self._cache[_params_key(normalized)] = evaluation
                if not valid or chi2_value is None:
                    continue
                trial_score = self._score_evaluation(evaluation, chi2_value)
                if trial_score is None:
                    trial_score = chi2_value
                valid_count += 1
                if best_trial_score is None or trial_score < best_trial_score:
                    best_trial = dict(trial)
                    best_trial_score = float(trial_score)
                    best_trial_chi2 = float(chi2_value)
                    best_trial_eval = evaluation
                    best_trial_valid = bool(passes_phase6a)

            summary = {
                "type": "coupled",
                "parameters": normalized_group,
                "num_trials": len(trials),
                "num_valid": valid_count,
                "cycle": cycle,
            }

            if best_trial is None or best_trial_score is None:
                summary["note"] = "no_valid_points"
                summaries.append(summary)
                continue

            summary["best_score"] = best_trial_score
            summary["best_chi2"] = best_trial_chi2
            summary["best_params"] = {name: float(best_trial[name]) for name in normalized_group}
            summary["passes_phase6a"] = best_trial_valid

            improvement = None
            if base_score is not None:
                improvement = float(base_score) - float(best_trial_score)
                summary["delta_score"] = improvement

            if improvement is not None and improvement <= 0.0:
                summaries.append(summary)
                continue

            for name in normalized_group:
                params[name] = float(best_trial[name])
            if best_trial_eval is not None:
                base_eval = best_trial_eval
                base_chi2 = best_trial_chi2
                base_score = best_trial_score
            summaries.append(summary)

        axis_scans.extend(summaries)
        for summary in summaries:
            self._notify_coupled_update(summary)
        return summaries

    def _evaluate(self, params: Mapping[str, float]) -> Dict[str, Any]:
        key = _params_key(params)
        if key in self._cache:
            return self._cache[key]

        prior_reason = prior_violation_reason(params, self.priors)
        if prior_reason is not None:
            diagnostics: Dict[str, Any] = {}
            record_prior_violation(diagnostics, prior_reason)
            result = {
                "status": "invalid",
                "chi2_total": math.inf,
                "chi2_breakdown": {},
                "passes_phase6a": False,
                "metadata": {
                    "error": "prior_violation",
                    "reason": prior_reason,
                },
                "diagnostics": diagnostics,
            }
            self._cache[key] = result
            return result

        result = evaluate_cosmology(self.model_type, dict(params), self.datasets, priors=self.priors)

        # Normalise the breakdown for JSON stability
        if isinstance(result, dict) and "chi2_breakdown" in result:
            breakdown = result["chi2_breakdown"]
            if isinstance(breakdown, dict):
                result["chi2_breakdown"] = {
                    name: float(value) if isinstance(value, (int, float)) else value
                    for name, value in breakdown.items()
                }

        self._cache[key] = result
        return result

    def _full_equilibrium_rescan(
        self,
        params: MutableMapping[str, float],
        axis_scans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        scan_sequence = tuple(
            dict.fromkeys(
                list(self.param_order)
                + list(getattr(self, "second_pass_params", ()))
            )
        )
        if not scan_sequence:
            return {
                "iterations": 0,
                "moved": False,
                "improved": False,
                "edges_pending": False,
                "chi2": None,
            }

        normalized_state = _normalize_float_dict(params)
        evaluation = self._evaluate(normalized_state)
        chi2_total = evaluation.get("chi2_total")
        last_chi2 = float(chi2_total) if isinstance(chi2_total, (int, float)) and math.isfinite(float(chi2_total)) else math.inf

        iterations_completed = 0
        final_moved = False
        final_improved = False
        final_edges = False

        for iteration in range(MAX_EQ_PASSES):
            self._log(f"   ↻ equilibrium iteration {iteration + 1}")
            iteration_moved = False
            iteration_edges = False

            for param in scan_sequence:
                prior_raw = params.get(param, self.reference_params.get(param, 0.0))
                try:
                    prior_value = float(prior_raw)
                except (TypeError, ValueError):
                    prior_value = float(self.reference_params.get(param, 0.0))

                scan_values = self._make_scan_range(param, params, pass_id=2)
                scan_reports = self._scan_axis_with_edge_rescan(param, params, scan_values, pass_id=2)
                axis_scans.extend(scan_reports)
                for report in scan_reports:
                    self._notify_scan_completed(report)

                if any(report.get("edge_hit") or report.get("edge_rescan") for report in scan_reports):
                    iteration_edges = True

                final_report = scan_reports[-1]
                new_best = final_report.get("best")
                if new_best is None:
                    continue

                try:
                    new_value = float(new_best)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(new_value):
                    continue

                if not math.isclose(prior_value, new_value, rel_tol=1e-12, abs_tol=1e-12):
                    iteration_moved = True
                params[param] = new_value

            normalized_state = _normalize_float_dict(params)
            evaluation = self._evaluate(normalized_state)
            chi2_total = evaluation.get("chi2_total")
            current_chi2 = float(chi2_total) if isinstance(chi2_total, (int, float)) and math.isfinite(float(chi2_total)) else math.inf

            iteration_improved = current_chi2 + self.improvement_tol < last_chi2
            if math.isfinite(current_chi2):
                last_chi2 = current_chi2

            iterations_completed = iteration + 1
            final_moved = iteration_moved
            final_improved = iteration_improved
            final_edges = iteration_edges

            if not iteration_moved and not iteration_improved and not iteration_edges:
                self._log(f"✅ full equilibrium reached after {iterations_completed} iteration{'s' if iterations_completed != 1 else ''}")
                return {
                    "iterations": iterations_completed,
                    "moved": False,
                    "improved": False,
                    "edges_pending": False,
                    "chi2": current_chi2 if math.isfinite(current_chi2) else None,
                }

        self._log(f"⚠️ full equilibrium not settled after {MAX_EQ_PASSES} iterations")
        return {
            "iterations": iterations_completed,
            "moved": final_moved,
            "improved": final_improved,
            "edges_pending": final_edges,
            "chi2": last_chi2 if math.isfinite(last_chi2) else None,
        }


__all__ = [
    "CoordinateBasinWalker",
    "DEFAULT_REFERENCES",
    "DEFAULT_PBUF_REFERENCE",
    "DEFAULT_LCDM_REFERENCE",
    "DEFAULT_SCAN_PRESETS",
    "DEFAULT_PARAM_ORDER",
    "DEFAULT_SECOND_PASS_PARAMS",
    "SECOND_PASS_PARAMS",
]
