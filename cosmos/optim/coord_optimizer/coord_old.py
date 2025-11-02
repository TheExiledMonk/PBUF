from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from cosmos.optim.dataset_evaluators import list_available_datasets
from cosmos.optim.parameter_defaults import (
    LCDM_PARAMETER_DEFAULTS,
    PBUF_PARAMETER_DEFAULTS,
)
from cosmos.optim.grid_pipeline import evaluate_cosmology

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm unavailable
    tqdm = None

DEFAULT_REFERENCES: Dict[str, Dict[str, float]] = {
    "pbuf": {**PBUF_PARAMETER_DEFAULTS, "Ol0": 0.0},
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

# Convenience aliases for callers that only work with a single model family
DEFAULT_PBUF_REFERENCE: Dict[str, float] = DEFAULT_REFERENCES["pbuf"]
DEFAULT_LCDM_REFERENCE: Dict[str, float] = DEFAULT_REFERENCES["lcdm"]

# Backwards compatibility export (legacy callers may import this name directly)
SECOND_PASS_PARAMS: Tuple[str, ...] = DEFAULT_SECOND_PASS_PARAMS["pbuf"]
OPTIMIZER_VERSION = "coord-opt-old-v1"
DEFAULT_DELTA_CHI2 = 20.0


@dataclass(frozen=True)
class ScanCurvePoint:
    value: float
    chi2: Optional[float]
    valid: bool
    passes_phase6a: bool
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


class CoordinateBasinWalkerOld:
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

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _debug(self, message: str) -> None:
        if self.debug_rejections:
            print(message)

    def _iter_with_progress(self, values: Sequence[float], param: str, pass_id: int):
        if not self.progress or not values:
            for value in values:
                yield value
            return
        desc = f"{param} (pass {pass_id})"
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

    def run(self) -> Dict[str, Any]:
        """
        Execute the basin walk and return a JSON-serialisable payload.
        """
        params: Dict[str, float] = _normalize_float_dict(self.reference_params)
        axis_scans: List[Dict[str, Any]] = []

        self._log("▶️  Starting coordinate basin walk")
        self._log(f"   Model: {self.model_type}")
        self._log(f"   Datasets: {', '.join(self.datasets)}")
        if self._progress_requested and not self.progress:
            self._log("   (Requested progress bars but tqdm is unavailable; falling back to plain iteration)")

        # First pass
        for param in self.param_order:
            scan_values = self._make_scan_range(param, params, pass_id=1)
            self._log(
                f"→ Pass 1: scanning {param} over {len(scan_values)} points "
                f"[range {scan_values[0]:.6g} .. {scan_values[-1]:.6g}]"
            )
            scan_summary = self._scan_axis(param, params, scan_values, pass_id=1)
            axis_scans.append(scan_summary)
            if scan_summary.get("best") is not None:
                params[param] = float(scan_summary["best"])
                self._log(
                    f"   ↳ best {param} = {scan_summary['best']:.6g}, "
                    f"χ² = {scan_summary.get('chi2_min', float('nan')):.3f}"
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
            scan_summary = self._scan_axis(param, params, scan_values, pass_id=2)
            axis_scans.append(scan_summary)
            if scan_summary.get("best") is not None:
                params[param] = float(scan_summary["best"])
                self._log(
                    f"   ↳ best {param} = {scan_summary['best']:.6g}, "
                    f"χ² = {scan_summary.get('chi2_min', float('nan')):.3f}"
                )
            else:
                self._log(f"   ↳ no valid points for {param}")

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

        if spec["type"] == "linear":
            values = _linear_range(float(spec["start"]), float(spec["stop"]), float(spec["step"]))
        elif spec["type"] == "linear_relative":
            values = _linear_relative(
                center=center,
                radius=float(spec.get("radius", 0.0)),
                step=float(spec.get("step", 0.0)),
                clip_min=float(spec["clip_min"]) if "clip_min" in spec else None,
                clip_max=float(spec["clip_max"]) if "clip_max" in spec else None,
            )
        elif spec["type"] == "log_relative":
            values = _log_relative(
                center=max(center, 1e-12),
                factors=spec.get("factors", []),
                clip_min=float(spec["clip_min"]) if "clip_min" in spec else None,
                clip_max=float(spec["clip_max"]) if "clip_max" in spec else None,
            )
        elif spec["type"] == "list":
            values = [float(v) for v in spec.get("values", [])]
        elif spec["type"] == "logspace":
            start_exp = float(spec["start_exp"])
            stop_exp = float(spec["stop_exp"])
            num = int(spec["num"])
            values = [float(v) for v in np.logspace(start_exp, stop_exp, num)]
        else:
            raise ValueError(f"Unsupported scan preset type '{spec['type']}' for parameter '{param}'")

        if not values:
            return [center]
        return _unique_sorted(values)

    def _scan_axis(
        self,
        param: str,
        params: MutableMapping[str, float],
        scan_values: Sequence[float],
        *,
        pass_id: int,
    ) -> Dict[str, Any]:
        points: List[ScanCurvePoint] = []
        valid_entries: List[Tuple[float, float]] = []

        for value in self._iter_with_progress(scan_values, param, pass_id):
            trial_params = dict(params)
            trial_params[param] = float(value)
            evaluation = self._evaluate(trial_params)

            status = evaluation.get("status")
            chi2_total = evaluation.get("chi2_total")
            passes_phase6a = bool(evaluation.get("passes_phase6a", True))

            valid = (
                status == "valid"
                and chi2_total is not None
                and math.isfinite(chi2_total)
            )
            if self.enforce_phase6a and self.model_type == "pbuf":
                valid = valid and passes_phase6a

            chi2_value = float(chi2_total) if valid else (float(chi2_total) if isinstance(chi2_total, (int, float)) and math.isfinite(chi2_total) else None)
            rejection_reason: Optional[str] = None

            if not valid:
                rejection_reason = self._diagnose_rejection(evaluation, passes_phase6a=passes_phase6a)
                md = evaluation.get("metadata", {}) or {}
                physics_reason = md.get("reason", "?")
                if rejection_reason:
                    rejection_reason = f"{rejection_reason}; physics_reason={physics_reason}"
                else:
                    rejection_reason = f"physics_reason={physics_reason}"
                if self.debug_rejections:
                    chi2_txt = (
                        "nan"
                        if chi2_value is None
                        else f"{chi2_value:.3f}"
                    )
                    self._debug(
                        f"   ✖ {param}={float(value):.6g} (pass {pass_id}) rejected "
                        f"[χ²={chi2_txt}]: {rejection_reason}"
                    )

            if valid and chi2_value is not None:
                valid_entries.append((float(value), chi2_value))

            points.append(
                ScanCurvePoint(
                    value=float(value),
                    chi2=chi2_value,
                    valid=bool(valid),
                    passes_phase6a=passes_phase6a,
                    rejection_reason=rejection_reason,
                )
            )

        best_value: Optional[float] = None
        chi2_min: Optional[float] = None
        left_edge: Optional[float] = None
        right_edge: Optional[float] = None

        if valid_entries:
            valid_entries.sort(key=lambda item: item[1])
            best_value, chi2_min = valid_entries[0]
            allowed = [
                (val, chi2)
                for val, chi2 in valid_entries
                if chi2 <= chi2_min + self.delta_chi2
            ]
            if allowed:
                left_edge = min(val for val, _ in allowed)
                right_edge = max(val for val, _ in allowed)

        summary: Dict[str, Any] = {
            "param": param,
            "pass": pass_id,
            "best": best_value,
            "chi2_min": chi2_min,
            "left_edge": left_edge,
            "right_edge": right_edge,
            "curve": [point.to_dict() for point in points],
            "num_points": len(scan_values),
            "num_valid": len(valid_entries),
        }

        if not valid_entries:
            summary["note"] = "no_valid_points"

        return summary

    def _evaluate(self, params: Mapping[str, float]) -> Dict[str, Any]:
        key = _params_key(params)
        if key in self._cache:
            return self._cache[key]

        result = evaluate_cosmology(self.model_type, dict(params), self.datasets)

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


__all__ = [
    "CoordinateBasinWalkerOld",
    "DEFAULT_REFERENCES",
    "DEFAULT_PBUF_REFERENCE",
    "DEFAULT_LCDM_REFERENCE",
    "DEFAULT_SCAN_PRESETS",
    "DEFAULT_PARAM_ORDER",
    "DEFAULT_SECOND_PASS_PARAMS",
    "SECOND_PASS_PARAMS",
]
