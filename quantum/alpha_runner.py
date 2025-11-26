"""α derivations driven by configuration data."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .config import AlphaConfig


@dataclass(frozen=True)
class AlphaSample:
    regulator: str
    field_set: str
    mixing_strength: float
    coupling_fraction: float
    cutoff_ratio: float
    alpha_value: float
    in_band: bool


@dataclass(frozen=True)
class AlphaResult:
    alpha_value: float
    alpha_error: float
    derived_parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    warnings: Tuple[str, ...]


def _logspace(start: float, stop: float, num: int) -> List[float]:
    if num < 1:
        raise ValueError("mixing_samples must be >= 1")
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def _generate_mixing_samples(config: AlphaConfig) -> List[float]:
    g_min, g_max = config.mixing_range
    if g_min <= 0 or g_max <= 0 or g_min >= g_max:
        raise ValueError("Invalid mixing_range bounds")
    start = math.log10(g_min)
    stop = math.log10(g_max)
    return [10 ** value for value in _logspace(start, stop, config.mixing_samples)]


def _derive_f_cut(loop_coeff: float, n_eff: float) -> float:
    if loop_coeff <= 0 or n_eff <= 0:
        raise ValueError("Loop coefficients and N_eff must be positive")
    return math.sqrt(1.0 / (loop_coeff * n_eff))


def _coupling_fraction(mixing_strength: float) -> float:
    g = mixing_strength
    return (g * g) / (1.0 + g * g)


def _derive_alpha(f_coup: float, f_cut: float, eps0: float) -> float:
    if eps0 <= 0:
        raise ValueError("eps0 must be positive")
    return f_coup * (f_cut ** 4) / eps0


def _build_samples(config: AlphaConfig, eps0: float) -> List[AlphaSample]:
    mixing_samples = _generate_mixing_samples(config)
    samples: List[AlphaSample] = []
    alpha_min, alpha_max = config.alpha_band
    for regulator, loop_coeff in config.regulators.items():
        for field_set, n_eff in config.field_sets.items():
            f_cut = _derive_f_cut(loop_coeff, n_eff)
            for mixing_strength in mixing_samples:
                f_coup = _coupling_fraction(mixing_strength)
                alpha_value = _derive_alpha(f_coup, f_cut, eps0)
                in_band = alpha_min <= alpha_value <= alpha_max
                samples.append(
                    AlphaSample(
                        regulator=regulator,
                        field_set=field_set,
                        mixing_strength=mixing_strength,
                        coupling_fraction=f_coup,
                        cutoff_ratio=f_cut,
                        alpha_value=alpha_value,
                        in_band=in_band,
                    )
                )
    return samples


def _select_reference_sample(samples: Sequence[AlphaSample], target_reg: str, target_field: str, alpha_target: float) -> AlphaSample:
    if not samples:
        raise ValueError("No α samples available")
    candidates = [s for s in samples if s.regulator == target_reg and s.field_set == target_field]
    if not candidates:
        candidates = list(samples)
    return min(candidates, key=lambda sample: abs(sample.alpha_value - alpha_target))


def _compute_alpha_summary(samples: Sequence[AlphaSample], config: AlphaConfig) -> Tuple[float, float, List[str]]:
    warnings: List[str] = []
    band_alphas = [sample.alpha_value for sample in samples if sample.in_band]
    if band_alphas:
        alpha_value = statistics.geometric_mean(band_alphas)
        alpha_error = statistics.pstdev(band_alphas) if len(band_alphas) > 1 else config.alpha_band[1] - config.alpha_band[0]
    else:
        all_alphas = [sample.alpha_value for sample in samples]
        alpha_value = statistics.mean(all_alphas)
        alpha_error = statistics.pstdev(all_alphas) if len(all_alphas) > 1 else 0.0
        warnings.append("No α samples fell inside the configured band; using global distribution")
    return alpha_value, alpha_error, warnings


def _validate_reproducibility(samples: Sequence[AlphaSample], config: AlphaConfig) -> List[str]:
    if not config.enforce_reproducibility or not config.reference_field:
        return []
    reference_field = config.reference_field
    per_regulator: Dict[str, List[float]] = {}
    for sample in samples:
        if sample.field_set == reference_field and sample.in_band:
            per_regulator.setdefault(sample.regulator, []).append(sample.alpha_value)
    if len(per_regulator) < 2:
        return [
            f"Insufficient α samples in band for field set {reference_field} to check reproducibility"
        ]
    global_mean = statistics.mean(
        statistics.mean(values) for values in per_regulator.values() if values
    )
    warnings: List[str] = []
    for regulator, values in per_regulator.items():
        if not values:
            continue
        regulator_mean = statistics.mean(values)
        if global_mean == 0:
            continue
        delta = abs(regulator_mean - global_mean) / abs(global_mean)
        if delta > config.warnings_threshold:
            warnings.append(
                f"Regulator {regulator} disagrees with field {reference_field} by {delta:.2%}"
            )
    return warnings


def run_alpha_pipeline(eps0: float, config: AlphaConfig) -> AlphaResult:
    samples = _build_samples(config, eps0)
    alpha_value, alpha_error, warnings = _compute_alpha_summary(samples, config)
    warnings.extend(_validate_reproducibility(samples, config))
    reference_sample = _select_reference_sample(samples, config.target_regulator, config.target_field_set, alpha_value)
    metadata = {
        "total_samples": len(samples),
        "band_hits": sum(1 for s in samples if s.in_band),
        "regulator_count": len(config.regulators),
        "field_count": len(config.field_sets),
        "mixing_range": list(config.mixing_range),
        "mixing_samples": config.mixing_samples,
    }
    derived = {
        "regulator": reference_sample.regulator,
        "field_set": reference_sample.field_set,
        "f_cut": reference_sample.cutoff_ratio,
        "f_coup": reference_sample.coupling_fraction,
        "mixing_strength": reference_sample.mixing_strength,
    }
    return AlphaResult(
        alpha_value=alpha_value,
        alpha_error=alpha_error,
        derived_parameters=derived,
        metadata=metadata,
        warnings=tuple(warnings),
    )


__all__ = ["AlphaResult", "run_alpha_pipeline"]
