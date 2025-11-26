"""CMB distance unit audit shared by LCDM and PBUF.

The helper focuses purely on unit consistency between both models:

* Cross-model ratios highlight kilometre-vs-megaparsec style mistakes.
* D_A vs D_M/(1 + z_star) keeps each model internally honest.
* The printed report deliberately mirrors the request so scientists can
  eyeball mismatches immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from cosmos.models.lcdm import cmb as lcdm_cmb
from cosmos.models.lcdm import distances as lcdm_distances
from cosmos.models.lcdm import utils as lcdm_utils
from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.pbuf import cmb as pbuf_cmb
from cosmos.models.pbuf import distances as pbuf_distances
from cosmos.models.pbuf import utils as pbuf_utils
from cosmos.models.pbuf.model import PBUFModel

__all__ = ["run_cmb_unit_audit"]


Integrator = Callable[[Callable[[float], float], float, float], float]


@dataclass
class ModelDistanceReport:
    name: str
    D_M: float
    D_A: float
    r_s: float
    internal_pass: bool
    internal_message: str


@dataclass
class RatioCheck:
    value: float
    passed: bool
    message: str


def _make_integrator(simpson_integral: Callable[..., float], steps: int) -> Integrator:
    """Wrap the model-local Simpson rule to freeze the number of steps."""

    even_steps = steps if steps % 2 == 0 else steps + 1

    def integrate(func: Callable[[float], float], lower: float, upper: float) -> float:
        return simpson_integral(func, lower, upper, n=even_steps)

    return integrate


def _compute_lcdm_distances(lcdm: LCDMModel, z_star: float, distance_steps: int, sound_steps: int) -> Tuple[float, float, float]:
    integrator = _make_integrator(lcdm_utils.simpson_integral, distance_steps)
    sound_integrator = _make_integrator(lcdm_utils.simpson_integral, sound_steps)
    params = lcdm.params
    D_M = lcdm_distances.comoving_distance(z_star, params, integrator)
    D_A = lcdm_distances.angular_diameter_distance(z_star, params, integrator)
    r_s = lcdm_cmb.sound_horizon(z_star, params, sound_integrator)
    return D_M, D_A, r_s


def _compute_pbuf_distances(pbuf: PBUFModel, z_star: float, distance_steps: int, sound_steps: int) -> Tuple[float, float, float]:
    integrator = _make_integrator(pbuf_utils.simpson_integral, distance_steps)
    sound_integrator = _make_integrator(pbuf_utils.simpson_integral, sound_steps)
    params = pbuf.params
    table = pbuf.thermal_table
    D_M = pbuf_distances.comoving_distance(z_star, params, table, integrator)
    D_A = pbuf_distances.angular_diameter_distance(z_star, params, table, integrator)
    r_s = pbuf_cmb.sound_horizon(z_star, params, table, sound_integrator)
    return D_M, D_A, r_s


def _internal_distance_report(name: str, D_M: float, D_A: float, z_star: float, tolerance: float) -> ModelDistanceReport:
    expected_D_A = D_M / (1.0 + z_star)
    if D_A == 0.0:
        rel_diff = float("inf") if expected_D_A != 0.0 else 0.0
    else:
        rel_diff = abs(D_A - expected_D_A) / abs(D_A)
    passed = rel_diff <= tolerance

    if passed:
        message = f"D_A matches D_M/(1+z_star) within {rel_diff:.3e} relative error."
    else:
        pct = rel_diff * 100.0
        message = f"D_A inconsistent with D_M/(1+z_star) by {pct:.2f}% (> {tolerance * 100.0:.2f}%)."

    return ModelDistanceReport(name=name, D_M=D_M, D_A=D_A, r_s=0.0, internal_pass=passed, internal_message=message)


def _ratio_check(name: str, numerator: float, denominator: float, ratio_tolerance: float, thousand_threshold: float, thousand_min: float) -> RatioCheck:
    if denominator == 0.0:
        raise ValueError(f"Cannot evaluate {name} ratio because the LCDM value is zero.")

    ratio = numerator / denominator
    if ratio > thousand_threshold or ratio < thousand_min:
        return RatioCheck(
            value=ratio,
            passed=False,
            message="Magnitude mismatch (~10^3 difference) suggests a units mix-up.",
        )

    fractional_offset = abs(ratio - 1.0)
    if fractional_offset > ratio_tolerance:
        return RatioCheck(
            value=ratio,
            passed=False,
            message=f"Ratio differs by {fractional_offset * 100.0:.2f}% (> {ratio_tolerance * 100.0:.2f}% tolerance).",
        )

    return RatioCheck(
        value=ratio,
        passed=True,
        message=f"Within {fractional_offset * 100.0:.2f}% of LCDM.",
    )


def run_cmb_unit_audit(
    lcdm: LCDMModel,
    pbuf: PBUFModel,
    *,
    z_star: float = 1090.0,
    distance_steps: int = 4096,
    sound_steps: int = 4096,
    internal_tolerance: float = 0.01,
    ratio_tolerance: float = 0.05,
    thousand_threshold: float = 100.0,
    thousand_min: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    """
    Compare LCDM and PBUF CMB distances at a fixed z_star to ensure unit consistency.

    The function prints the requested PASS/FAIL table and returns a dictionary with
    the raw numbers for any downstream scripting.
    """

    if z_star <= 0.0:
        raise ValueError("z_star must be positive.")

    lcdm_D_M, lcdm_D_A, lcdm_r_s = _compute_lcdm_distances(lcdm, z_star, distance_steps, sound_steps)
    pbuf_D_M, pbuf_D_A, pbuf_r_s = _compute_pbuf_distances(pbuf, z_star, distance_steps, sound_steps)

    lcdm_report = _internal_distance_report("LCDM", lcdm_D_M, lcdm_D_A, z_star, internal_tolerance)
    lcdm_report.r_s = lcdm_r_s

    pbuf_report = _internal_distance_report("PBUF", pbuf_D_M, pbuf_D_A, z_star, internal_tolerance)
    pbuf_report.r_s = pbuf_r_s

    ratio_checks = {
        "D_M": _ratio_check("D_M", pbuf_D_M, lcdm_D_M, ratio_tolerance, thousand_threshold, thousand_min),
        "D_A": _ratio_check("D_A", pbuf_D_A, lcdm_D_A, ratio_tolerance, thousand_threshold, thousand_min),
        "r_s": _ratio_check("r_s", pbuf_r_s, lcdm_r_s, ratio_tolerance, thousand_threshold, thousand_min),
    }

    lines = [
        "=== CMB Unit Audit ===",
        "",
        "LCDM:",
        f"  D_M:      {lcdm_report.D_M:12.3f} Mpc",
        f"  D_A:      {lcdm_report.D_A:12.3f} Mpc",
        f"  r_s:      {lcdm_report.r_s:12.3f} Mpc",
        f"  Check:    {'PASS' if lcdm_report.internal_pass else 'FAIL'} ({lcdm_report.internal_message})",
        "",
        "PBUF:",
        f"  D_M:      {pbuf_report.D_M:12.3f} Mpc",
        f"  D_A:      {pbuf_report.D_A:12.3f} Mpc",
        f"  r_s:      {pbuf_report.r_s:12.3f} Mpc",
        f"  Check:    {'PASS' if pbuf_report.internal_pass else 'FAIL'} ({pbuf_report.internal_message})",
        "",
        "LCDM vs PBUF:",
    ]

    for key in ("D_M", "D_A", "r_s"):
        check = ratio_checks[key]
        lines.append(f"  {key} ratio:   {check.value:8.4f}  ({'PASS' if check.passed else 'FAIL'} - {check.message})")

    model_pass = lcdm_report.internal_pass and pbuf_report.internal_pass
    ratios_pass = all(check.passed for check in ratio_checks.values())
    overall_pass = model_pass and ratios_pass

    lines.extend(
        [
            "",
            "=== Summary ===",
            "PASS" if overall_pass else "FAIL",
        ]
    )

    print("\n".join(lines))

    return {
        "lcdm": {"D_M": lcdm_report.D_M, "D_A": lcdm_report.D_A, "r_s": lcdm_report.r_s},
        "pbuf": {"D_M": pbuf_report.D_M, "D_A": pbuf_report.D_A, "r_s": pbuf_report.r_s},
        "ratios": {name: check.value for name, check in ratio_checks.items()},
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    try:
        from cosmos.models import create_model  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - compatibility fallback
        from cosmos import create_model  # type: ignore

    try:
        lcdm_model = create_model("lcdm")
        pbuf_model = create_model("pbuf")
    except ValueError as err:  # pragma: no cover - example helper
        raise SystemExit(
            "Supply explicit LCDM/PBUF parameter dictionaries when using the audit helper."
        ) from err

    run_cmb_unit_audit(lcdm_model, pbuf_model, z_star=1090.0)
