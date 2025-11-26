"""Standalone thermal configuration validator.

This script reconstructs the thermal curve generation and Phase-7a thermal
sanity checks using the ground-truth formulas currently implemented in
quantum/thermal/table.py, quantum/alpha_runner.py, and
cosmos2/models/pbuf/phase7a.py. It does not import those modules; everything
is reimplemented here for isolation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


T_CMB0 = 2.7255
ANCHORS = (1.0e10, 1.0e8, 4.0e3, T_CMB0, 10.0)
REFINEMENT_SPAN = 3.0


def _load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text())


def _coerce_list(text: str | None, cast=float) -> List[float | str]:
    if text is None:
        return []
    values: List[float | str] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(cast(part))
        except Exception:
            values.append(part)
    return values


def _range_from_arg(text: str | None, default: List[float]) -> List[float]:
    return _coerce_list(text) if text else default


def _logspace_with_refinements(t_min: float, t_max: float, num: int, dense_points: int) -> np.ndarray:
    base = np.logspace(np.log10(t_min), np.log10(t_max), num=num, endpoint=True)
    refinements: List[np.ndarray] = []
    for anchor in ANCHORS:
        if anchor <= t_min or anchor >= t_max:
            continue
        left = max(anchor / REFINEMENT_SPAN, t_min)
        right = min(anchor * REFINEMENT_SPAN, t_max)
        refined = np.logspace(np.log10(left), np.log10(right), num=max(4, dense_points))
        refinements.append(refined)
    if refinements:
        mesh = np.unique(np.concatenate([base, *refinements]))
    else:
        mesh = base
    mesh.sort()
    return mesh[::-1]  # High -> low temperature order


def _epsilon_mode(mode: str, temps: np.ndarray, beta: float, t_star: float, power: float, eps_min: float) -> Tuple[np.ndarray, np.ndarray]:
    ratio = temps / t_star
    if mode == "off":
        eps = np.ones_like(temps)
        dln = np.zeros_like(temps)
        return eps, dln

    if mode == "linear":
        eps = 1.0 - beta * ratio
        dln = -(beta * ratio) / np.maximum(eps, eps_min)
    elif mode == "power":
        powered = np.power(ratio, power)
        eps = 1.0 - beta * powered
        dln = -(beta * power * powered) / np.maximum(eps, eps_min)
    elif mode == "exp":
        powered = np.power(ratio, power)
        eps = np.exp(-beta * powered)
        dln = -beta * power * powered
    else:
        raise ValueError(f"Unsupported thermal mode '{mode}'")

    clamped = eps < eps_min
    if np.any(clamped):
        eps = np.where(clamped, eps_min, eps)
        dln = np.where(clamped, 0.0, dln)
    return eps, dln


def _derive_curves(
    mode: str,
    temps: np.ndarray,
    beta: float,
    t_star: float,
    power: float,
    eps_min: float,
    alpha_qm: float,
) -> Dict[str, np.ndarray]:
    eps, dln_eps = _epsilon_mode(mode, temps, beta, t_star, power, eps_min)
    alpha = alpha_qm * eps
    dln_alpha = dln_eps
    z = np.maximum(temps / T_CMB0 - 1.0, 0.0)
    a = 1.0 / (1.0 + z)
    return {
        "T": temps,
        "a": a,
        "epsilon0": eps,
        "alpha": alpha,
        "dln_epsilon0_dlnT": dln_eps,
        "dln_alpha_dlnT": dln_alpha,
    }


@dataclass(frozen=True)
class Phase7aThresholds:
    alpha_max_abs: float
    alpha_step_max: float
    epsilon0_max: float
    epsilon0_step_max: float
    k_sat_step_max: float
    alpha_deriv_max: float
    epsilon_deriv_max: float
    rmax_min: float
    rmax_max: float
    rmax_step_factor: float


def load_phase7a_thresholds(path: Path) -> Phase7aThresholds:
    payload = _load_json(path)
    return Phase7aThresholds(
        alpha_max_abs=float(payload.get("alpha_max_abs", 0.1)),
        alpha_step_max=float(payload.get("alpha_step_max", 0.01)),
        epsilon0_max=float(payload.get("epsilon0_max", 2.0)),
        epsilon0_step_max=float(payload.get("epsilon0_step_max", 0.02)),
        k_sat_step_max=float(payload.get("k_sat_step_max", 0.02)),
        alpha_deriv_max=float(payload.get("alpha_deriv_max", 30.0)),
        epsilon_deriv_max=float(payload.get("epsilon_deriv_max", 30.0)),
        rmax_min=float(payload.get("Rmax_min", 1.0e5)),
        rmax_max=float(payload.get("Rmax_max", 1.0e10)),
        rmax_step_factor=float(payload.get("Rmax_step_factor", 3.0)),
    )


def _format_a(val: float) -> str:
    return f"{val:.3e}"


def _check_range(values: np.ndarray, mask: np.ndarray, label: str, limit: str, a_vals: np.ndarray, reasons: List[str]) -> None:
    if not mask.any():
        return
    idx = int(np.argmax(mask))
    reasons.append(f"{label}[{idx}] at a≈{_format_a(a_vals[idx])}={values[idx]:.3g} outside {limit}")


def _check_steps(values: np.ndarray, limit: float, label: str, a_vals: np.ndarray, reasons: List[str]) -> None:
    diffs = np.abs(np.diff(values))
    if diffs.size == 0:
        return
    mask = diffs >= limit
    if not mask.any():
        return
    idx = int(np.argmax(mask))
    reasons.append(
        f"Δ{label}[{idx}->{idx+1}] between a≈{_format_a(a_vals[idx])} and a≈{_format_a(a_vals[idx+1])} is {diffs[idx]:.3g} ≥ {limit:.3g}"
    )


def _check_derivatives(alpha: np.ndarray, eps: np.ndarray, k_sat: np.ndarray, dln_alpha: np.ndarray, dln_eps: np.ndarray, cfg: Phase7aThresholds, a_vals: np.ndarray, reasons: List[str]) -> None:
    if not np.all(np.isfinite(dln_alpha)):
        idx = int(np.argmax(~np.isfinite(dln_alpha)))
        reasons.append(f"dln alpha/dln T not finite at idx {idx} (a≈{_format_a(a_vals[idx])})")
        return
    if not np.all(np.isfinite(dln_eps)):
        idx = int(np.argmax(~np.isfinite(dln_eps)))
        reasons.append(f"dln epsilon/dln T not finite at idx {idx} (a≈{_format_a(a_vals[idx])})")
        return

    mask_alpha = np.abs(dln_alpha) >= cfg.alpha_deriv_max
    if mask_alpha.any():
        idx = int(np.argmax(mask_alpha))
        reasons.append(
            f"|dln alpha/dln T|={np.abs(dln_alpha[idx]):.3g} ≥ {cfg.alpha_deriv_max:.3g} at idx {idx} (a≈{_format_a(a_vals[idx])})"
        )
        return

    mask_eps = np.abs(dln_eps) >= cfg.epsilon_deriv_max
    if mask_eps.any():
        idx = int(np.argmax(mask_eps))
        reasons.append(
            f"|dln epsilon/dln T|={np.abs(dln_eps[idx]):.3g} ≥ {cfg.epsilon_deriv_max:.3g} at idx {idx} (a≈{_format_a(a_vals[idx])})"
        )
        return

    positive = k_sat > 0.0
    if positive.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            dln_k = ((eps[positive] * dln_eps[positive]) - (alpha[positive] * dln_alpha[positive])) / k_sat[positive]
        bad = ~np.isfinite(dln_k)
        if bad.any():
            idxs = np.nonzero(positive)[0]
            bad_idx = int(idxs[np.argmax(bad)])
            reasons.append(f"dln k_sat/dln T not finite at idx {bad_idx} (a≈{_format_a(a_vals[bad_idx])})")


def check_phase7a(curves: Mapping[str, np.ndarray], cfg: Phase7aThresholds, metadata: Mapping[str, object] | None = None) -> Tuple[bool, List[str]]:
    a_vals = curves["a"]
    alpha = curves["alpha"]
    eps = curves["epsilon0"]
    k_sat = eps - alpha
    dln_alpha = curves["dln_alpha_dlnT"]
    dln_eps = curves["dln_epsilon0_dlnT"]

    reasons: List[str] = []

    for label, values in (("alpha", alpha), ("epsilon0", eps), ("k_sat", k_sat), ("a", a_vals)):
        if not np.all(np.isfinite(values)):
            idx = int(np.argmax(~np.isfinite(values)))
            reasons.append(f"{label} contains non-finite entry at idx {idx} (a≈{_format_a(a_vals[idx])})")
            return False, reasons

    _check_range(alpha, np.abs(alpha) > cfg.alpha_max_abs, "alpha", f"[-{cfg.alpha_max_abs:.3g}, {cfg.alpha_max_abs:.3g}]", a_vals, reasons)
    _check_range(eps, (eps <= 0.0) | (eps > cfg.epsilon0_max), "epsilon0", f"(0, {cfg.epsilon0_max:.3g}]", a_vals, reasons)
    _check_range(k_sat, (k_sat < 0.0) | (k_sat > 1.0), "k_sat", "[0.0, 1.0]", a_vals, reasons)

    _check_derivatives(alpha, eps, k_sat, dln_alpha, dln_eps, cfg, a_vals, reasons)
    if reasons:
        return False, reasons

    _check_steps(alpha, cfg.alpha_step_max, "alpha", a_vals, reasons)
    _check_steps(eps, cfg.epsilon0_step_max, "epsilon0", a_vals, reasons)
    _check_steps(k_sat, cfg.k_sat_step_max, "k_sat", a_vals, reasons)

    rmax_val = None
    if metadata:
        rmax_val = metadata.get("Rmax") or metadata.get("R_max") or metadata.get("Rmax_val")
    if rmax_val is not None:
        rmax_val = float(rmax_val)
        if not (cfg.rmax_min <= rmax_val <= cfg.rmax_max):
            reasons.append(f"R_max {rmax_val:.3g} outside [{cfg.rmax_min:.3g}, {cfg.rmax_max:.3g}]")
        if a_vals.size >= 2:
            ratios = np.ones(a_vals.size - 1)
            if np.any(ratios >= cfg.rmax_step_factor):
                idx = int(np.argmax(ratios >= cfg.rmax_step_factor))
                reasons.append(
                    f"R_max ratio at idx {idx} (a≈{_format_a(a_vals[idx])}) = {ratios[idx]:.3g} >= {cfg.rmax_step_factor:.3g}"
                )

    return len(reasons) == 0, reasons


def _stats(values: np.ndarray) -> Dict[str, float]:
    return {"min": float(np.min(values)), "max": float(np.max(values))}


def evaluate_configuration(
    *,
    mode: str,
    beta: float,
    t_star: float,
    power: float,
    eps_min: float,
    alpha_qm: float,
    t_min: float,
    t_max: float,
    num_points: int,
    dense_points: int,
    thresholds: Phase7aThresholds,
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    temps = _logspace_with_refinements(t_min, t_max, num_points, dense_points)
    curves = _derive_curves(mode, temps, beta, t_star, power, eps_min, alpha_qm)
    ok, reasons = check_phase7a(curves, thresholds, metadata)
    eps = curves["epsilon0"]
    alpha = curves["alpha"]
    dln_eps = curves["dln_epsilon0_dlnT"]
    dln_alpha = curves["dln_alpha_dlnT"]
    clamp_count = int(np.count_nonzero(eps == eps_min))

    return {
        "pass": ok,
        "reasons": reasons,
        "params": {
            "thermal_mode": mode,
            "beta": beta,
            "T_star": t_star,
            "power_index": power,
            "eps_min": eps_min,
        },
        "metadata": dict(metadata),
        "stats": {
            "epsilon0": _stats(eps),
            "alpha": _stats(alpha),
            "dln_epsilon0_dlnT": _stats(np.abs(dln_eps)),
            "dln_alpha_dlnT": _stats(np.abs(dln_alpha)),
            "clamp_count": clamp_count,
            "grid_points": int(temps.size),
        },
    }


def _load_micro(path: Path) -> Dict[str, object]:
    payload = _load_json(path)
    if isinstance(payload, dict) and "metadata" in payload:
        meta = payload["metadata"]
        if isinstance(meta, dict):
            return dict(meta)
    return dict(payload) if isinstance(payload, dict) else {}


def _resolve_default(value: float | None, fallback_keys: Sequence[str], payload: Mapping[str, object], default: float) -> float:
    if value is not None:
        return float(value)
    for key in fallback_keys:
        if key in payload:
            try:
                return float(payload[key])
            except Exception:
                continue
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone thermal configuration validator")
    parser.add_argument("--micro", type=Path, required=True, help="Path to quantum output or thermal metadata JSON")
    parser.add_argument("--phase7a", type=Path, default=Path("configs/phase7a/pbuf.json"), help="Path to Phase-7a thresholds JSON")
    parser.add_argument("--thermal-modes", type=str, help="Comma list of thermal modes to scan (default: micro or linear)")
    parser.add_argument("--beta", type=str, help="Comma list of beta values")
    parser.add_argument("--t-star", type=str, help="Comma list of T_star values")
    parser.add_argument("--power", type=str, help="Comma list of power indices")
    parser.add_argument("--eps-min", type=float, help="Override eps_min")
    parser.add_argument("--t-min", type=float, help="Override t_min for grid")
    parser.add_argument("--t-max", type=float, help="Override t_max for grid")
    parser.add_argument("--num-points", type=int, help="Override base grid points")
    parser.add_argument("--dense-points", type=int, help="Override refinement points")
    parser.add_argument("--output", type=Path, help="Optional output file (JSON lines); defaults to stdout")
    parser.add_argument("--regulator", type=str, help="Regulator name(s) to include in metadata (comma list)")
    parser.add_argument("--field-content", type=str, help="Field content name(s) (comma list)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    micro = _load_micro(args.micro)
    thresholds = load_phase7a_thresholds(args.phase7a)

    alpha_qm = _resolve_default(None, ("alpha_qm", "alpha_QM"), micro, 0.03)
    eps_min = args.eps_min if args.eps_min is not None else float(micro.get("eps_min", 1.0e-4))
    beta_default = float(micro.get("beta", 0.05))
    t_star_default = float(micro.get("t_star", micro.get("T_star", 1.0e6)))
    power_default = float(micro.get("power", micro.get("power_index", 1.0)))
    mode_default = str(micro.get("mode", micro.get("thermal_mode", "linear")))

    modes = [m.lower() for m in (_coerce_list(args.thermal_modes, cast=str) or [mode_default])]
    betas = _range_from_arg(args.beta, [beta_default])
    t_stars = _range_from_arg(args.t_star, [t_star_default])
    powers = _range_from_arg(args.power, [power_default])

    t_min = args.t_min if args.t_min is not None else float(micro.get("t_min", 2.725))
    t_max = args.t_max if args.t_max is not None else float(micro.get("t_max", 1.0e12))
    num_points = args.num_points if args.num_points is not None else int(micro.get("num_points", micro.get("iterations", 512)))
    dense_points = args.dense_points if args.dense_points is not None else int(micro.get("dense_points", 24))

    regulators = _coerce_list(args.regulator, cast=str) or [micro.get("regulator", "thermal_default")]
    fields = _coerce_list(args.field_content, cast=str) or [micro.get("field_content", "SM_full")]

    output_handle = args.output.open("w") if args.output else None
    sink = output_handle or None

    all_records: List[Dict[str, object]] = []
    pass_records: List[Dict[str, object]] = []

    for reg, field in itertools.product(regulators, fields):
        base_meta = {
            "regulator": reg,
            "field_content": field,
            "f_cut": micro.get("f_cut_T", micro.get("f_cut")),
            "f_coup": micro.get("f_coup_T", micro.get("f_coup")),
            "alpha_qm": alpha_qm,
            "eps_min": eps_min,
            "beta_base": beta_default,
            "t_star_base": t_star_default,
            "power_base": power_default,
        }
        for mode, beta_val, t_star_val, power_val in itertools.product(modes, betas, t_stars, powers):
            record = evaluate_configuration(
                mode=mode,
                beta=float(beta_val),
                t_star=float(t_star_val),
                power=float(power_val),
                eps_min=eps_min,
                alpha_qm=alpha_qm,
                t_min=t_min,
                t_max=t_max,
                num_points=max(int(num_points), 32),
                dense_points=max(int(dense_points), 0),
                thresholds=thresholds,
                metadata=base_meta,
            )
            all_records.append(record)
            line = json.dumps(record)
            # Screen output: only show passing configurations to highlight valid islands.
            if record.get("pass"):
                pass_records.append(record)
                print(line)
            if sink:
                sink.write(line + "\n")

    if sink:
        sink.close()

    if pass_records:
        print("\n=== Valid configuration islands (ranges over passing grid points) ===")
        grouped: Dict[tuple[str, str, str], Dict[str, List[float]]] = {}
        for rec in pass_records:
            meta = rec.get("metadata", {}) or {}
            params = rec.get("params", {}) or {}
            reg = str(meta.get("regulator", "?"))
            field = str(meta.get("field_content", "?"))
            mode = str(params.get("thermal_mode", "?"))
            key = (reg, field, mode)
            bucket = grouped.setdefault(key, {"beta": [], "t_star": [], "power_index": [], "count": 0})
            bucket["beta"].append(float(params.get("beta", 0.0)))
            bucket["t_star"].append(float(params.get("T_star", 0.0)))
            bucket["power_index"].append(float(params.get("power_index", 0.0)))
            bucket["count"] += 1

        def _rng(values: List[float]) -> str:
            return f"[{min(values):.3g}, {max(values):.3g}]" if values else "n/a"

        for (reg, field, mode), bucket in sorted(grouped.items()):
            print(
                f"regulator={reg}, field={field}, mode={mode}: "
                f"beta {_rng(bucket['beta'])}, T_star {_rng(bucket['t_star'])}, power {_rng(bucket['power_index'])} "
                f"(passes={bucket['count']})"
            )
    else:
        print("\nNo passing configurations found in the scanned grid.")


if __name__ == "__main__":
    main()
