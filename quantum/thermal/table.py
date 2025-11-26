"""
Programmatic thermal table generation for the Quantum -> Cosmos handoff.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

T_CMB0 = 2.7255

THERMAL_MODES = ("off", "linear", "power", "exp")
PHASE_HINT_THRESHOLDS = (
    (1.0e10, "BBN"),
    (1.0e8, "QCD"),
    (4.0e3, "recombination"),
)


class ThermalGenerationError(RuntimeError):
    """Raised when the exporter inputs are invalid."""


@dataclass(slots=True)
class ThermalModelConfig:
    mode: str
    beta: float = 0.05
    t_star: float = 1.0e6
    power: float = 1.0
    alpha_qm: float = 0.03
    eps_min: float = 1.0e-4

    def __post_init__(self) -> None:
        mode_norm = self.mode.lower()
        if mode_norm not in THERMAL_MODES:
            raise ThermalGenerationError(f"Unsupported thermal mode '{self.mode}'.")
        self.mode = mode_norm
        if self.t_star <= 0.0:
            raise ThermalGenerationError("Reference temperature T* must be positive.")
        if self.beta < 0.0:
            raise ThermalGenerationError("β must be non-negative.")
        if self.power <= 0.0:
            raise ThermalGenerationError("Power index p must be positive.")
        if self.alpha_qm <= 0.0:
            raise ThermalGenerationError("α_QM must be positive.")
        if self.eps_min <= 0.0:
            raise ThermalGenerationError("ε_min must be positive.")


@dataclass(slots=True)
class ThermalTableSpec:
    model: ThermalModelConfig
    t_min: float = 2.725
    t_max: float = 1.0e12
    num_points: int = 512
    dense_points: int = 24
    table_version: int = 11
    method_version: int = 11
    regulator: str = "thermal_default"
    field_content: str = "SM_full"
    f_cut_T: float = 1.0e12
    f_coup_T: float = 1.0e8
    notes: str = "auto-generated via exporter"

    def __post_init__(self) -> None:
        if self.t_min <= 0.0 or self.t_max <= 0.0:
            raise ThermalGenerationError("Temperature bounds must be positive.")
        if self.t_min >= self.t_max:
            raise ThermalGenerationError("t_min must be smaller than t_max.")
        if self.num_points < 32:
            raise ThermalGenerationError("At least 32 base samples are required.")
        if self.method_version < 11:
            raise ThermalGenerationError("method_version must be >= 11.")
        if self.table_version < 11:
            raise ThermalGenerationError("table_version must be >= 11.")


@dataclass(slots=True)
class ThermalTableRow:
    z: float
    a: float
    T_K: float
    epsilon0_T: float
    alpha_T: float
    dln_epsilon0_dlnT: float
    dln_alpha_dlnT: float
    g_star: float
    g_starS: float
    sigma_epsilon0: float
    sigma_alpha: float
    cov_epsilon0_alpha: float
    validity_flag: str
    T_range_tag: str
    phase6a_hint: str
    z_from_T: float
    notes: str
    units: str
    hash_row: str


@dataclass(slots=True)
class ThermalTable:
    rows: List[ThermalTableRow]
    metadata: Dict[str, object]


def _logspace_with_refinements(t_min: float, t_max: float, num: int, dense_points: int) -> np.ndarray:
    base = np.logspace(np.log10(t_min), np.log10(t_max), num=num, endpoint=True)
    anchors = [threshold for threshold, _ in PHASE_HINT_THRESHOLDS]
    anchors.extend([T_CMB0 * (1.0 + 0.0), 10.0])
    refinements: List[float] = []
    span = 3.0
    for anchor in anchors:
        if anchor <= t_min or anchor >= t_max:
            continue
        left = max(anchor / span, t_min)
        right = min(anchor * span, t_max)
        refined = np.logspace(np.log10(left), np.log10(right), num=max(4, dense_points))
        refinements.append(refined)
    if refinements:
        mesh = np.unique(np.concatenate([base, *refinements]))
    else:
        mesh = base
    mesh.sort()
    return mesh


def _phase6a_hint(temp: float) -> str:
    for threshold, label in PHASE_HINT_THRESHOLDS:
        if temp >= threshold:
            return label
    return "late"


def _g_star(temp: float) -> float:
    if temp >= 1.0e11:
        return 106.75
    if temp >= 1.0e9:
        return 90.0
    if temp >= 1.0e7:
        return 60.0
    if temp >= 1.0e5:
        return 20.0
    if temp >= 1.0e3:
        return 10.75
    if temp >= 10.0:
        return 7.0
    return 3.36


def _g_star_s(temp: float) -> float:
    if temp >= 1.0e11:
        return 106.75
    if temp >= 1.0e9:
        return 85.0
    if temp >= 1.0e7:
        return 50.0
    if temp >= 1.0e5:
        return 18.0
    if temp >= 1.0e3:
        return 10.75
    if temp >= 10.0:
        return 6.5
    return 3.90


def _epsilon_mode(mode: str, temps: np.ndarray, cfg: ThermalModelConfig) -> tuple[np.ndarray, np.ndarray]:
    ratio = temps / cfg.t_star
    if mode == "off":
        eps = np.ones_like(temps)
        dln = np.zeros_like(temps)
        return eps, dln

    if mode == "linear":
        eps = 1.0 - cfg.beta * ratio
        dln = -(cfg.beta * ratio) / np.maximum(eps, cfg.eps_min)
    elif mode == "power":
        powered = np.power(ratio, cfg.power)
        eps = 1.0 - cfg.beta * powered
        dln = -(cfg.beta * cfg.power * powered) / np.maximum(eps, cfg.eps_min)
    elif mode == "exp":
        powered = np.power(ratio, cfg.power)
        eps = np.exp(-cfg.beta * powered)
        dln = -cfg.beta * cfg.power * powered
    else:  # pragma: no cover - validated upstream
        raise ThermalGenerationError(f"Unhandled thermal mode '{mode}'.")

    clamped = eps < cfg.eps_min
    if np.any(clamped):
        eps = np.where(clamped, cfg.eps_min, eps)
        dln = np.where(clamped, 0.0, dln)
    return eps, dln


def _row_hash(payload: Dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_common_fields(temp: float) -> Dict[str, object]:
    z = float(max(temp / T_CMB0 - 1.0, 0.0))
    a = float(1.0 / (1.0 + z))
    hint = _phase6a_hint(temp)
    range_tag = f"{hint}:{temp:.3e}"
    return {"z": z, "a": a, "phase6a_hint": hint, "T_range_tag": range_tag}


def _build_rows(mesh: np.ndarray, cfg: ThermalModelConfig) -> Iterable[Dict[str, object]]:
    eps, dln = _epsilon_mode(cfg.mode, mesh, cfg)
    alpha = cfg.alpha_qm * eps
    for temp, eps_val, dln_val, alpha_val in zip(mesh, eps, dln, alpha, strict=True):
        fields = _row_common_fields(float(temp))
        g_val = _g_star(float(temp))
        g_s_val = _g_star_s(float(temp))
        row = {
            **fields,
            "T_K": float(temp),
            "epsilon0_T": float(eps_val),
            "alpha_T": float(alpha_val),
            "dln_epsilon0_dlnT": float(dln_val),
            "dln_alpha_dlnT": float(dln_val),
            "g_star": float(g_val),
            "g_starS": float(g_s_val),
            "sigma_epsilon0": 0.0,
            "sigma_alpha": 0.0,
            "cov_epsilon0_alpha": 0.0,
            "validity_flag": "ok" if dln_val != 0.0 else "clamped" if eps_val == cfg.eps_min else "flat",
            "z_from_T": float(fields["z"]),
            "notes": "auto",
            "units": "Kelvin",
        }
        yield row


def generate_thermal_table(spec: ThermalTableSpec) -> ThermalTable:
    mesh = _logspace_with_refinements(spec.t_min, spec.t_max, spec.num_points, spec.dense_points)
    rows = []
    for row in _build_rows(mesh[::-1], spec.model):  # High->low temperature order
        row["hash_row"] = _row_hash(row)
        rows.append(ThermalTableRow(**row))

    metadata: Dict[str, object] = {
        "mode": spec.model.mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "table_version": spec.table_version,
        "method_version": spec.method_version,
        "regulator": spec.regulator,
        "field_content": spec.field_content,
        "f_cut_T": spec.f_cut_T,
        "f_coup_T": spec.f_coup_T,
        "alpha_qm": spec.model.alpha_qm,
        "beta": spec.model.beta,
        "t_star": spec.model.t_star,
        "power": spec.model.power,
        "eps_min": spec.model.eps_min,
        "t_min": spec.t_min,
        "t_max": spec.t_max,
        "num_points": spec.num_points,
        "dense_points": spec.dense_points,
        "notes": spec.notes,
    }
    return ThermalTable(rows=rows, metadata=metadata)


def save_table(table: ThermalTable, path: Path) -> None:
    payload = {
        "metadata": table.metadata,
        "rows": [asdict(row) for row in table.rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
