"""GW vs EM propagation prediction in PBUF."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np

from cosmos2.models.lcdm.utils import C_LIGHT
from cosmos2.parameters.central_authority import MPC_TO_KM

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable


DEFAULT_ZMAX = 5.0
DEFAULT_POINTS = 200


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-based cumulative trapezoidal integral."""

    if y.size == 0:
        return y.copy()

    cumulative = np.zeros_like(y, dtype=float)
    for idx in range(1, y.size):
        dx = x[idx] - x[idx - 1]
        cumulative[idx] = cumulative[idx - 1] + 0.5 * dx * (y[idx] + y[idx - 1])
    return cumulative


def _parse_z_keys(raw: object | None) -> list[float]:
    """Convert a CLI/config z_key entry into a list of floats."""

    if raw is None:
        return []

    if isinstance(raw, str):
        tokens = re.split(r"[,\s]+", raw.strip())
        return [float(tok) for tok in tokens if tok]

    if isinstance(raw, Sequence):
        values: list[float] = []
        for item in raw:
            if item is None:
                continue
            if isinstance(item, str) and not item.strip():
                continue
            values.append(float(item))
        return values

    raise ValueError("z_key must be a comma-separated list or a sequence of numbers.")


def _evaluate_wave_speed_callable(func: Iterable[float], values: np.ndarray) -> np.ndarray:
    """Try to evaluate a user-provided wave-speed helper over the grid."""

    try:
        computed = np.asarray(func(values), dtype=float)
    except Exception:
        computed = np.asarray([float(func(float(val))) for val in values], dtype=float)
    return computed


def _resolve_wave_speed_ratio(
    model: PredictionModelAdapter, z_grid: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Return c_eff(z)/c sampled across the grid and a flag if wave-speed was used."""

    ratio = np.ones_like(z_grid, dtype=float)
    used_wave_speed = False
    raw_model = model.raw_model

    wave_speed_func = getattr(raw_model, "wave_speed_of_z", None)
    if callable(wave_speed_func):
        try:
            candidate = _evaluate_wave_speed_callable(wave_speed_func, z_grid)
            candidate = np.nan_to_num(candidate, nan=1.0, posinf=1.0, neginf=1.0)
            if np.nanmax(candidate) > 1e4:
                candidate = candidate / float(C_LIGHT)
            ratio = np.clip(candidate, 0.0, np.inf)
            used_wave_speed = True
        except Exception:
            used_wave_speed = False

    if not used_wave_speed:
        try:
            a_grid = np.clip(1.0 / (1.0 + z_grid), 1e-12, np.inf)
            stiffness = np.asarray(model.elastic_stiffness(a_grid), dtype=float)
            stiffness = np.nan_to_num(stiffness, nan=1.0, posinf=1.0, neginf=1.0)
            ratio = np.clip(stiffness, 0.0, np.inf)
            used_wave_speed = True
        except AttributeError:
            ratio = np.ones_like(z_grid, dtype=float)
            used_wave_speed = False

    return ratio, used_wave_speed


@register_prediction
class GWPropagationPrediction(PredictionModule):
    name = "gw-propagation"
    version = "v1"
    description = "Luminosity distance and arrival-time comparison for GW vs EM signals."

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument(
            "--zmax",
            type=float,
            default=DEFAULT_ZMAX,
            help="Maximum redshift for the prediction (default: 5.0).",
        )
        parser.add_argument(
            "--points",
            type=int,
            default=DEFAULT_POINTS,
            help="Number of redshift samples (default: 200).",
        )
        parser.add_argument(
            "--z-key",
            type=str,
            default=None,
            help="Comma-separated redshifts for summary output.",
        )
        parser.add_argument(
            "--anchor-equal-c0",
            dest="anchor_equal_c0",
            action="store_true",
            help="Enforce c_EM(z=0)=c_GW(z=0) (default behavior).",
        )
        parser.add_argument(
            "--no-anchor-equal-c0",
            dest="anchor_equal_c0",
            action="store_false",
            help="Allow c_EM(z=0) and c_GW(z=0) to differ.",
        )
        parser.add_argument("--output-table", action="store_true", help="Export the GW/EM summary table.")
        parser.add_argument("--output-plot", action="store_true", help="Include canonical plots for this module.")
        parser.set_defaults(anchor_equal_c0=True)
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        zmax = float(config.get("zmax", DEFAULT_ZMAX))
        if zmax <= 0.0:
            raise ValueError("zmax must be positive.")

        points = max(int(config.get("points", DEFAULT_POINTS)), 2)
        anchor_equal_c0 = bool(config.get("anchor_equal_c0", True))
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))
        z_keys = _parse_z_keys(config.get("z_key"))

        if z_keys and (min(z_keys) < 0.0 or max(z_keys) > zmax):
            raise ValueError("Each z_key entry must be between 0 and zmax.")

        z_grid = np.linspace(0.0, zmax, points, dtype=float)

        try:
            H_vals = np.asarray(model.H_of_z(z_grid), dtype=float)
        except AttributeError:
            metadata = {
                "model": type(model.raw_model).__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": "missing_H_api",
                "summary": "GW propagation prediction unsupported (missing H(z) API).",
            }
            return PredictionResult(
                name=self.name,
                version=self.version,
                metadata=metadata,
                results={},
                tables=[],
                plots=[],
                status="error",
            )

        if H_vals.shape != z_grid.shape:
            raise RuntimeError("Model H(z) grid size mismatch.")

        H_safe = np.clip(np.nan_to_num(H_vals, nan=0.0, posinf=0.0, neginf=0.0), 1e-30, None)

        ceff_ratio, used_wave_speed = _resolve_wave_speed_ratio(model, z_grid)
        if anchor_equal_c0 and ceff_ratio.size:
            anchor = float(ceff_ratio[0])
            if not np.isfinite(anchor) or anchor <= 0.0:
                anchor = 1.0
            ceff_ratio = ceff_ratio / anchor

        delta_c = np.zeros_like(ceff_ratio)
        rc_series = 1.0 + delta_c
        speed_ratio_gw = ceff_ratio * rc_series

        dl_integrand_em = ceff_ratio / H_safe
        dl_integrand_gw = speed_ratio_gw / H_safe
        dl_cumulative_em = _cumulative_trapezoid(dl_integrand_em, z_grid)
        dl_cumulative_gw = _cumulative_trapezoid(dl_integrand_gw, z_grid)
        dl_em = (1.0 + z_grid) * float(C_LIGHT) * dl_cumulative_em
        dl_gw = (1.0 + z_grid) * float(C_LIGHT) * dl_cumulative_gw

        rd_series = np.ones_like(z_grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            rd_series = np.divide(dl_gw, dl_em, out=np.ones_like(dl_em), where=dl_em != 0.0)

        rc_values = np.ones_like(z_grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            rc_values = np.divide(speed_ratio_gw, ceff_ratio, out=np.ones_like(speed_ratio_gw), where=ceff_ratio != 0.0)

        t_integrand_base = 1.0 / ((1.0 + z_grid) * H_safe)
        t_integrand_gw = t_integrand_base * np.divide(
            ceff_ratio, speed_ratio_gw, out=np.ones_like(ceff_ratio), where=speed_ratio_gw != 0.0
        )
        t_em = MPC_TO_KM * _cumulative_trapezoid(t_integrand_base, z_grid)
        t_gw = MPC_TO_KM * _cumulative_trapezoid(t_integrand_gw, z_grid)
        delta_t = t_gw - t_em

        def _interpolate(values: np.ndarray) -> list[float]:
            if not z_keys:
                return []
            return np.interp(np.asarray(z_keys, dtype=float), z_grid, values).tolist()

        rd_summary = _interpolate(rd_series)
        dl_em_summary = _interpolate(dl_em)
        dl_gw_summary = _interpolate(dl_gw)
        delta_t_summary = _interpolate(delta_t)
        rc_summary = _interpolate(rc_values)

        tables: list[PredictionTable] = []
        if output_table:
            rows = [
                [
                    float(z),
                    float(dl_em_val),
                    float(dl_gw_val),
                    float(rd_val),
                    float(t_em_val),
                    float(t_gw_val),
                    float(delta_val),
                    float(rc_val),
                ]
                for z, dl_em_val, dl_gw_val, rd_val, t_em_val, t_gw_val, delta_val, rc_val in zip(
                    z_grid, dl_em, dl_gw, rd_series, t_em, t_gw, delta_t, rc_values
                )
            ]
            tables.append(
                PredictionTable(
                    name="gw_propagation_vs_z",
                    columns=[
                        "z",
                        "DL_EM_Mpc",
                        "DL_GW_Mpc",
                        "R_D",
                        "t_EM_s",
                        "t_GW_s",
                        "Delta_t_s",
                        "R_c",
                    ],
                    rows=rows,
                    metadata={
                        "points": len(z_grid),
                        "zmax": float(zmax),
                        "anchor_equal_c0": anchor_equal_c0,
                        "used_wave_speed": used_wave_speed,
                    },
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            plots.append(
                PredictionPlot(
                    name="DL_ratio_vs_z",
                    description="GW/EM luminosity distance ratio",
                    data={"z": z_grid.tolist(), "R_D": rd_series.tolist()},
                    metadata={"xlabel": "redshift z", "ylabel": "D_L^GW / D_L^EM"},
                )
            )
            plots.append(
                PredictionPlot(
                    name="Delta_t_vs_z",
                    description="Arrival time difference between GW and EM signals",
                    data={"z": z_grid.tolist(), "Delta_t_s": delta_t.tolist()},
                    metadata={"xlabel": "redshift z", "ylabel": "Delta_t_GW-EM [s]"},
                )
            )

        results = {
            "zmax": float(zmax),
            "points": points,
            "z_keys": list(z_keys),
            "RD_at_z": rd_summary,
            "DL_EM_at_z": dl_em_summary,
            "DL_GW_at_z": dl_gw_summary,
            "Delta_t_GW_EM_at_z": delta_t_summary,
            "Rc_at_z": rc_summary,
        }

        metadata = {
            "model": type(model.raw_model).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": "GW vs EM propagation in PBUF: luminosity-distance ratio and arrival-time difference",
            "anchor_equal_c0": anchor_equal_c0,
            "used_wave_speed": used_wave_speed,
            "zmax": float(zmax),
            "points": points,
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )
