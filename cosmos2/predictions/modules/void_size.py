"""Prediction module for the PBUF void-size intuition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import numpy as np

from cosmos2.kernels.common.growth import solve_growth
from cosmos2.models.model_factory import create_model as create_cosmos2_model

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable

_DEFAULT_ZMAX = 1.0
_DEFAULT_POINTS = 100
_DEFAULT_R_REF = 25.0
_DEFAULT_ETA = 0.5
_DEFAULT_GAMMA = 1.0
_DEFAULT_BETA = 0.3
_MIN_A = 1e-12
_GRID_MIN_STEPS = 2048
_GRID_MAX_STEPS = 20000
_MODULE_NAME = "void-size"


def _build_growth_grid(a_targets: np.ndarray, *, min_steps: int = _GRID_MIN_STEPS, max_steps: int = _GRID_MAX_STEPS) -> np.ndarray:
    clipped = np.clip(a_targets, _MIN_A, 1.0)
    if clipped.size == 0:
        raise ValueError("Need at least one redshift sample to evaluate the growth factor.")
    a_min = float(np.min(clipped))
    start = max(min(a_min * 0.1, 1e-4), 1e-9)
    steps = int(np.clip(clipped.size * 8, min_steps, max_steps))
    return np.logspace(np.log10(start), 0.0, num=steps, dtype=float)


def _interpolate_log_space(grid: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    log_grid = np.log(grid)
    clipped_targets = np.clip(targets, grid[0], grid[-1])
    return np.interp(np.log(clipped_targets), log_grid, values)


def _interpolate_safe(z_grid: np.ndarray, values: np.ndarray, target: float) -> float:
    if target <= z_grid[0]:
        return float(values[0])
    if target >= z_grid[-1]:
        return float(values[-1])
    return float(np.interp(target, z_grid, values))


def _build_lcdm_reference(model: PredictionModelAdapter) -> PredictionModelAdapter:
    params = model.parameters
    lcdm_kwargs = {
        "H0": float(params.get("H0", 67.4)),
        "Omega_m0": float(params.get("Omega_m0", 0.315)),
        "Omega_b0": float(params.get("Omega_b0", 0.049)),
        "Omega_r0": float(params.get("Omega_r0", 9.0e-5)),
        "Omega_k0": float(params.get("Omega_k0", 0.0)),
    }
    lcdm_model = create_cosmos2_model("lcdm", **lcdm_kwargs)
    return PredictionModelAdapter(lcdm_model)


def _solve_growth_with_adapter(model: PredictionModelAdapter, z_values: np.ndarray) -> np.ndarray:
    if z_values.size == 0:
        return np.array([], dtype=float)
    a_targets = 1.0 / np.clip(1.0 + z_values, _MIN_A, np.inf)
    a_grid = _build_growth_grid(a_targets)
    H_grid = model.H(a_grid)
    if H_grid.shape != a_grid.shape:
        raise RuntimeError("H(a) grid mismatch while solving growth.")
    H0 = float(model.parameters.get("H0", 67.4))
    if H0 <= 0.0 or not np.isfinite(H0):
        raise ValueError("Model reports non-positive H0.")
    E_grid = np.clip(np.asarray(H_grid, dtype=float) / H0, 1e-12, np.inf)
    Omega_m0 = float(model.parameters.get("Omega_m0", 0.3))
    D_grid, _ = solve_growth(a_grid, E_grid, omega_m0=Omega_m0)
    return _interpolate_log_space(a_grid, D_grid, a_targets)


def _growth_series(model: PredictionModelAdapter, z_values: np.ndarray) -> np.ndarray:
    raw = model.raw_model
    candidate = getattr(raw, "growth_factor", None)
    if callable(candidate):
        try:
            return np.asarray(candidate(z_values), dtype=float)
        except Exception:
            pass
    return _solve_growth_with_adapter(model, z_values)


def _missing_alpha_result() -> PredictionResult:
    return PredictionResult(
        name=_MODULE_NAME,
        version="v1",
        metadata={"error": "missing_alpha"},
        results={},
        tables=[],
        plots=[],
        status="error",
    )


def _missing_growth_result() -> PredictionResult:
    return PredictionResult(
        name=_MODULE_NAME,
        version="v1",
        metadata={"error": "missing_growth_api"},
        results={},
        tables=[],
        plots=[],
        status="error",
    )


@register_prediction
class VoidSizePrediction(PredictionModule):
    name = _MODULE_NAME
    version = "v1"
    description = "Predict PBUF void sizes via elastic + growth scaling."

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument("--zmax", type=float, default=_DEFAULT_ZMAX, help="Maximum redshift to sample.")
        parser.add_argument("--points", type=int, default=_DEFAULT_POINTS, help="Number of redshift samples.")
        parser.add_argument(
            "--compare-lcdm",
            action="store_true",
            help="Include the ΛCDM-style reference ratios in the output.",
        )
        parser.add_argument(
            "--R_ref_Mpc",
            type=float,
            default=_DEFAULT_R_REF,
            help="Present-day reference void radius in Mpc.",
        )
        parser.add_argument(
            "--eta-growth",
            type=float,
            default=_DEFAULT_ETA,
            help="Exponent controlling D(z) scaling (eta).",
        )
        parser.add_argument(
            "--gamma-alpha",
            type=float,
            default=_DEFAULT_GAMMA,
            help="Elastic slack strength multiplier (gamma_alpha).",
        )
        parser.add_argument(
            "--beta-z",
            type=float,
            default=_DEFAULT_BETA,
            help="Redshift exponent for the reference ΛCDM void curve.",
        )
        parser.add_argument("--output-table", action="store_true", help="Emit the void-radius table.")
        parser.add_argument("--output-plot", action="store_true", help="Emit the void-size/ratio plots.")
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        zmax = float(config.get("zmax", _DEFAULT_ZMAX))
        if zmax <= 0.0:
            raise ValueError("zmax must be positive.")
        points = max(int(config.get("points", _DEFAULT_POINTS)), 2)
        compare_lcdm = bool(config.get("compare_lcdm", False))
        R_ref_0 = float(config.get("R_ref_Mpc", _DEFAULT_R_REF))
        eta = float(config.get("eta_growth", _DEFAULT_ETA))
        gamma = float(config.get("gamma_alpha", _DEFAULT_GAMMA))
        beta = float(config.get("beta_z", _DEFAULT_BETA))
        output_plot = bool(config.get("output_plot", False))
        output_table = bool(config.get("output_table", False))

        try:
            alpha = float(model.alpha)
        except AttributeError:
            return _missing_alpha_result()

        z_grid = np.linspace(0.0, zmax, points, dtype=float)
        a_grid = 1.0 / np.clip(1.0 + z_grid, _MIN_A, np.inf)

        try:
            D_pbuf = _growth_series(model, z_grid)
        except Exception:
            return _missing_growth_result()

        if compare_lcdm:
            lcdm_adapter = _build_lcdm_reference(model)
            try:
                D_lcdm = _growth_series(lcdm_adapter, z_grid)
            except Exception:
                return _missing_growth_result()
        else:
            D_lcdm = np.full_like(D_pbuf, np.nan, dtype=float)

        safe_D_pbuf = np.clip(D_pbuf, 1e-12, np.inf)
        if compare_lcdm:
            safe_D_lcdm = np.clip(D_lcdm, 1e-12, np.inf)
            S_growth = np.power(safe_D_lcdm / safe_D_pbuf, eta)
        else:
            S_growth = np.power(safe_D_pbuf, -eta)

        S_elastic = 1.0 + gamma * alpha
        R_ref = R_ref_0 * np.power(1.0 + z_grid, -beta)
        R_void = R_ref * S_growth * S_elastic
        R_void = np.nan_to_num(R_void, nan=0.0, posinf=0.0, neginf=0.0)

        ratio_series = (
            np.divide(R_void, R_ref, out=np.zeros_like(R_void), where=R_ref != 0.0)
            if compare_lcdm
            else None
        )

        table_rows: list[list[object]] = []
        for idx, (z_val, a_val, D_pb, D_lc, Sg, Rv, Rb) in enumerate(
            zip(z_grid, a_grid, D_pbuf, D_lcdm, S_growth, R_void, R_ref)
        ):
            ratio_value = float(ratio_series[idx]) if ratio_series is not None else None
            table_rows.append(
                [
                    float(z_val),
                    float(a_val),
                    float(D_pb),
                    float(D_lc) if compare_lcdm else None,
                    float(Sg),
                    float(S_elastic),
                    float(Rv),
                    float(Rb),
                    ratio_value,
                ]
            )

        tables: list[PredictionTable] = []
        if output_table:
            tables.append(
                PredictionTable(
                    name="void_radius_vs_z",
                    columns=[
                        "z",
                        "a",
                        "D_PBUF",
                        "D_LCDM",
                        "S_growth",
                        "S_elastic",
                        "R_PBUF_Mpc",
                        "R_LCDM_Mpc",
                        "ratio",
                    ],
                    rows=table_rows,
                    metadata={
                        "points": len(z_grid),
                        "alpha": alpha,
                        "compare_lcdm": compare_lcdm,
                    },
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            plots.append(
                PredictionPlot(
                    name="R_void_vs_z",
                    data={
                        "z": z_grid.tolist(),
                        "R_PBUF_Mpc": R_void.tolist(),
                        "R_ref_Mpc": R_ref.tolist(),
                    },
                    metadata={"xlabel": "redshift z", "ylabel": "R_void [Mpc]"},
                )
            )
            if compare_lcdm and ratio_series is not None:
                plots.append(
                    PredictionPlot(
                        name="void_ratio_vs_z",
                        data={"z": z_grid.tolist(), "ratio": ratio_series.tolist()},
                        metadata={"xlabel": "redshift z", "ylabel": "PBUF / LCDM void size"},
                    )
                )

        R_void_z0 = _interpolate_safe(z_grid, R_void, 0.0)
        R_void_z05 = _interpolate_safe(z_grid, R_void, 0.5)
        R_void_z1 = _interpolate_safe(z_grid, R_void, 1.0)
        R_ref_z0 = _interpolate_safe(z_grid, R_ref, 0.0)

        if ratio_series is not None:
            ratio_z0 = _interpolate_safe(z_grid, ratio_series, 0.0)
            ratio_z05 = _interpolate_safe(z_grid, ratio_series, 0.5)
            ratio_z1 = _interpolate_safe(z_grid, ratio_series, 1.0)
        else:
            ratio_z0 = None
            ratio_z05 = None
            ratio_z1 = None

        results = {
            "zmax": zmax,
            "R_void_z0_Mpc": R_void_z0,
            "R_void_z0p5_Mpc": R_void_z05,
            "R_void_z1_Mpc": R_void_z1,
            "R_ref_z0_Mpc": R_ref_z0,
            "ratio_PBUF_over_LCDM_z0": ratio_z0,
            "ratio_PBUF_over_LCDM_z0p5": ratio_z05,
            "ratio_PBUF_over_LCDM_z1": ratio_z1,
            "eta_growth": eta,
            "gamma_alpha": gamma,
            "alpha": alpha,
        }

        metadata = {
            "model": model.raw_model.__class__.__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "points": points,
            "compare_lcdm": compare_lcdm,
            "R_ref_Mpc": R_ref_0,
            "eta_growth": eta,
            "gamma_alpha": gamma,
            "beta_z": beta,
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )
