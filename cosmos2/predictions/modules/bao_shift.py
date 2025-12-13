"""Prediction module for BAO shift (D_V/r_d and α parameters)."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from cosmos2.models.model_factory import create_model as create_cosmos2_model

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable

_DEFAULT_ZMIN = 0.1
_DEFAULT_ZMAX = 2.0
_DEFAULT_POINTS = 10
_STANDARD_PIVOTS = (0.35, 0.57)


def _safe_float(value: float | np.ndarray | None) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _series_to_list(values: Sequence[float]) -> list[float | None]:
    return [_safe_float(val) for val in np.asarray(values, dtype=float)]


def _build_z_grid(zmin: float, zmax: float, points: int) -> np.ndarray:
    zmin_val = float(zmin)
    zmax_val = float(zmax)
    if zmin_val < 0.0:
        raise ValueError("zmin must be non-negative.")
    if zmax_val < zmin_val:
        raise ValueError("zmax must be greater than or equal to zmin.")
    if points < 2:
        raise ValueError("points must be at least 2.")
    return np.linspace(zmin_val, zmax_val, num=points, dtype=float)


def _build_z_samples(z_input: float | None, zmin: float, zmax: float, points: int) -> np.ndarray:
    if z_input is not None:
        pivot = float(z_input)
        if pivot < 0.0:
            raise ValueError("z must be non-negative.")
        samples = np.array([pivot, *_STANDARD_PIVOTS], dtype=float)
        samples = np.clip(samples, 0.0, np.inf)
        unique = np.unique(samples)
        unique.sort()
        return unique
    return _build_z_grid(zmin, zmax, points)


def _select_pivot(z_input: float | None, grid: np.ndarray) -> float:
    if z_input is not None:
        return float(z_input)
    if grid.size == 0:
        return float(_STANDARD_PIVOTS[-1])
    for candidate in _STANDARD_PIVOTS:
        if grid[0] - 1e-6 <= candidate <= grid[-1] + 1e-6:
            return float(candidate)
    return float(grid[len(grid) // 2])


def _compute_volume_distance(z: np.ndarray, DM: np.ndarray, H: np.ndarray, c_km_s: float) -> np.ndarray:
    safe_DM = np.clip(np.asarray(DM, dtype=float), 0.0, np.inf)
    safe_H = np.clip(np.asarray(H, dtype=float), 1e-12, np.inf)
    safe_z = np.clip(z, 0.0, np.inf)
    inside = (1.0 + z) ** 2 * safe_DM ** 2 * c_km_s * safe_z / safe_H
    inside = np.clip(inside, 0.0, np.inf)
    return np.power(inside, 1.0 / 3.0)


def _evaluate_series(adapter: PredictionModelAdapter, z_values: np.ndarray, r_d: float, c_km_s: float) -> dict[str, np.ndarray]:
    z_arr = np.asarray(z_values, dtype=float)
    DM = adapter.comoving_distance(z_arr)
    H = adapter.H_of_z(z_arr)
    DV = _compute_volume_distance(z_arr, DM, H, c_km_s)
    if r_d > 0.0:
        dv_over_rd = DV / r_d
    else:
        dv_over_rd = np.full_like(DV, np.nan)
    return {"z": z_arr, "DM": DM, "H": H, "DV": DV, "DV_over_rd": dv_over_rd}


def _evaluate_scalar(adapter: PredictionModelAdapter, z: float, r_d: float, c_km_s: float) -> dict[str, float]:
    series = _evaluate_series(adapter, np.array([z], dtype=float), r_d, c_km_s)
    return {
        key: float(series[key][0]) for key in ("DM", "H", "DV", "DV_over_rd")
    }


def _empty_series(length: int) -> dict[str, np.ndarray]:
    nan_arr = np.full(length, np.nan, dtype=float)
    return {"z": nan_arr, "DM": nan_arr, "H": nan_arr, "DV": nan_arr, "DV_over_rd": nan_arr}


def _build_lcdm_reference(adapter: PredictionModelAdapter) -> PredictionModelAdapter:
    params = adapter.parameters
    lcdm_kwargs = {
        "H0": float(params.get("H0", 67.4)),
        "Omega_m0": float(params.get("Omega_m0", 0.315)),
        "Omega_b0": float(params.get("Omega_b0", 0.0)),
        "Omega_r0": float(params.get("Omega_r0", 9.0e-5)),
        "Omega_k0": float(params.get("Omega_k0", 0.0)),
    }
    lcdm_model = create_cosmos2_model("lcdm", **lcdm_kwargs)
    return PredictionModelAdapter(lcdm_model)


def _missing_bao_result() -> PredictionResult:
    return PredictionResult(
        name="bao-shift",
        version="v1",
        metadata={"error": "missing_H_or_distance_or_rd"},
        results={},
        tables=[],
        plots=[],
        status="error",
    )


@register_prediction
class BAOShiftPrediction(PredictionModule):
    name = "bao-shift"
    version = "v1"
    description = "Compute BAO distance ratios and α-parameters for a PBUF model."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument("--z", type=float, help="Redshift at which to report a headline BAO shift.")
        parser.add_argument("--zmin", type=float, default=_DEFAULT_ZMIN, help="Minimum redshift for the sample grid.")
        parser.add_argument("--zmax", type=float, default=_DEFAULT_ZMAX, help="Maximum redshift for the sample grid.")
        parser.add_argument("--points", type=int, default=_DEFAULT_POINTS, help="Number of redshift samples (ignored when --z is provided).")
        parser.add_argument("--compare-lcdm", action="store_true", help="Build a ΛCDM reference and compute α ratios.")
        parser.add_argument("--output-table", action="store_true", help="Export a table of BAO quantities vs. redshift.")
        parser.add_argument("--output-plot", action="store_true", help="Emit BAO shift plots.")
        super().register(parser)

    def run_prediction(
        self, model: PredictionModelAdapter, config: dict[str, object]
    ) -> PredictionResult:
        z_input = config.get("z")
        pivot_flag = float(z_input) if z_input is not None else None
        zmin = float(config.get("zmin", _DEFAULT_ZMIN))
        zmax = float(config.get("zmax", _DEFAULT_ZMAX))
        points = max(int(config.get("points", _DEFAULT_POINTS)), 2)
        compare_lcdm = bool(config.get("compare_lcdm"))
        output_table = bool(config.get("output_table"))
        output_plot = bool(config.get("output_plot"))

        try:
            z_values = _build_z_samples(pivot_flag, zmin, zmax, points)
            pivot_z = _select_pivot(pivot_flag, z_values)
            c_km_s = float(model.constants.get("c_km_per_s", 299_792.458))
            r_d_model = float(model.sound_horizon())
            if r_d_model <= 0.0:
                raise ValueError("Model reports a non-positive sound horizon.")
            model_series = _evaluate_series(model, z_values, r_d_model, c_km_s)
        except AttributeError:
            return _missing_bao_result()

        reference_adapter: PredictionModelAdapter | None = None
        r_d_lcdm: float | None = None
        reference_series = _empty_series(z_values.size)

        if compare_lcdm:
            try:
                reference_adapter = _build_lcdm_reference(model)
                r_d_lcdm = float(reference_adapter.sound_horizon())
                if r_d_lcdm > 0.0:
                    c_lcdm = float(reference_adapter.constants.get("c_km_per_s", c_km_s))
                    reference_series = _evaluate_series(reference_adapter, z_values, r_d_lcdm, c_lcdm)
                else:
                    reference_adapter = None
            except Exception:
                reference_adapter = None
                reference_series = _empty_series(z_values.size)
                r_d_lcdm = None

        dv_model = model_series["DV"]
        dm_model = model_series["DM"]
        h_model = model_series["H"]
        dv_over_rd_model = model_series["DV_over_rd"]

        dv_ref = reference_series["DV"]
        dm_ref = reference_series["DM"]
        h_ref = reference_series["H"]
        dv_over_rd_ref = reference_series["DV_over_rd"]

        alpha_iso_series = np.full(z_values.shape, np.nan, dtype=float)
        alpha_perp_series = np.full(z_values.shape, np.nan, dtype=float)
        alpha_parallel_series = np.full(z_values.shape, np.nan, dtype=float)

        reference_available = (
            reference_adapter is not None
            and r_d_lcdm is not None
            and r_d_lcdm > 0.0
            and r_d_model > 0.0
        )

        if reference_available:
            ratio_model = dm_model / r_d_model
            ratio_ref = dm_ref / r_d_lcdm
            valid_perp = (ratio_ref > 0.0) & np.isfinite(ratio_ref)
            alpha_perp_series[valid_perp] = ratio_model[valid_perp] / ratio_ref[valid_perp]

            valid_parallel = (
                (h_model > 0.0)
                & (h_ref > 0.0)
                & np.isfinite(h_model)
                & np.isfinite(h_ref)
            )
            denom = h_model * r_d_model
            numer = h_ref * r_d_lcdm
            safe_denom = np.clip(denom, 1e-12, np.inf)
            alpha_parallel_series[valid_parallel] = numer[valid_parallel] / safe_denom[valid_parallel]

            valid_iso = np.isfinite(alpha_perp_series) & np.isfinite(alpha_parallel_series)
            alpha_iso_series[valid_iso] = (
                np.power(alpha_perp_series[valid_iso], 2.0 / 3.0)
                * np.power(alpha_parallel_series[valid_iso], 1.0 / 3.0)
            )

        table_rows = []
        for (
            z_val,
            dv_p,
            dv_l,
            dv_over_p,
            dv_over_l,
            dm_p,
            dm_l,
            h_p,
            h_l,
            alpha_iso,
            alpha_perp,
            alpha_parallel,
        ) in zip(
            z_values,
            dv_model,
            dv_ref,
            dv_over_rd_model,
            dv_over_rd_ref,
            dm_model,
            dm_ref,
            h_model,
            h_ref,
            alpha_iso_series,
            alpha_perp_series,
            alpha_parallel_series,
        ):
            table_rows.append(
                [
                    _safe_float(z_val),
                    _safe_float(dv_p),
                    _safe_float(dv_l),
                    _safe_float(dv_over_p),
                    _safe_float(dv_over_l),
                    _safe_float(dm_p),
                    _safe_float(dm_l),
                    _safe_float(h_p),
                    _safe_float(h_l),
                    _safe_float(alpha_iso),
                    _safe_float(alpha_perp),
                    _safe_float(alpha_parallel),
                ]
            )

        tables: list[PredictionTable] = []
        if output_table:
            tables.append(
                PredictionTable(
                    name="bao_shift_vs_z",
                    columns=[
                        "z",
                        "DV_PBUF",
                        "DV_LCDM",
                        "DV_over_rd_PBUF",
                        "DV_over_rd_LCDM",
                        "DM_PBUF",
                        "DM_LCDM",
                        "H_PBUF",
                        "H_LCDM",
                        "alpha_iso",
                        "alpha_perp",
                        "alpha_parallel",
                    ],
                    rows=table_rows,
                    metadata={"points": len(table_rows)},
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            dv_over_rd_plot = PredictionPlot(
                name="DV_over_rd_vs_z",
                data={"z": z_values.tolist(), "DV_over_rd": _series_to_list(dv_over_rd_model)},
                description="Volume-averaged BAO distance D_V(z) divided by r_d (PBUF).",
                metadata={"xlabel": "redshift z", "ylabel": "D_V(z)/r_d (PBUF)"},
            )
            plots.append(dv_over_rd_plot)

            if reference_available:
                alpha_iso_list = _series_to_list(alpha_iso_series)
                if any(value is not None for value in alpha_iso_list):
                    plots.append(
                        PredictionPlot(
                            name="alpha_iso_vs_z",
                            data={"z": z_values.tolist(), "alpha_iso": alpha_iso_list},
                            description="Isotropic BAO shift α_iso(z) relative to ΛCDM.",
                            metadata={"xlabel": "redshift z", "ylabel": "α_iso(z) = D_V/r_d (PBUF)/ref"},
                        )
                    )

        model_pivot = _evaluate_scalar(model, pivot_z, r_d_model, c_km_s)
        reference_pivot = (
            _evaluate_scalar(reference_adapter, pivot_z, r_d_lcdm, float(reference_adapter.constants.get("c_km_per_s", c_km_s)))
            if reference_available and reference_adapter is not None and r_d_lcdm is not None
            else None
        )

        pivot_alpha_iso = pivot_alpha_perp = pivot_alpha_parallel = None
        if reference_pivot is not None and reference_available:
            dm_ratio_model = model_pivot["DM"] / r_d_model
            dm_ratio_ref = reference_pivot["DM"] / r_d_lcdm
            if dm_ratio_ref > 0.0:
                pivot_alpha_perp = dm_ratio_model / dm_ratio_ref
            if (
                model_pivot["H"] > 0.0
                and reference_pivot["H"] > 0.0
                and r_d_model > 0.0
                and r_d_lcdm > 0.0
            ):
                pivot_alpha_parallel = (reference_pivot["H"] * r_d_lcdm) / (model_pivot["H"] * r_d_model)
            if pivot_alpha_perp is not None and pivot_alpha_parallel is not None:
                pivot_alpha_iso = pivot_alpha_perp ** (2.0 / 3.0) * pivot_alpha_parallel ** (1.0 / 3.0)

        results = {
            "z_pivot": float(pivot_z),
            "alpha_iso_pivot": _safe_float(pivot_alpha_iso),
            "alpha_perp_pivot": _safe_float(pivot_alpha_perp),
            "alpha_parallel_pivot": _safe_float(pivot_alpha_parallel),
            "DV_over_rd_PBUF_pivot": _safe_float(model_pivot["DV_over_rd"]),
            "DV_over_rd_LCDM_pivot": _safe_float(reference_pivot["DV_over_rd"]) if reference_pivot is not None else None,
            "rd_PBUF": _safe_float(r_d_model),
            "rd_LCDM": _safe_float(r_d_lcdm) if r_d_lcdm is not None else None,
        }

        metadata = {
            "model": model.raw_model.__class__.__name__,
            "compare_lcdm": compare_lcdm,
            "z_input": float(pivot_flag) if pivot_flag is not None else None,
            "zmin": zmin,
            "zmax": zmax,
            "points": points,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )
