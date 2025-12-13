"""Fast ISW × LSS cross-correlation amplitude estimator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from cosmos2.kernels.common.growth import solve_growth
from cosmos2.models.model_factory import create_model

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable

_DEFAULT_ZMIN = 0.0
_DEFAULT_ZMAX = 2.0
_DEFAULT_POINTS = 300
_DEFAULT_ZEFF = 0.5
_DEFAULT_SIGMA_Z = 0.3
_MIN_POINTS = 2
_MIN_A = 1e-12
_MIN_INTEGRATION_STEPS = 2048
_MAX_INTEGRATION_STEPS = 20000


def _build_z_grid(zmin: float, zmax: float, points: int) -> np.ndarray:
    if points < _MIN_POINTS:
        raise ValueError("points must be at least 2.")
    if zmax <= zmin:
        raise ValueError("zmax must be greater than zmin.")
    return np.linspace(zmin, zmax, num=points, dtype=float)


def _build_integration_grid(a_targets: np.ndarray) -> np.ndarray:
    clipped = np.clip(a_targets, _MIN_A, 1.0)
    if clipped.size == 0:
        raise ValueError("Need at least one scale factor to solve growth.")
    a_min = float(np.min(clipped))
    start = max(min(a_min * 0.5, 1e-4), 1e-9)
    steps = int(np.clip(clipped.size * 8, _MIN_INTEGRATION_STEPS, _MAX_INTEGRATION_STEPS))
    return np.logspace(np.log10(start), 0.0, num=steps, dtype=float)


def _interpolate_log_space(grid: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    log_grid = np.log(grid)
    clipped = np.clip(targets, grid[0], grid[-1])
    return np.interp(np.log(clipped), log_grid, values)


def _solve_growth_with_adapter(model: PredictionModelAdapter, z_values: np.ndarray) -> np.ndarray:
    if z_values.size == 0:
        return np.array([], dtype=float)
    a_targets = 1.0 / np.clip(1.0 + z_values, _MIN_A, np.inf)
    a_grid = _build_integration_grid(a_targets)
    H_grid = model.H(a_grid)
    if H_grid.shape != a_grid.shape:
        raise RuntimeError("H(a) grid size mismatch while solving growth.")
    H0 = float(model.parameters.get("H0", 67.4))
    if H0 <= 0.0:
        raise ValueError("Model reports non-positive H0.")
    E_grid = np.clip(np.asarray(H_grid, dtype=float) / H0, 1e-12, np.inf)
    omega_m0 = float(model.omega_m0())
    D_grid, _ = solve_growth(a_grid, E_grid, omega_m0=omega_m0)
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


def _build_lcdm_reference(model: PredictionModelAdapter) -> PredictionModelAdapter:
    params = model.parameters
    lcdm_kwargs = {
        "H0": float(params.get("H0", 67.4)),
        "Omega_m0": float(params.get("Omega_m0", 0.315)),
        "Omega_b0": float(params.get("Omega_b0", 0.049)),
        "Omega_r0": float(params.get("Omega_r0", 9e-5)),
        "Omega_k0": float(params.get("Omega_k0", 0.0)),
    }
    lcdm_model = create_model("lcdm", **lcdm_kwargs)
    return PredictionModelAdapter(lcdm_model)


def _build_nz(z_values: np.ndarray, z_eff: float, sigma_z: float) -> np.ndarray:
    if sigma_z <= 0.0:
        raise ValueError("sigma_z must be positive.")
    raw = np.exp(-0.5 * ((z_values - float(z_eff)) / float(sigma_z)) ** 2)
    integral = np.trapezoid(raw, z_values)
    if integral <= 0.0:
        raise ValueError("Tracer distribution integral is non-positive.")
    return raw / integral


def _compute_phi(model: PredictionModelAdapter, a_values: np.ndarray, D_values: np.ndarray) -> np.ndarray:
    safe_a = np.clip(a_values, _MIN_A, np.inf)
    E_vals = np.asarray(model.background.E(a_values), dtype=float)
    omega_m0 = float(model.omega_m0())
    omega_m = omega_m0 * np.power(safe_a, -3.0) / np.clip(np.square(E_vals), _MIN_A, np.inf)
    phi = omega_m * np.asarray(D_values, dtype=float) / safe_a
    return np.asarray(phi, dtype=float)


def _missing_api_result() -> PredictionResult:
    return PredictionResult(
        name="isw-cross",
        version="v1",
        metadata={"error": "missing_growth_or_background_api"},
        results={},
        tables=[],
        plots=[],
        status="error",
    )


@register_prediction
class ISWCrossPrediction(PredictionModule):
    name = "isw-cross"
    version = "v1"
    description = "Predicts the ISW × LSS cross-correlation amplitude relative to ΛCDM."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=_DEFAULT_ZMIN, help="Minimum redshift for the kernel (default 0.0)")
        parser.add_argument("--zmax", type=float, default=_DEFAULT_ZMAX, help="Maximum redshift for the kernel (default 2.0)")
        parser.add_argument("--points", type=int, default=_DEFAULT_POINTS, help="Number of redshift samples (default 300)")
        parser.add_argument("--z-eff", type=float, default=_DEFAULT_ZEFF, help="Tracer redshift peak (default 0.5)")
        parser.add_argument("--sigma-z", type=float, default=_DEFAULT_SIGMA_Z, help="Tracer redshift width (default 0.3)")
        parser.add_argument("--compare-lcdm", action="store_true", help="Include the ΛCDM-style reference amplitude")
        parser.add_argument("--output-plot", action="store_true", help="Emit ISW source/kernel plots")
        parser.add_argument("--output-table", action="store_true", help="Export the ISW kernel table")
        super().register(parser)

    def run_prediction(self, model: PredictionModelAdapter, config: dict[str, object]) -> PredictionResult:
        zmin = float(config.get("zmin", _DEFAULT_ZMIN))
        zmax = float(config.get("zmax", _DEFAULT_ZMAX))
        points = max(int(config.get("points", _DEFAULT_POINTS)), _MIN_POINTS)
        z_eff = float(config.get("z_eff", _DEFAULT_ZEFF))
        sigma_z = float(config.get("sigma_z", _DEFAULT_SIGMA_Z))
        compare_lcdm = bool(config.get("compare_lcdm", False))
        output_plot = bool(config.get("output_plot", False))
        output_table = bool(config.get("output_table", False))
        z_values = _build_z_grid(zmin, zmax, points)
        a_values = 1.0 / np.clip(1.0 + z_values, _MIN_A, np.inf)
        try:
            growth_pbuf = _growth_series(model, z_values)
            phi_pbuf = _compute_phi(model, a_values, growth_pbuf)
        except AttributeError:
            return _missing_api_result()
        dphi_pbuf = np.gradient(phi_pbuf, z_values)
        n_z = _build_nz(z_values, z_eff, sigma_z)
        kernel_pbuf = n_z * dphi_pbuf
        A_raw_pbuf = np.trapezoid(kernel_pbuf, z_values)
        max_abs_pbuf = float(np.max(np.abs(dphi_pbuf)))
        A_ISW_pbuf = float(A_raw_pbuf / max_abs_pbuf) if max_abs_pbuf > 0.0 else 0.0

        phi_lcdm: np.ndarray | None = None
        dphi_lcdm: np.ndarray | None = None
        kernel_lcdm: np.ndarray | None = None
        A_ISW_lcdm: float | None = None
        ratio: float | None = None

        if compare_lcdm:
            try:
                lcdm_adapter = _build_lcdm_reference(model)
                growth_lcdm = _growth_series(lcdm_adapter, z_values)
                phi_lcdm = _compute_phi(lcdm_adapter, a_values, growth_lcdm)
            except AttributeError:
                return _missing_api_result()
            dphi_lcdm = np.gradient(phi_lcdm, z_values)
            kernel_lcdm = n_z * dphi_lcdm
            A_raw_lcdm = np.trapezoid(kernel_lcdm, z_values)
            max_abs_lcdm = float(np.max(np.abs(dphi_lcdm)))
            A_ISW_lcdm = float(A_raw_lcdm / max_abs_lcdm) if max_abs_lcdm > 0.0 else 0.0
            if A_ISW_lcdm != 0.0:
                ratio = float(A_ISW_pbuf / A_ISW_lcdm)

        tables: list[PredictionTable] = []
        if output_table:
            rows: list[list[float | None]] = []
            for idx, z_val in enumerate(z_values):
                row = [
                    float(z_val),
                    float(a_values[idx]),
                    float(phi_pbuf[idx]),
                    float(dphi_pbuf[idx]),
                    float(n_z[idx]),
                    float(kernel_pbuf[idx]),
                    float(phi_lcdm[idx]) if phi_lcdm is not None else None,
                    float(dphi_lcdm[idx]) if dphi_lcdm is not None else None,
                    float(kernel_lcdm[idx]) if kernel_lcdm is not None else None,
                ]
                rows.append(row)
            tables.append(
                PredictionTable(
                    name="isw_kernel_vs_z",
                    columns=["z", "a", "Phi_PBUF", "dPhi_dz_PBUF", "n_z", "kernel_PBUF", "Phi_LCDM", "dPhi_dz_LCDM", "kernel_LCDM"],
                    rows=rows,
                    metadata={"points": len(z_values)},
                )
            )

        plots: list[PredictionPlot] = []
        if output_plot:
            source_data = {"z": z_values.tolist(), "dPhi_dz_PBUF": dphi_pbuf.tolist()}
            if dphi_lcdm is not None:
                source_data["dPhi_dz_LCDM"] = dphi_lcdm.tolist()
            plots.append(
                PredictionPlot(
                    name="isw_source_vs_z",
                    data=source_data,
                    metadata={"xlabel": "redshift z", "ylabel": "ISW source dΦ/dz"},
                )
            )
            kernel_data = {"z": z_values.tolist(), "kernel_PBUF": kernel_pbuf.tolist()}
            if kernel_lcdm is not None:
                kernel_data["kernel_LCDM"] = kernel_lcdm.tolist()
            plots.append(
                PredictionPlot(
                    name="isw_kernel_vs_z_plot",
                    data=kernel_data,
                    metadata={"xlabel": "redshift z", "ylabel": "n(z) · dΦ/dz"},
                )
            )

        metadata = {
            "model": model.raw_model.__class__.__name__,
            "compare_lcdm": compare_lcdm,
            "points": len(z_values),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        results = {
            "zmin": float(zmin),
            "zmax": float(zmax),
            "z_eff": float(z_eff),
            "sigma_z": float(sigma_z),
            "A_ISW_PBUF": A_ISW_pbuf,
            "A_ISW_LCDM": A_ISW_lcdm,
            "ratio_PBUF_over_LCDM": ratio,
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )
