"""Prediction module for the baryon sound horizon."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np

from cosmos2.kernels import lcdm_math
from cosmos2.models.model_factory import create_model as create_cosmos2_model
from cosmos2.parameters.central_authority import MPC_TO_KM

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult, PredictionTable


@dataclass(frozen=True)
class SoundHorizonEvaluation:
    a_grid: np.ndarray
    H_grid: np.ndarray
    c_s: np.ndarray
    integrand: np.ndarray
    cumulative: np.ndarray
    r_d_Mpc: float
    a_drag: float
    z_drag: float


@register_prediction
class SoundHorizonPrediction(PredictionModule):
    name = "sound-horizon"
    version = "v1"
    description = "Computes the baryon sound horizon r_d for the selected model."

    def register(self, parser: "argparse.ArgumentParser") -> None:  # type: ignore[override]
        parser.add_argument(
            "--resolution",
            type=int,
            default=2000,
            help="Number of integration points for the sound-horizon grid.",
        )
        parser.add_argument(
            "--compare-lcdm",
            action="store_true",
            help="Compute Δr_d/r_d against an LCDM reference worth the same parameters.",
        )
        parser.add_argument(
            "--output-plot",
            action="store_true",
            help="Emit the canonical integrand plot (and optional cumulative curve).",
        )
        parser.add_argument(
            "--output-table",
            action="store_true",
            help="Export the integrand table (a, H, c_s, integrand).",
        )
        super().register(parser)

    def run_prediction(
        self, model: "PredictionModelAdapter", config: dict[str, object]
    ) -> PredictionResult:
        resolution = max(int(config.get("resolution", 2000)), 10)
        compare_lcdm = bool(config.get("compare_lcdm"))
        output_plot = bool(config.get("output_plot"))
        output_table = bool(config.get("output_table"))

        evaluation = _evaluate_sound_horizon(model, resolution)
        r_d_Mpc = evaluation.r_d_Mpc
        r_d_km = r_d_Mpc * MPC_TO_KM

        tables = []
        if output_table:
            rows = [
                [float(a), float(H), float(c_s), float(integrand)]
                for a, H, c_s, integrand in zip(
                    evaluation.a_grid, evaluation.H_grid, evaluation.c_s, evaluation.integrand
                )
            ]
            tables.append(
                PredictionTable(
                    name="sound_horizon_integrand",
                    columns=["a", "H", "c_s", "integrand"],
                    rows=rows,
                    metadata={"points": len(evaluation.a_grid)},
                )
            )

        plots = []
        if output_plot:
            plots.append(
                PredictionPlot(
                    name="sound_horizon_curve",
                    data={
                        "a": evaluation.a_grid.tolist(),
                        "integrand": evaluation.integrand.tolist(),
                    },
                    description="Canonical integrand c_s/(a^2 H) vs. scale factor",
                    metadata={"xlabel": "scale factor a", "ylabel": "c_s/(a^2 H)"},
                )
            )
            plots.append(
                PredictionPlot(
                    name="sound_horizon_cumulative",
                    data={"a": evaluation.a_grid.tolist(), "r_d": evaluation.cumulative.tolist()},
                    description="Cumulative sound horizon build-up vs. scale factor",
                    metadata={"xlabel": "scale factor a", "ylabel": "cumulative r_d (Mpc)"},
                )
            )

        delta_vs_lcdm = None
        lcdm_r_d = None
        if compare_lcdm:
            try:
                lcdm_params = _extract_lcdm_params(model.parameters)
                lcdm_model = create_cosmos2_model("lcdm", **lcdm_params)
                lcdm_eval = _evaluate_sound_horizon(PredictionModelAdapter(lcdm_model), resolution)
                lcdm_r_d = lcdm_eval.r_d_Mpc
                if lcdm_r_d > 0.0:
                    delta_vs_lcdm = (r_d_Mpc - lcdm_r_d) / lcdm_r_d
            except Exception:
                delta_vs_lcdm = None

        metadata = {
            "model": model.raw_model.__class__.__name__,
            "resolution": resolution,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "a_drag": evaluation.a_drag,
            "z_drag": evaluation.z_drag,
            "compare_lcdm": compare_lcdm,
        }

        results = {
            "r_d_Mpc": r_d_Mpc,
            "r_d_km": r_d_km,
            "a_drag": evaluation.a_drag,
            "z_drag": evaluation.z_drag,
        }
        if delta_vs_lcdm is not None:
            results["delta_vs_lcdm"] = delta_vs_lcdm
        if lcdm_r_d is not None:
            results["r_d_lcdm_Mpc"] = lcdm_r_d

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            tables=tables,
            plots=plots,
        )


def _extract_lcdm_params(parameters: dict[str, float | str]) -> dict[str, float]:
    """Helper to build LCDM parameter dict that mirrors the input model."""
    params = {
        "H0": float(parameters.get("H0", 67.4)),
        "Omega_m0": float(parameters.get("Omega_m0", 0.315)),
        "Omega_b0": float(parameters.get("Omega_b0", 0.049)),
        "Omega_r0": float(parameters.get("Omega_r0", 9.0e-5)),
        "Omega_k0": float(parameters.get("Omega_k0", 0.0)),
    }
    return params


def _evaluate_sound_horizon(model: "PredictionModelAdapter", resolution: int) -> SoundHorizonEvaluation:
    """Compute the integrand grid and cumulative result for a given model."""

    parameters = model.parameters
    H0 = float(parameters.get("H0", 67.4))
    Omega_b0 = float(parameters.get("Omega_b0", 0.0))
    if Omega_b0 <= 0.0:
        raise ValueError("Omega_b0 must be positive to build the sound horizon integrand.")
    omega_gamma0 = float(
        parameters.get("Omega_gamma0", lcdm_math.omega_gamma0_from_Tcmb(H0))
    )
    if omega_gamma0 <= 0.0:
        raise ValueError("Omega_gamma0 must be positive to build the photon density.")

    a_drag, z_drag = _resolve_drag_epoch(model, H0)
    a_drag = max(min(a_drag, 1.0), 1e-9)
    a_min = max(1e-12, min(a_drag * 1e-6, 1e-8))
    a_grid = np.logspace(np.log10(a_min), np.log10(a_drag), resolution)

    H_grid = model.H(a_grid)
    temperature = model.temperature(a_grid)
    tcmb = float(model.constants.get("Tcmb", 2.7255))
    theta = np.clip((temperature / tcmb) ** 4, 0.0, None)
    rho_b = Omega_b0 / np.clip(a_grid**3, 1e-30, None)
    rho_gamma = omega_gamma0 * theta
    R = np.clip((3.0 * rho_b) / np.clip(4.0 * rho_gamma, 1e-30, None), 0.0, None)
    c_s = lcdm_math.C_LIGHT / np.sqrt(np.clip(3.0 * (1.0 + R), 1e-30, None))

    denom = np.clip(a_grid * a_grid * H_grid, 1e-30, None)
    integrand = c_s / denom
    integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)
    r_d_Mpc = float(np.trapz(integrand, a_grid))
    cumulative = _cumulative_trapz(integrand, a_grid)

    return SoundHorizonEvaluation(
        a_grid=a_grid,
        H_grid=H_grid,
        c_s=c_s,
        integrand=integrand,
        cumulative=cumulative,
        r_d_Mpc=r_d_Mpc,
        a_drag=a_drag,
        z_drag=z_drag,
    )


def _resolve_drag_epoch(model: "PredictionModelAdapter", H0: float) -> tuple[float, float]:
    """Locate the scale factor/redshift where baryons decouple."""

    raw = model.raw_model
    z_drag = None
    a_drag = None

    drag_scale = getattr(raw, "drag_scale_factor", None)
    if callable(drag_scale):
        try:
            a_drag = float(drag_scale())
            if a_drag > 0.0:
                z_drag = max(0.0, 1.0 / a_drag - 1.0)
        except Exception:
            a_drag = None

    if z_drag is None:
        drag_redshift = getattr(raw, "drag_redshift", None)
        if callable(drag_redshift):
            try:
                z_drag = float(drag_redshift())
                if z_drag >= 0.0:
                    a_drag = 1.0 / (1.0 + z_drag)
            except Exception:
                z_drag = None

    if z_drag is None or a_drag is None or a_drag <= 0.0:
        Omega_m0 = float(model.parameters.get("Omega_m0", 0.315))
        Omega_b0 = float(model.parameters.get("Omega_b0", 0.049))
        z_drag = float(lcdm_math._z_drag_eh(Omega_m0, Omega_b0, H0))
        a_drag = 1.0 / (1.0 + z_drag) if z_drag > 0.0 else 1.0

    return a_drag, z_drag


def _cumulative_trapz(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with a zero baseline at the first point."""

    dx = np.diff(x)
    mid = 0.5 * (values[:-1] + values[1:])
    cum = np.concatenate([[0.0], np.cumsum(dx * mid)])
    return cum
