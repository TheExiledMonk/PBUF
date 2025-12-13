"""CMB lensing convergence prediction following the unified runner contract."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult

logger = logging.getLogger(__name__)

_DEFAULT_ELL_MIN = 8.0
_DEFAULT_ELL_MAX = 2000.0
_DEFAULT_N_ELL = 400
_MIN_VALID_POINTS = 10
_NOTE_TEXT = "CMB lensing convergence spectrum and effective amplitude."
_DESCRIPTION_TEXT = (
    "CMB lensing prediction module.\n"
    "This module computes the convergence power spectrum C_ell^{kappa kappa} for the current cosmological model "
    "using the model's own lensing backend. It applies masking to remove non-physical or numerically unstable points "
    "and summarizes the total lensing strength via an effective amplitude A_L_eff. No external Planck defaults or "
    "hard-coded parameters are used."
)


def _build_ell_grid(ell_min: float, ell_max: float, n_ell: int) -> np.ndarray:
    if n_ell < 2:
        raise ValueError("n_ell must be at least 2.")
    if ell_max <= ell_min:
        raise ValueError("ell_max must exceed ell_min.")
    if ell_min < 2.0:
        ell_min = 2.0
    return np.linspace(ell_min, ell_max, n_ell, dtype=float)


def _load_reference_spectrum(config: dict[str, Any], ell: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    payload = config.get("reference_spectrum")
    if not isinstance(payload, dict):
        return None
    ell_ref = payload.get("ell")
    cl_ref = payload.get("cl_kappa")
    if ell_ref is None or cl_ref is None:
        return None
    try:
        ell_arr = np.asarray(ell_ref, dtype=float)
        cl_arr = np.asarray(cl_ref, dtype=float)
    except Exception:
        logger.warning("Reference spectrum arrays are invalid; ignoring relative amplitude.")
        return None
    if ell_arr.shape != cl_arr.shape or ell_arr.shape != ell.shape:
        logger.warning("Reference spectrum shape mismatch; ignoring relative amplitude.")
        return None
    if not np.allclose(ell_arr, ell, rtol=1e-8, atol=1e-12):
        logger.warning("Reference spectrum ell grid differs from prediction; ignoring relative amplitude.")
        return None
    return ell_arr, cl_arr


def _build_meta(
    ell_min: float,
    ell_max: float,
    n_ell: int,
    model_name: str,
    backend_source: str | None,
    reference_used: bool,
) -> dict[str, Any]:
    notes = _NOTE_TEXT
    metadata = {
        "ell_min": float(ell_min),
        "ell_max": float(ell_max),
        "n_ell": int(n_ell),
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0",
        "notes": notes,
        "description": _DESCRIPTION_TEXT,
        "lensing_backend": backend_source or "unknown",
        "reference_used": reference_used,
    }
    return metadata


def _error_result(message: str) -> PredictionResult:
    return PredictionResult(
        name="cmb-lensing",
        version="1.0",
        metadata={"error": message},
        results={},
        tables=[],
        plots=[],
        status="error",
    )


@register_prediction
class CMBLensingPrediction(PredictionModule):
    name = "cmb-lensing"
    version = "1.0"
    description = "CMB lensing convergence spectrum and effective amplitude."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument(
            "--ell-min",
            type=float,
            default=_DEFAULT_ELL_MIN,
            help="Minimum multipole for the lensing prediction grid (default 8).",
        )
        parser.add_argument(
            "--ell-max",
            type=float,
            default=_DEFAULT_ELL_MAX,
            help="Maximum multipole for the lensing prediction grid (default 2000).",
        )
        parser.add_argument(
            "--n-ell",
            type=int,
            default=_DEFAULT_N_ELL,
            help="Number of linear multipoles to sample (default 400).",
        )
        super().register(parser)

    def run_prediction(
        self, model: PredictionModelAdapter, config: dict[str, Any]
    ) -> PredictionResult:
        try:
            ell_min = float(config.get("ell_min", _DEFAULT_ELL_MIN))
            ell_max = float(config.get("ell_max", _DEFAULT_ELL_MAX))
            n_ell = max(2, int(config.get("n_ell", _DEFAULT_N_ELL)))
        except (TypeError, ValueError) as exc:
            logger.exception("Invalid elliptic grid configuration: %s", exc)
            return _error_result("invalid_grid_configuration")

        try:
            ell_grid = _build_ell_grid(ell_min, ell_max, n_ell)
        except ValueError as exc:
            logger.exception("Failed to build ell grid: %s", exc)
            return _error_result("invalid_ell_grid")

        lensing_backend = model.lensing
        try:
            cl_kappa_flat = np.asarray(lensing_backend.compute_cmb_kappa(ell_grid), dtype=float)
        except Exception:
            logger.exception("Failed to compute CMB lensing spectrum.")
            return _error_result("cmb_lensing_compute_failure")

        if cl_kappa_flat.shape != ell_grid.shape:
            logger.error("Lensing backend returned mismatched array shapes.")
            return _error_result("unexpected_lensing_output_shape")

        mask_valid = np.isfinite(cl_kappa_flat) & (cl_kappa_flat >= 0.0)
        valid_count = int(mask_valid.sum())
        warnings: list[str] | None = None
        if valid_count < _MIN_VALID_POINTS:
            message = (
                "CMB lensing mask has fewer than 10 valid multipoles; treat the spectrum with care."
                if valid_count > 0
                else "CMB lensing mask contains no valid multipoles."
            )
            logger.warning(message)
            warnings = [message]

        summary = _compute_summary(ell_grid, cl_kappa_flat, mask_valid, config)

        model_cls = getattr(model.raw_model, "__class__", None)
        model_name = getattr(model_cls, "__name__", "model")
        reference_used = summary.get("A_L_rel") is not None
        meta_payload = _build_meta(
            ell_min=ell_grid[0],
            ell_max=ell_grid[-1],
            n_ell=len(ell_grid),
            model_name=model_name,
            backend_source=lensing_backend.backend_source,
            reference_used=reference_used,
        )

        prediction = {
            "name": "cmb_lensing",
            "ell": ell_grid.tolist(),
            "cl_kappa": cl_kappa_flat.tolist(),
            "mask_valid": mask_valid.tolist(),
            "summary": summary,
            "meta": meta_payload,
        }

        plot_data = {
            "ell": ell_grid[mask_valid].tolist(),
            "cl_kappa": cl_kappa_flat[mask_valid].tolist(),
        }
        plots = [
            PredictionPlot(
                name="cmb_lensing_plot",
                data=plot_data,
                description="CMB lensing convergence (cmb_lensing_plot)",
                metadata={"xlabel": "ell", "ylabel": "C_ell^{kappa kappa}"},
            )
        ]

        result_meta: dict[str, Any] = {
            "model": model_name,
            "valid_points": valid_count,
            "lensing_backend": lensing_backend.backend_source,
        }
        if warnings:
            result_meta["warnings"] = warnings
        if reference_used:
            result_meta["reference_used"] = True

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=result_meta,
            results=prediction,
            plots=plots,
        )


def _compute_summary(
    ell: np.ndarray, cl_kappa: np.ndarray, mask_valid: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    valid_count = int(mask_valid.sum())
    A_L_eff = float("nan")
    if valid_count > 0:
        ell_valid = ell[mask_valid]
        cl_valid = cl_kappa[mask_valid]
        if ell_valid.size >= 2:
            integrated_power = float(np.trapz(cl_valid, ell_valid))
            ell_span = float(ell_valid.max() - ell_valid.min())
            if ell_span > 0.0:
                A_L_eff = integrated_power / ell_span
        else:
            A_L_eff = float("nan")

    A_L_rel = _compute_relative_amplitude(ell, cl_kappa, mask_valid, config)

    summary: dict[str, Any] = {"A_L_eff": float(A_L_eff)}
    if A_L_rel is not None:
        summary["A_L_rel"] = float(A_L_rel)

    return summary


def _compute_relative_amplitude(
    ell: np.ndarray, cl_kappa: np.ndarray, mask_valid: np.ndarray, config: dict[str, Any]
) -> float | None:
    reference = _load_reference_spectrum(config, ell)
    if reference is None:
        return None
    _, ref_cl = reference
    mask_both = mask_valid & np.isfinite(ref_cl) & (ref_cl > 0.0)
    if mask_both.sum() < 2:
        return None
    numerator = float(np.trapz(cl_kappa[mask_both] * ref_cl[mask_both], ell[mask_both]))
    denominator = float(np.trapz(ref_cl[mask_both] ** 2, ell[mask_both]))
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator
