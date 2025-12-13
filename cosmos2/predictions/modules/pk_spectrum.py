"""Matter power spectrum prediction module computing P(k,z) on a configurable k grid."""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult

_DEFAULT_K_MIN = 1e-4
_DEFAULT_K_MAX = 1.0
_DEFAULT_N_K = 200
_DEFAULT_Z_SAMPLES = (0.0, 0.5, 1.0)
_MIN_VALID_POINTS = 10
_K_GRID_SPLIT = re.compile(r"[,\s;]+")
_DEFAULT_K_UNITS = "model-specific (e.g. h/Mpc)"
_DEFAULT_P_UNITS = "model-specific (e.g. (Mpc/h)^3)"
_META_NOTES = (
    "Matter power spectrum P(k,z) evaluated on a log-spaced k-grid for a small set of redshifts."
)
_META_DESCRIPTION = (
    "Matter power spectrum P(k,z) on a log-spaced k-grid for a small set of redshifts. "
    "The module uses the model's own power_spectrum backend and provides summary amplitudes "
    "at representative scales for quick comparison between cosmological models."
)

logger = logging.getLogger(__name__)


def _parse_z_samples(value: object | None) -> list[float]:
    if value is None:
        return list(_DEFAULT_Z_SAMPLES)
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("z_samples cannot be empty.")
        tokens = [token for token in _K_GRID_SPLIT.split(trimmed) if token]
        if not tokens:
            raise ValueError("z_samples cannot be empty.")
        return [float(token) for token in tokens]
    if isinstance(value, Sequence):
        parsed: list[float] = []
        for item in value:
            if item is None:
                continue
            parsed.append(float(item))
        if not parsed:
            raise ValueError("z_samples must include at least one value.")
        return parsed
    if isinstance(value, Iterable):
        parsed = [float(item) for item in list(value)]
        if not parsed:
            raise ValueError("z_samples must include at least one value.")
        return parsed
    raise ValueError("Unable to parse z_samples specification.")


def _format_z_label(z: float) -> str:
    formatted = f"{z:.3f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted.replace(".", "p")


def _P_at_k(target_k: float, k_grid: np.ndarray, P_k: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    idx_valid = np.where(mask)[0]
    if idx_valid.size == 0:
        return None
    distances = np.abs(k_grid[idx_valid] - target_k)
    nearest = idx_valid[np.argmin(distances)]
    value = P_k[nearest]
    return float(value) if np.isfinite(value) else None


def _find_z_index(z_samples: list[float], target_z: float) -> int | None:
    for idx, z in enumerate(z_samples):
        if np.isclose(z, target_z, rtol=1e-3, atol=1e-3):
            return idx
    return None


@register_prediction
class PKSpectrumPrediction(PredictionModule):
    """Matter power spectrum predictions on a log-spaced k grid."""

    name = "pk-spectrum"
    version = "1.0"
    description = "Matter power spectrum P(k,z) using the model's power spectrum backend."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument(
            "--k-min",
            type=float,
            default=_DEFAULT_K_MIN,
            help=f"Minimum k for the log grid (default {_DEFAULT_K_MIN}).",
        )
        parser.add_argument(
            "--k-max",
            type=float,
            default=_DEFAULT_K_MAX,
            help=f"Maximum k for the log grid (default {_DEFAULT_K_MAX}).",
        )
        parser.add_argument(
            "--n-k",
            type=int,
            default=_DEFAULT_N_K,
            help=f"Number of k samples (default {_DEFAULT_N_K}).",
        )
        parser.add_argument(
            "--z-samples",
            type=str,
            help="Comma/space/semicolon separated redshifts (default 0.0,0.5,1.0).",
        )
        super().register(parser)

    def run_prediction(
        self, model: PredictionModelAdapter, config: dict[str, object]
    ) -> PredictionResult:
        k_min = float(config.get("k_min", _DEFAULT_K_MIN))
        k_max = float(config.get("k_max", _DEFAULT_K_MAX))
        n_k = int(config.get("n_k", _DEFAULT_N_K))
        if k_min <= 0.0:
            raise ValueError("k_min must be positive.")
        if k_max <= k_min:
            raise ValueError("k_max must be greater than k_min.")
        if n_k < 2:
            raise ValueError("n_k must be at least 2.")

        raw_z = config.get("z_samples")
        z_samples = _parse_z_samples(raw_z)
        if not z_samples:
            raise ValueError("At least one redshift is required.")

        k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k, dtype=float)

        try:
            pk_config = model.matter.pk_config()
        except AttributeError:
            pk_config = {}

        k_units = str(
            config.get("k_units")
            or pk_config.get("k_units")
            or pk_config.get("k_unit")
            or _DEFAULT_K_UNITS
        )
        P_units = str(
            config.get("P_units")
            or pk_config.get("P_units")
            or pk_config.get("P_unit")
            or _DEFAULT_P_UNITS
        )

        mask_valid = np.ones_like(k_grid, dtype=bool)
        P_dict: dict[float, np.ndarray] = {}
        for z in z_samples:
            try:
                P_vals = np.asarray(model.matter.power_spectrum(k_grid, z), dtype=float)
            except AttributeError:
                return PredictionResult(
                    name=self.name,
                    version=self.version,
                    metadata={"error": "missing_matter_power_spectrum_api"},
                    results={},
                    tables=[],
                    plots=[],
                    status="error",
                )
            if P_vals.shape != k_grid.shape:
                raise RuntimeError("power_spectrum returned an array with unexpected shape.")
            is_finite = np.isfinite(P_vals)
            is_non_negative = P_vals >= 0.0
            valid_mask = is_finite & is_non_negative
            safe_vals = P_vals.copy()
            safe_vals[~valid_mask] = np.nan
            P_dict[z] = safe_vals
            mask_valid &= valid_mask

        valid_count = int(np.count_nonzero(mask_valid))
        if valid_count < _MIN_VALID_POINTS:
            logger.warning(
                "pk-spectrum mask valid points = %d (< %d); results may be incomplete.",
                valid_count,
                _MIN_VALID_POINTS,
            )

        summary = {
            "P0p1_z0": None,
            "P0p2_z0": None,
            "P0p1_z1": None,
            "sigma8_like": None,
        }

        idx_z0 = _find_z_index(z_samples, 0.0)
        idx_z1 = _find_z_index(z_samples, 1.0)

        if idx_z0 is not None:
            P_z0 = P_dict[z_samples[idx_z0]]
            summary["P0p1_z0"] = _P_at_k(0.1, k_grid, P_z0, mask_valid)
            summary["P0p2_z0"] = _P_at_k(0.2, k_grid, P_z0, mask_valid)

        if idx_z1 is not None:
            P_z1 = P_dict[z_samples[idx_z1]]
            summary["P0p1_z1"] = _P_at_k(0.1, k_grid, P_z1, mask_valid)

        sigma8_like = None
        if idx_z0 is not None and valid_count > 1:
            P_valid = P_dict[z_samples[idx_z0]][mask_valid]
            k_valid = k_grid[mask_valid]
            if k_valid.size > 1:
                integrand = k_valid**3 * P_valid
                ln_k = np.log(k_valid)
                integral = np.trapezoid(integrand, ln_k)
                if np.isfinite(integral):
                    sigma8_like = float(integral)
        summary["sigma8_like"] = sigma8_like

        plot_data: dict[str, list[float]] = {}
        if valid_count > 0:
            k_valid = k_grid[mask_valid]
            plot_data["k"] = k_valid.tolist()
            for z in z_samples:
                P_vals = P_dict[z][mask_valid]
                if P_vals.size != k_valid.size:
                    continue
                plot_data[f"P_z{_format_z_label(z)}"] = P_vals.tolist()

        plots: list[PredictionPlot] = []
        if plot_data:
            plots.append(
                PredictionPlot(
                    name="pk_spectrum_plot",
                    description="Matter power spectrum P(k,z) curves.",
                    data=plot_data,
                    metadata={
                        "xlabel": f"k [{k_units}]",
                        "ylabel": f"P(k) [{P_units}]",
                        "log_x": True,
                        "log_y": True,
                    },
                )
            )

        meta_payload = {
            "k_min": float(k_min),
            "k_max": float(k_max),
            "n_k": int(n_k),
            "k_units": k_units,
            "P_units": P_units,
            "model_name": getattr(model.raw_model.__class__, "__name__", "model"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": self.version,
            "notes": _META_NOTES,
            "description": _META_DESCRIPTION,
        }

        metadata: dict[str, object] = {
            "model": meta_payload["model_name"],
            "valid_points": valid_count,
            "description": _META_DESCRIPTION,
            "summary": "Matter power spectrum prediction sampled at selected redshifts.",
        }
        if valid_count < _MIN_VALID_POINTS:
            metadata.setdefault("warnings", []).append(
                f"Only {valid_count} valid samples (need ≥ {_MIN_VALID_POINTS}); P(k) may be noisy."
            )

        prediction_payload = {
            "name": self.name,
            "k": k_grid.tolist(),
            "z_samples": [float(z) for z in z_samples],
            "P_k_arrays": [P_dict[z].tolist() for z in z_samples],
            "mask_valid": mask_valid.tolist(),
            "summary": summary,
            "meta": meta_payload,
        }

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=prediction_payload,
            plots=plots,
        )
