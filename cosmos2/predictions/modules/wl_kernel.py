"""Weak-lensing efficiency kernel prediction module."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Mapping
import logging

import numpy as np

from ..model_api import PredictionModelAdapter
from ..registry import PredictionModule, register_prediction
from ..structures import PredictionPlot, PredictionResult
from ..wl_utils import wl_source_distribution

logger = logging.getLogger(__name__)


@register_prediction
class WeakLensingKernelPrediction(PredictionModule):
    """Compute a lightweight weak-lensing efficiency kernel."""

    name = "wl-kernel"
    version = "1.0"
    description = "Weak-lensing kernel sensitivity W(z) for a given source redshift distribution."

    def register(self, parser: argparse.ArgumentParser) -> None:  # type: ignore[override]
        parser.add_argument("--zmin", type=float, default=0.0, help="Minimum redshift (default 0.0).")
        parser.add_argument("--zmax", type=float, default=3.0, help="Maximum redshift (default 3.0).")
        parser.add_argument("--points", type=int, default=300, help="Samples along the redshift axis (default 300).")
        parser.add_argument(
            "--no-normalize",
            dest="normalize",
            action="store_false",
            help="Disable scaling W(z) so that max(W)=1; raw kernel is kept instead.",
        )
        parser.add_argument(
            "--source-type",
            choices=["lsst_like", "euclid_like", "simple"],
            help="Override the builtin source distribution type.",
        )
        parser.add_argument("--source-z0", type=float, help="Override the z0 parameter for n(z).")
        parser.add_argument("--source-alpha", type=float, help="Override the α parameter for n(z).")
        parser.add_argument("--source-beta", type=float, help="Override the β parameter for n(z).")
        super().register(parser)

    def run_prediction(
        self, model: PredictionModelAdapter, config: dict[str, Any]
    ) -> PredictionResult:
        zmin = float(config.get("zmin", 0.0))
        if zmin < 0.0:
            raise ValueError("zmin must be non-negative.")
        zmax = float(config.get("zmax", 3.0))
        if zmax <= zmin:
            raise ValueError("zmax must exceed zmin.")
        points = max(int(config.get("points", 300)), 2)
        normalize_kernel = bool(config.get("normalize", True))
        source_config = self._build_source_config(config)

        z_grid = np.linspace(zmin, zmax, points, dtype=float)
        H_vals = np.asarray(model.background.H(z_grid), dtype=float)
        chi_vals = np.asarray(model.background.comoving_distance(z_grid), dtype=float)
        try:
            c_value = float(model.background.c_value())
        except Exception:
            c_value = float("nan")

        valid_c = np.isfinite(c_value) and (c_value > 0.0)
        mask_base = (
            np.isfinite(H_vals)
            & (H_vals > 0.0)
            & np.isfinite(chi_vals)
            & (chi_vals >= 0.0)
        )
        if not valid_c:
            logger.warning("Invalid speed of light from background; W(z) outputs will be NaN.")
            mask_base = np.zeros_like(mask_base, dtype=bool)

        n_z = wl_source_distribution(z_grid, source_config)
        if n_z.shape != z_grid.shape:
            raise RuntimeError("Source distribution returned incompatible shape.")
        mask_n = np.isfinite(n_z) & (n_z >= 0.0)
        mask_base &= mask_n

        W_raw = np.full_like(z_grid, np.nan)
        valid_indices = np.nonzero(mask_base)[0]
        if valid_indices.size > 0:
            z_valid = z_grid[valid_indices]
            H_valid = H_vals[valid_indices]
            chi_valid = chi_vals[valid_indices]
            n_valid = n_z[valid_indices]
            for idx, (z_i, chi_i, H_i) in zip(valid_indices, zip(z_valid, chi_valid, H_valid)):
                mask_sources = z_valid >= z_i
                if not np.any(mask_sources):
                    continue
                z_s = z_valid[mask_sources]
                chi_s = chi_valid[mask_sources]
                n_s = n_valid[mask_sources]
                mask_chis = chi_s > 0.0
                if not np.any(mask_chis):
                    continue
                z_s = z_s[mask_chis]
                chi_s = chi_s[mask_chis]
                n_s = n_s[mask_chis]
                integrand = n_s * (chi_s - chi_i) / chi_s
                integral = np.trapz(integrand, z_s)
                W_raw[idx] = (H_i / c_value) * chi_i * integral

        mask_W = mask_base & np.isfinite(W_raw)
        W_norm = np.full_like(W_raw, np.nan)
        if np.any(mask_W):
            W_max = np.nanmax(W_raw[mask_W])
            if np.isfinite(W_max) and W_max > 0.0:
                if normalize_kernel:
                    W_norm[mask_W] = W_raw[mask_W] / W_max
                else:
                    W_norm[mask_W] = W_raw[mask_W]

        mask_valid = mask_W & np.isfinite(W_norm)
        valid_count = int(np.count_nonzero(mask_valid))
        if valid_count < 10:
            logger.warning("Weak-lensing kernel returned only %d valid points; check inputs.", valid_count)

        z_peak = None
        W_peak_value = None
        z_median = None
        if valid_count:
            valid_indices = np.nonzero(mask_valid)[0]
            W_norm_valid = W_norm[mask_valid]
            idx_peak = int(np.argmax(W_norm_valid))
            full_peak_idx = int(valid_indices[idx_peak])
            z_peak = float(z_grid[full_peak_idx])
            W_peak_value = float(W_raw[full_peak_idx]) if np.isfinite(W_raw[full_peak_idx]) else None

            W_weights = W_norm_valid.copy()
            z_valid = z_grid[mask_valid]
            total_W = float(np.sum(W_weights))
            if np.isfinite(total_W) and total_W > 0.0:
                W_weights /= total_W
                sort_idx = np.argsort(z_valid)
                z_sorted = z_valid[sort_idx]
                w_sorted = W_weights[sort_idx]
                cdf = np.cumsum(w_sorted)
                median_idx = int(np.searchsorted(cdf, 0.5))
                if median_idx >= len(z_sorted):
                    median_idx = len(z_sorted) - 1
                z_median = float(z_sorted[median_idx])

        summary = {
            "z_peak": z_peak,
            "W_peak_value": W_peak_value,
            "z_median": z_median,
        }

        metadata = {
            "model": type(model.raw_model).__name__,
            "model_name": type(model.raw_model).__name__,
            "description": self._meta_description(),
            "summary": summary,
            "valid_points": valid_count,
            "mask_valid_fraction": valid_count / float(z_grid.size) if z_grid.size else 0.0,
            "normalize_kernel": normalize_kernel,
            "source_distribution": source_config or "default",
            "z_min": zmin,
            "z_max": zmax,
            "n_points": points,
        }

        meta = {
            "model_name": type(model.raw_model).__name__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": self.version,
            "z_min": zmin,
            "z_max": zmax,
            "n_points": points,
            "n_z_model": self._n_z_model_label(source_config),
            "distance_unit": "same as χ(z)",
            "notes": "Weak-lensing kernel W(z) computed from H(z), comoving distance, and a source n(z).",
            "description": self._meta_description(),
            "normalize_kernel": normalize_kernel,
        }

        results = {
            "name": self.name,
            "z": z_grid.tolist(),
            "chi": chi_vals.tolist(),
            "n_z": n_z.tolist(),
            "W_raw": W_raw.tolist(),
            "W_norm": W_norm.tolist(),
            "mask_valid": mask_valid.tolist(),
            "summary": summary,
            "meta": meta,
        }

        plots: list[PredictionPlot] = []
        if valid_count:
            z_plot = z_grid[mask_valid]
            plots.append(
                PredictionPlot(
                    name="wl_kernel_vs_z",
                    description="Normalized weak-lensing kernel W(z) versus the source distribution.",
                    data={
                        "z": z_plot.tolist(),
                        "W_norm": W_norm[mask_valid].tolist(),
                        "n_z": n_z[mask_valid].tolist(),
                    },
                    metadata={
                        "xlabel": "redshift z",
                        "ylabel": "W_norm(z)",
                        "notes": "n(z) shown to illustrate source sensitivity.",
                    },
                )
            )

        return PredictionResult(
            name=self.name,
            version=self.version,
            metadata=metadata,
            results=results,
            plots=plots,
        )

    @staticmethod
    def _build_source_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
        explicit = config.get("source_distribution")
        if isinstance(explicit, Mapping):
            return dict(explicit)
        overrides: dict[str, Any] = {}
        source_type = config.get("source_type")
        if source_type:
            overrides["type"] = source_type
        params: dict[str, Any] = {}
        for key in ("z0", "alpha", "beta"):
            value = config.get(f"source_{key}")
            if value is not None:
                params[key] = value
        if params:
            overrides["parameters"] = params
        return overrides or None

    def _n_z_model_label(self, source_config: dict[str, Any] | None) -> str:
        if not source_config:
            return "default"
        return str(source_config.get("type") or "unspecified")

    @staticmethod
    def _meta_description() -> str:
        return (
            "Weak-lensing efficiency kernel W(z) for a given source redshift distribution n(z). "
            "The kernel encodes how matter at different redshifts contributes to the observed "
            "shear signal, and is computed from H(z), comoving distance, and n(z)."
        )
