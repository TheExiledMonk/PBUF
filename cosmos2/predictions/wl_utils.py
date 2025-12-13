"""Helper utilities for weak-lensing source distributions."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_SOURCE_PRESETS: dict[str, dict[str, float]] = {
    "lsst_like": {"z0": 0.3, "alpha": 2.0, "beta": 1.5},
    "euclid_like": {"z0": 0.9, "alpha": 2.0, "beta": 1.4},
    "simple": {"z0": 0.5, "alpha": 2.0, "beta": 1.2},
}


def _coerce_float(value: Any) -> float | None:
    """Return a float if convertible, otherwise None."""

    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    return float_value


def wl_source_distribution(
    z_grid: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """
    Return a normalized n(z) on the provided redshift grid.

    Parameters
    ----------
    z_grid:
        Redshift samples (will be treated as-is).
    config:
        Optional descriptor specifying ``type`` and ``parameters`` for the
        parametric distribution. Supported ``type`` values: "lsst_like",
        "euclid_like", "simple". Additional keys ``z0``, ``alpha``, and
        ``beta`` override the defaults.
    """

    z_arr = np.asarray(z_grid, dtype=float)
    if z_arr.size == 0:
        return z_arr.copy()

    preset_type = "lsst_like"
    overrides: dict[str, float] = {}
    if isinstance(config, Mapping):
        explicit_type = config.get("type")
        if isinstance(explicit_type, str) and explicit_type.strip():
            preset_type = explicit_type.strip().lower()
        source_parameters = config.get("parameters")
        if isinstance(source_parameters, Mapping):
            for key in ("z0", "alpha", "beta"):
                override = _coerce_float(source_parameters.get(key))
                if override is not None:
                    overrides[key] = override
        for key in ("z0", "alpha", "beta"):
            override = _coerce_float(config.get(key))
            if override is not None:
                overrides[key] = override

    preset = _SOURCE_PRESETS.get(preset_type, _SOURCE_PRESETS["simple"])
    z0 = max(overrides.get("z0", preset["z0"]), 1e-4)
    alpha = max(overrides.get("alpha", preset["alpha"]), 0.0)
    beta = max(overrides.get("beta", preset["beta"]), 0.1)

    positive_mask = z_arr >= 0.0
    safe_z = np.where(positive_mask, z_arr, 0.0)
    exponent = -(safe_z / z0) ** beta
    raw = np.zeros_like(z_arr, dtype=float)
    raw[positive_mask] = np.power(safe_z[positive_mask], alpha, dtype=float) * np.exp(exponent[positive_mask])

    integral = np.trapz(raw, z_arr)
    if integral <= 0.0 or not np.isfinite(integral):
        return np.zeros_like(z_arr, dtype=float)
    return raw / integral


__all__ = ["wl_source_distribution"]
