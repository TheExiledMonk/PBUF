"""PBUF model implementation and optimisation helpers."""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence

import numpy as np

from cosmos.datasets import CMBDataset, get_dataset
from cosmos.fits.bao_aniso import run_bao_aniso_fit
from cosmos.fits.bao_iso import run_bao_iso_fit
from cosmos.fits.galaxy_pk import run_galaxy_pk_fit
from cosmos.fits.lensing_cross import run_lensing_cross_fit
from cosmos.fits.rsd import run_rsd_fit
from cosmos.fits.wl import run_wl_s8_fit
from cosmos.models.pbuf.microphysics import ensure_thermal_table, run_microphysics_bootstrap as _run_bootstrap
from cosmos.models.pbuf.model import PBUFModel
from cosmos.models.pbuf.optim import BasinConfig, BasinWalker
from cosmos.models.pbuf.params import PBUFParams
from cosmos.optim.utils.bounds import load_bounds

_DEFAULT_DATASETS = ["cmb"]
_PBUF_PARAM_SPECS: List[Dict[str, float | str]] = [
    {"name": "H0", "lower": 40.0, "upper": 90.0, "prior": "flat"},
    {"name": "Omega_m0", "lower": 0.05, "upper": 0.8, "prior": "flat"},
    {"name": "Omega_b0", "lower": 0.02, "upper": 0.08, "prior": "flat"},
    {"name": "Rmax", "lower": 1.0e5, "upper": 1.0e8, "prior": "flat"},
]


def run_microphysics_bootstrap(datasets: Sequence[str] | None = None) -> Dict[str, object]:
    """
    Trigger the Quantum bootstrap pipeline exactly once prior to optimisation.
    """

    return _run_bootstrap(list(datasets or _DEFAULT_DATASETS))


def get_optimisable_parameters() -> List[Dict[str, float | str]]:
    return list(_PBUF_PARAM_SPECS)


def get_boundaries() -> dict:
    """Return the system-level bounds for the PBUF parameters."""
    return load_bounds("pbuf")


def evaluate_chi2(params: Dict[str, float], datasets: Sequence[str] | None = None) -> float:
    dataset_names = [name.lower() for name in (datasets or _DEFAULT_DATASETS)]
    model = PBUFModel(thermal_table=ensure_thermal_table(), **_coerce_params(params))

    total = 0.0
    for name in dataset_names:
        if name == "cmb":
            dataset = get_dataset("cmb")
            total += _cmb_chi2(model, dataset)
        elif name == "bao_iso":
            dataset = get_dataset("bao_iso")
            chi2, _ = run_bao_iso_fit(model, dataset)
            total += chi2
        elif name == "bao_aniso":
            dataset = get_dataset("bao_aniso")
            chi2, _ = run_bao_aniso_fit(model, dataset)
            total += chi2
        elif name == "rsd":
            dataset = get_dataset("rsd")
            chi2, _ = run_rsd_fit(model, dataset)
            total += chi2
        elif name == "wl_s8":
            dataset = get_dataset("wl_s8")
            chi2, _ = run_wl_s8_fit(model, dataset)
            total += chi2
        elif name == "galaxy_pk":
            dataset = get_dataset("galaxy_pk")
            chi2, _ = run_galaxy_pk_fit(model, dataset)
            total += chi2
        elif name == "lensing_cross":
            dataset = get_dataset("lensing_cross")
            chi2, _ = run_lensing_cross_fit(model, dataset)
            total += chi2
        else:
            raise ValueError(f"PBUF evaluate_chi2 does not support dataset '{name}'.")
    return float(total)


def build_basin_walker(config: BasinConfig | None = None, bounds: dict | None = None) -> BasinWalker:
    resolved = config or BasinConfig(datasets=_DEFAULT_DATASETS)
    return BasinWalker(sys.modules[__name__], resolved, bounds=bounds)


def _coerce_params(params: Dict[str, float]) -> Dict[str, float]:
    required = {spec["name"] for spec in _PBUF_PARAM_SPECS}
    missing = sorted(key for key in required if key not in params)
    if missing:
        raise ValueError(f"PBUF evaluate_chi2 missing parameters: {missing}")
    coerced = {name: float(params[name]) for name in required}
    return coerced


def _cmb_chi2(model: PBUFModel, dataset: CMBDataset) -> float:
    output = model.cmb(dataset)
    predicted = np.array([output.R, output.l_A, output.theta_star])
    residual = predicted - dataset.observed
    return float(residual.T @ dataset.inv_covariance @ residual)


__all__ = [
    "BasinConfig",
    "BasinWalker",
    "PBUFModel",
    "PBUFParams",
    "build_basin_walker",
    "evaluate_chi2",
    "get_optimisable_parameters",
    "run_microphysics_bootstrap",
    "get_boundaries",
]
