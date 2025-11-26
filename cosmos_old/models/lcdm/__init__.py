"""LCDM model implementation package."""

from __future__ import annotations

import json
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
from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.lcdm.optim import BasinConfig, BasinWalker
from cosmos.models.lcdm.params import LCDMParams
from cosmos.optim.utils.bounds import load_bounds

_DEFAULT_DATASETS = ["cmb"]
_LCDM_PARAM_SPECS: List[Dict[str, float | str]] = [
    {"name": "H0", "lower": 40.0, "upper": 90.0, "prior": "flat"},
    {"name": "Omega_m0", "lower": 0.05, "upper": 0.6, "prior": "flat"},
    {"name": "Omega_b0", "lower": 0.02, "upper": 0.08, "prior": "flat"},
    {"name": "Omega_r0", "lower": 9.0e-5, "upper": 9.0e-5, "prior": "fixed"},
    {"name": "Omega_k0", "lower": -0.2, "upper": 0.2, "prior": "flat"},
]


def run_microphysics_bootstrap(datasets: Sequence[str] | None = None) -> Dict[str, object]:
    return {
        "micro_hash": None,
        "thermal_table_path": None,
        "datasets": list(datasets or _DEFAULT_DATASETS),
        "engine_source": "lcdm-noop",
    }


def get_optimisable_parameters() -> List[Dict[str, float | str]]:
    return list(_LCDM_PARAM_SPECS)


def get_boundaries() -> dict:
    """Return the system-level bounds for the LCDM parameters."""
    return load_bounds("lcdm")


def evaluate_chi2(params: Dict[str, float], datasets: Sequence[str] | None = None) -> float:
    dataset_names = [name.lower() for name in (datasets or _DEFAULT_DATASETS)]
    model = LCDMModel(**_coerce_params(params))
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
            raise ValueError(f"LCDM evaluate_chi2 does not support dataset '{name}'.")
    return float(total)


def build_basin_walker(config: BasinConfig | None = None, bounds: dict | None = None) -> BasinWalker:
    resolved = config or BasinConfig(datasets=_DEFAULT_DATASETS)
    return BasinWalker(sys.modules[__name__], resolved, bounds=bounds)


def _coerce_params(params: Dict[str, float]) -> Dict[str, float]:
    required = {spec["name"] for spec in _LCDM_PARAM_SPECS}
    missing = sorted(key for key in required if key not in params)
    if missing:
        raise ValueError(f"LCDM evaluate_chi2 missing parameters: {missing}")
    return {name: float(params[name]) for name in required}


def _cmb_chi2(model: LCDMModel, dataset: CMBDataset) -> float:
    output, residual, chi2 = model.chi2_cmb(dataset)

    if not getattr(dataset, "_sigmas_logged", False):
        print("Sigmas used for chi2:", dataset.sigmas)
        dataset._sigmas_logged = True

    weighted = dataset.inv_covariance @ residual
    contributions = residual * weighted

    info = {
        "H0": float(model.params.H0),
        "Omega_m0": float(model.params.Omega_m0),
        "Omega_b0": float(model.params.Omega_b0),
        "chi2_total": chi2,
        "chi2_components": {
            "cmb_distance": float(contributions[0]),
            "lA": float(contributions[1]),
            "theta_star": float(contributions[2]),
            "r_s": float(output.r_s_Mpc),
        },
    }

    print("χ² evaluation info:")
    print(json.dumps(info, indent=2))

    return chi2


__all__ = [
    "BasinConfig",
    "BasinWalker",
    "LCDMModel",
    "LCDMParams",
    "build_basin_walker",
    "evaluate_chi2",
    "get_optimisable_parameters",
    "run_microphysics_bootstrap",
    "get_boundaries",
]
