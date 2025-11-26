"""MG-ΛCDM model helpers."""

from __future__ import annotations

from typing import Dict, List

from cosmos.models.mg_lcdm.model import MGLCDMModel
from cosmos.models.mg_lcdm.params import MGLCDMParams, get_default_parameters
from cosmos.optim.utils.bounds import load_bounds


_MG_PARAM_SPECS: List[Dict[str, float | str]] = [
    {"name": "H0", "lower": 40.0, "upper": 90.0, "prior": "flat"},
    {"name": "Omega_m0", "lower": 0.05, "upper": 0.6, "prior": "flat"},
    {"name": "Omega_b0", "lower": 0.02, "upper": 0.08, "prior": "flat"},
    {"name": "Omega_r0", "lower": 9.0e-5, "upper": 9.0e-5, "prior": "fixed"},
    {"name": "Omega_k0", "lower": -0.2, "upper": 0.2, "prior": "flat"},
    {"name": "mu0", "lower": -1.0, "upper": 1.0, "prior": "flat"},
    {"name": "Sigma0", "lower": -1.0, "upper": 1.0, "prior": "flat"},
]


def get_optimisable_parameters() -> List[Dict[str, float | str]]:
    return list(_MG_PARAM_SPECS)


def get_boundaries() -> dict:
    return load_bounds("lcdm")


__all__ = [
    "MGLCDMModel",
    "MGLCDMParams",
    "get_optimisable_parameters",
    "get_boundaries",
    "get_default_parameters",
]
