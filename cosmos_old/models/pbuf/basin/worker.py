"""PBUF dataset workers for the basin engine."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cosmos.datasets import get_dataset
from cosmos.fits.bao_aniso import run_bao_aniso_fit
from cosmos.fits.bao_iso import run_bao_iso_fit
from cosmos.fits.galaxy_pk import run_galaxy_pk_fit
from cosmos.fits.lensing_cross import run_lensing_cross_fit
from cosmos.fits.rsd import run_rsd_fit
from cosmos.fits.wl import run_wl_s8_fit
from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.pbuf.microphysics import ensure_thermal_table, get_last_bootstrap_metadata
from cosmos.models.pbuf.model import PBUFModel
from cosmos.models.pbuf.sanity import check_pbuf_sanity
from cosmos.models.pbuf.thermal_table import ThermalTable
from cosmos.models.basin_engine import BaseWorker
from fits.sh0es.sh0es_prior import run_sh0es_prior
from fits.sn.sn_pantheon import run_sn_pantheon_fit

ModelParams = Dict[str, float]
_PBUF_REQUIRED = {"H0", "Omega_m0", "Omega_b0", "Rmax"}


def _lcdm_factory(**kwargs: float) -> LCDMModel:
    return LCDMModel(**kwargs)


def _ensure_params(params: ModelParams) -> ModelParams:
    normalized = {}
    missing = [name for name in _PBUF_REQUIRED if name not in params]
    if missing:
        raise ValueError(f"PBUF worker missing parameters {missing}")
    for name in _PBUF_REQUIRED:
        normalized[name] = float(params[name])
    normalized.setdefault("Omega_r0", 9.0e-5)
    if "alpha" in params:
        normalized["alpha"] = float(params["alpha"])
    return normalized


class _PBUFWorkerBase(BaseWorker):
    _shared_table: ThermalTable | None = None
    _shared_metadata: dict[str, Any] | None = None

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.dataset_loader = get_dataset(dataset_name)
        self._thermal = self._load_thermal_table()
        self._thermal_metadata = self._load_thermal_metadata()

    def build_model(self, params: ModelParams) -> PBUFModel:
        normalized = _ensure_params(params)
        return PBUFModel(
            thermal_table=self._thermal,
            thermal_metadata=self._thermal_metadata,
            **normalized,
        )

    @classmethod
    def _load_thermal_table(cls) -> ThermalTable:
        if cls._shared_table is None:
            cls._shared_table = ensure_thermal_table()
        return cls._shared_table

    @classmethod
    def _load_thermal_metadata(cls) -> dict[str, Any] | None:
        if cls._shared_metadata is None:
            cls._shared_metadata = get_last_bootstrap_metadata()
        return cls._shared_metadata

    def run_model_sanity(self, params: ModelParams, model: PBUFModel) -> tuple[bool, list[str]]:
        normalized = _ensure_params(params)
        result = check_pbuf_sanity(normalized, model, lcdm_model_factory=_lcdm_factory)
        return result.ok, list(result.reasons)


class PBUFWorkerCMB(_PBUFWorkerBase):
    def __init__(self):
        super().__init__("cmb")

    def compute_chi2(self, model: PBUFModel) -> float:
        output = model.cmb(self.dataset_loader)
        predicted = np.array([output.R, output.l_A, output.theta_star], dtype=float)
        residual = predicted - self.dataset_loader.observed
        return float(residual.T @ self.dataset_loader.inv_covariance @ residual)


class PBUFWorkerSN(_PBUFWorkerBase):
    def __init__(self):
        super().__init__("sn")

    def compute_chi2(self, model: PBUFModel) -> float:
        chi2, _ = run_sn_pantheon_fit(model, self.dataset_loader)
        return chi2


class PBUFWorkerSH0ES(_PBUFWorkerBase):
    def __init__(self):
        super().__init__("sh0es")

    def compute_chi2(self, model: PBUFModel) -> float:
        chi2, _ = run_sh0es_prior(model, self.dataset_loader)
        return chi2


class PBUFWorkerBAOISO(_PBUFWorkerBase):
    def __init__(self):
        super().__init__("bao_iso")

    def compute_chi2(self, model: PBUFModel) -> float:
        chi2, _ = run_bao_iso_fit(model, self.dataset_loader)
        return chi2


class PBUFWorkerBAOANISO(_PBUFWorkerBase):
    def __init__(self):
        super().__init__("bao_aniso")

    def compute_chi2(self, model: PBUFModel) -> float:
        chi2, _ = run_bao_aniso_fit(model, self.dataset_loader)
        return chi2


class PBUFWorkerRSD(_PBUFWorkerBase):
    def __init__(self):
        super().__init__("rsd")

    def compute_chi2(self, model: PBUFModel) -> float:
        chi2, _ = run_rsd_fit(model, self.dataset_loader)
        return chi2


class PBUFWorkerWL(_PBUFWorkerBase):
    def __init__(self):
        super().__init__("wl_s8")

    def compute_chi2(self, model: PBUFModel) -> float:
        chi2, _ = run_wl_s8_fit(model, self.dataset_loader)
        return chi2


class PBUFWorkerLensingCross(_PBUFWorkerBase):
    def __init__(self):
        super().__init__("lensing_cross")

    def compute_chi2(self, model: PBUFModel) -> float:
        chi2, _ = run_lensing_cross_fit(model, self.dataset_loader)
        return chi2


class PBUFWorkerGalaxyPK(_PBUFWorkerBase):
    def __init__(self):
        super().__init__("galaxy_pk")

    def compute_chi2(self, model: PBUFModel) -> float:
        chi2, _ = run_galaxy_pk_fit(model, self.dataset_loader)
        return chi2
