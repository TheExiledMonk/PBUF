from __future__ import annotations

from typing import Dict

from cosmos.datasets import get_dataset
from cosmos.fits.bao_aniso import run_bao_aniso_fit
from cosmos.fits.bao_iso import run_bao_iso_fit
from cosmos.fits.galaxy_pk import run_galaxy_pk_fit
from cosmos.fits.rsd import run_rsd_fit
from cosmos.fits.lensing_cross import run_lensing_cross_fit
from cosmos.fits.wl import run_wl_s8_fit
from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.lcdm.sanity import check_lcdm_sanity
from cosmos.models.basin_engine import BaseWorker
from fits.sh0es.sh0es_prior import run_sh0es_prior
from fits.sn.sn_pantheon import run_sn_pantheon_fit

ModelParams = Dict[str, float]
_LCDM_REQUIRED = {"H0", "Omega_m0", "Omega_b0", "Omega_r0", "Omega_k0"}


def _normalize_params(params: ModelParams) -> ModelParams:
    normalized: ModelParams = {}
    missing = [name for name in _LCDM_REQUIRED if name not in params]
    if missing:
        raise ValueError(f"LCDM worker missing parameters {missing}")
    for name in _LCDM_REQUIRED:
        normalized[name] = float(params[name])
    return normalized


class _LCDMWorkerBase(BaseWorker):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.dataset_loader = get_dataset(dataset_name)

    def build_model(self, params: ModelParams) -> LCDMModel:
        normalized = _normalize_params(params)
        return LCDMModel(**normalized)

    def run_model_sanity(self, params: ModelParams, model: LCDMModel) -> tuple[bool, list[str]]:
        normalized = _normalize_params(params)
        result = check_lcdm_sanity(normalized, model)
        return result.ok, list(result.reasons)


class LCDMWorkerCMB(_LCDMWorkerBase):
    def __init__(self):
        super().__init__("cmb")

    def compute_chi2(self, model: LCDMModel) -> float:
        output, residual, chi2 = model.chi2_cmb(self.dataset_loader)
        return float(chi2)


class LCDMWorkerSN(_LCDMWorkerBase):
    def __init__(self):
        super().__init__("sn")

    def compute_chi2(self, model: LCDMModel) -> float:
        chi2, _ = run_sn_pantheon_fit(model, self.dataset_loader)
        return chi2


class LCDMWorkerSH0ES(_LCDMWorkerBase):
    def __init__(self):
        super().__init__("sh0es")

    def compute_chi2(self, model: LCDMModel) -> float:
        chi2, _ = run_sh0es_prior(model, self.dataset_loader)
        return chi2


class LCDMWorkerBAOISO(_LCDMWorkerBase):
    def __init__(self):
        super().__init__("bao_iso")

    def compute_chi2(self, model: LCDMModel) -> float:
        chi2, _ = run_bao_iso_fit(model, self.dataset_loader)
        return chi2


class LCDMWorkerBAOANISO(_LCDMWorkerBase):
    def __init__(self):
        super().__init__("bao_aniso")

    def compute_chi2(self, model: LCDMModel) -> float:
        chi2, _ = run_bao_aniso_fit(model, self.dataset_loader)
        return chi2


class LCDMWorkerRSD(_LCDMWorkerBase):
    def __init__(self):
        super().__init__("rsd")

    def compute_chi2(self, model: LCDMModel) -> float:
        chi2, _ = run_rsd_fit(model, self.dataset_loader)
        return chi2


class LCDMWorkerWL(_LCDMWorkerBase):
    def __init__(self):
        super().__init__("wl_s8")

    def compute_chi2(self, model: LCDMModel) -> float:
        chi2, _ = run_wl_s8_fit(model, self.dataset_loader)
        return chi2


class LCDMWorkerLensingCross(_LCDMWorkerBase):
    def __init__(self):
        super().__init__("lensing_cross")

    def compute_chi2(self, model: LCDMModel) -> float:
        chi2, _ = run_lensing_cross_fit(model, self.dataset_loader)
        return chi2


class LCDMWorkerGalaxyPK(_LCDMWorkerBase):
    def __init__(self):
        super().__init__("galaxy_pk")

    def compute_chi2(self, model: LCDMModel) -> float:
        chi2, _ = run_galaxy_pk_fit(model, self.dataset_loader)
        return chi2
