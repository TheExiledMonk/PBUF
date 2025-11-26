"""Convenience helpers for LCDM optimisations."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import threading
import numpy as np

from cosmos.datasets import get_dataset
from cosmos.fits.bao_aniso import run_bao_aniso_fit
from cosmos.fits.bao_iso import run_bao_iso_fit
from cosmos.fits.cc import run_cc_fit
from cosmos.fits.rsd import run_rsd_fit
from cosmos.models.lcdm.model import LCDMModel

ParamDict = Dict[str, float]


class LCDMBasinModel:
    """Evaluator used by the optimisation CLI for LCDM."""

    def __init__(self, *, dataset_weights: dict[str, float] | None = None) -> None:
        self._datasets: Dict[str, Any] = {}
        self._dataset_lock = threading.Lock()
        self._dataset_weights = {
            key.lower(): float(value)
            for key, value in (dataset_weights or {}).items()
        }

    def ensure_quantum_and_thermal_table(self, *, datasets: Sequence[str] | None = None) -> None:
        """LCDM has no quantum step, so this is a no-op."""
        return None

    def evaluate(self, params: ParamDict, dataset_names: Sequence[str]) -> float:
        sanitized = {key: float(value) for key, value in params.items()}
        model = LCDMModel(**sanitized)
        total = 0.0
        for dataset_name in dataset_names:
            key = dataset_name.lower()
            weight = self._dataset_weights.get(key, 1.0)
            if key == "cmb":
                dataset = self._ensure_dataset("cmb")
                total += weight * _cmb_chi2(model, dataset)
            elif key == "bao_iso":
                dataset = self._ensure_dataset("bao_iso")
                total += weight * _lcdm_bao_iso_chi2(model, dataset)
            elif key == "bao_aniso":
                dataset = self._ensure_dataset("bao_aniso")
                total += weight * _lcdm_bao_aniso_chi2(model, dataset)
            elif key == "cc":
                dataset = self._ensure_dataset("cc")
                total += weight * _lcdm_cc_chi2(model, dataset)
            elif key == "rsd":
                dataset = self._ensure_dataset("rsd")
                total += weight * _lcdm_rsd_chi2(model, dataset)
            else:
                raise ValueError(f"Dataset '{dataset_name}' is not supported for LCDM optimisation.")
        return total

    def _ensure_dataset(self, name: str) -> Any:
        key = name.lower()
        with self._dataset_lock:
            if key not in self._datasets:
                self._datasets[key] = get_dataset(key)
            return self._datasets[key]


def _cmb_chi2(model: LCDMModel, dataset: Any) -> float:
    output = model.cmb(dataset)
    predicted = np.array([output.R, output.l_A, output.theta_star], dtype=float)
    residual = predicted - dataset.observed
    return float(residual.T @ dataset.inv_covariance @ residual)


def _lcdm_bao_iso_chi2(model: LCDMModel, dataset: Any) -> float:
    chi2, _ = run_bao_iso_fit(model, dataset)
    return chi2


def _lcdm_bao_aniso_chi2(model: LCDMModel, dataset: Any) -> float:
    chi2, _ = run_bao_aniso_fit(model, dataset)
    return chi2


def _lcdm_cc_chi2(model: LCDMModel, dataset: Any) -> float:
    chi2, _ = run_cc_fit(model, dataset)
    return chi2


def _lcdm_rsd_chi2(model: LCDMModel, dataset: Any) -> float:
    chi2, _ = run_rsd_fit(model, dataset)
    return chi2
