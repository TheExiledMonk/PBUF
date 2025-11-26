from __future__ import annotations

from typing import Sequence

from cosmos.models.basin_engine import BaseWorker, BasinManager, OptimisationConfig, ParameterSpec
from cosmos.models.lcdm import get_optimisable_parameters, run_microphysics_bootstrap
from cosmos.optim.utils.bounds import load_bounds

from .worker import (
    LCDMWorkerBAOANISO,
    LCDMWorkerBAOISO,
    LCDMWorkerCMB,
    LCDMWorkerGalaxyPK,
    LCDMWorkerRSD,
    LCDMWorkerSH0ES,
    LCDMWorkerSN,
    LCDMWorkerWL,
    LCDMWorkerLensingCross,
)


class ModelLoader:
    MODEL_NAME = "lcdm"

    def __init__(self, dataset_names: Sequence[str]):
        self.dataset_names = [name.lower() for name in dataset_names]
        self.bounds_payload = load_bounds(self.MODEL_NAME)
        self.parameter_specs = self._build_parameter_specs()

    def _build_parameter_specs(self) -> list[ParameterSpec]:
        raw_specs = get_optimisable_parameters()
        specs: list[ParameterSpec] = []
        for record in raw_specs:
            specs.append(
                ParameterSpec(
                    name=record["name"],
                    lower=float(record["lower"]),
                    upper=float(record["upper"]),
                    prior=str(record.get("prior", "flat")),
                )
            )
        return specs

    def _worker_factory(self) -> Sequence[BaseWorker]:
        workers: list[BaseWorker] = []
        for dataset in self.dataset_names:
            if dataset == "cmb":
                workers.append(LCDMWorkerCMB())
            elif dataset == "sn":
                workers.append(LCDMWorkerSN())
            elif dataset == "sh0es":
                workers.append(LCDMWorkerSH0ES())
            elif dataset == "bao_iso":
                workers.append(LCDMWorkerBAOISO())
            elif dataset == "bao_aniso":
                workers.append(LCDMWorkerBAOANISO())
            elif dataset == "rsd":
                workers.append(LCDMWorkerRSD())
            elif dataset == "wl_s8":
                workers.append(LCDMWorkerWL())
            elif dataset == "lensing_cross":
                workers.append(LCDMWorkerLensingCross())
            elif dataset == "galaxy_pk":
                workers.append(LCDMWorkerGalaxyPK())
            else:
                raise NotImplementedError(f"LCDM dataset '{dataset}' is not supported.")
        return workers

    def _run_microphysics(self) -> dict[str, object]:
        return run_microphysics_bootstrap(self.dataset_names)

    def build_manager(self, config: OptimisationConfig) -> BasinManager:
        return BasinManager(
            model_name=self.MODEL_NAME,
            parameter_specs=self.parameter_specs,
            bounds_payload=self.bounds_payload,
            dataset_names=self.dataset_names,
            config=config,
            worker_factory=self._worker_factory,
            quantum_runner=self._run_microphysics,
        )
