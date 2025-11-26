"""Model-level helpers that power optimisation and optimisation-adjacent tooling."""

from __future__ import annotations

from typing import Callable, Dict, Protocol, Sequence, Tuple

ParamDict = Dict[str, float]
SanityFn = Callable[[ParamDict], Tuple[bool, str | None]]


class OptimisationModel(Protocol):
    def ensure_quantum_and_thermal_table(self, *, datasets: Sequence[str] | None = None) -> None:
        ...

    def evaluate(self, params: ParamDict, datasets: Sequence[str]) -> float:
        ...

    def phase7a_checker(self) -> SanityFn:
        ...

    def phase6a_checker(self) -> SanityFn:
        ...


def get_model(model_name: str, *, dataset_weights: dict[str, float] | None = None) -> OptimisationModel:
    normalized = model_name.strip().lower()
    if normalized == "lcdm":
        from cosmos.models.lcdm.basin_utils import LCDMBasinModel

        return LCDMBasinModel(dataset_weights=dataset_weights)
    if normalized == "pbuf":
        from cosmos.models.pbuf.basin_utils import PBUFBasinModel

        return PBUFBasinModel(dataset_weights=dataset_weights)
    raise ValueError(f"Unsupported model '{model_name}'.")


def get_phase7a_checker(model_name: str, *, model_context: OptimisationModel | None = None) -> SanityFn:
    normalized = model_name.strip().lower()
    if normalized == "pbuf":
        if model_context is None:
            raise ValueError("PBUF phase7a checker requires an ensured model context.")
        return model_context.phase7a_checker()
    raise ValueError(f"Unsupported model '{model_name}'.")


def get_phase6a_checker(model_name: str, *, model_context: OptimisationModel | None = None) -> SanityFn:
    normalized = model_name.strip().lower()
    if normalized == "lcdm":
        from cosmos.models.lcdm.phase6a import phase6a_lcdm

        return phase6a_lcdm
    if normalized in {"ede", "ede_lcdm"}:
        from cosmos.models.ede_lcdm.phase6a import phase6a_ede

        return phase6a_ede
    if normalized in {"running_lambda", "lcdm_rlambda", "rlambda"}:
        from cosmos.models.running_lambda.phase6a import phase6a_running_lambda

        return phase6a_running_lambda
    if normalized == "pbuf":
        return get_phase7a_checker(model_name, model_context=model_context)
    raise ValueError(f"Unsupported model '{model_name}'.")
