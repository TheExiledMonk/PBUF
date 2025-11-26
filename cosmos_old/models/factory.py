"""Model factory helpers for the joint fit runner."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

from cosmos.interfaces import CosmologyModel
from cosmos.models.desi_mod import MODEL_OBJECT as DESI_MOD_MODEL_OBJECT
from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.pbuf.microphysics import ensure_thermal_table, run_microphysics_bootstrap
from cosmos.models.pbuf.model import PBUFModel

ParamDict = Dict[str, float]
ModelFactory = Callable[[ParamDict], CosmologyModel]


def _sanitize_params(params: ParamDict) -> ParamDict:
    return {key: float(value) for key, value in params.items()}


def make_model_factory(model_name: str, *, datasets: Sequence[str] | None = None) -> ModelFactory:
    """
    Return a callable that instantiates the requested cosmology model.

    The returned factory will coerce every parameter value to float, ensuring the
    downstream fits can rely on clean numerics.
    """

    normalized = model_name.strip().lower()
    if normalized == "lcdm":
        def _factory(params: ParamDict) -> CosmologyModel:
            return LCDMModel(**_sanitize_params(params))

        return _factory

    if normalized in {"lcdm_mg", "mg_lcdm"}:
        def _factory(params: ParamDict) -> CosmologyModel:
            from cosmos.models.mg_lcdm.model import MGLCDMModel

            return MGLCDMModel(**_sanitize_params(params))

        return _factory

    if normalized == "pbuf":
        ordered = [name.strip().lower() for name in (datasets or [])]
        unique = list(dict.fromkeys(ordered))
        metadata = run_microphysics_bootstrap(unique)
        thermal_table = ensure_thermal_table()

        def _factory(params: ParamDict) -> CosmologyModel:
            return PBUFModel(
                thermal_table=thermal_table,
                thermal_metadata=metadata,
                **_sanitize_params(params),
            )

        return _factory

    if normalized == "desi_mod":
        def _factory(_: ParamDict) -> CosmologyModel:
            # DESI_mod is not yet plugged into the runtime fits; expose the
            # module bundle so pytest can exercise the math.
            return DESI_MOD_MODEL_OBJECT  # type: ignore[return-value]

        return _factory

    if normalized == "dgp":
        def _factory(params: ParamDict) -> CosmologyModel:
            from cosmos.models.dgp.model import DGPModel

            return DGPModel(**_sanitize_params(params))

        return _factory

    if normalized in {"running_lambda", "lcdm_rlambda", "rlambda"}:
        def _factory(params: ParamDict) -> CosmologyModel:
            from cosmos.models.running_lambda.model import RunningLambdaModel

            return RunningLambdaModel(**_sanitize_params(params))

        return _factory

    raise ValueError(f"Unknown model '{model_name}'.")
