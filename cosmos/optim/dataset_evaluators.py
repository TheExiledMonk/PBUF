"""
Stateless dataset evaluators for deterministic grid scoring.

Each evaluator instantiates a fresh cosmological model from the provided
parameter dictionary and computes the corresponding χ² for a single dataset.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable
import numpy as np

from cosmos.fits.bao.aniso.chi2 import chi_squared_bao_aniso
from cosmos.fits.bao.iso.chi2 import chi_squared_bao_iso
from cosmos.fits.cc.chi2 import chi_squared_cc
from cosmos.fits.cmb.observables import chi_squared_cmb  # Use standardized version
from cosmos.fits.rsd.chi2 import chi_squared_rsd
from cosmos.fits.sn.chi2 import chi_squared_sn
from cosmos.fits.sn.pantheon.chi2 import chi2_sn_pantheon, chi2_sn_pantheon_abs
from cosmos.fits.sn.sh0es.chi2 import chi2_sn_sh0es
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF

OMEGA_R0 = 9.2e-5
OMEGA_K0 = 0.0
DEFAULT_T_CMB = 2.7255


def _penalty_value(order: int) -> float:
    """
    Generate progressively longer all-nine sentinels.

    order=0 -> 99999, order=1 -> 999999, etc.
    """
    length = 5 + order
    digits = int("9" * length)
    return float(digits)


CHI2_PENALTIES: Dict[str, float] = {
    "validation_failed": _penalty_value(0),
    "physics_scorecard_failed": _penalty_value(0),
    "phase6a_prefilter_failed": _penalty_value(1),
    "phase6a_prefilter_error": _penalty_value(2),
    "dataset_eval_error": _penalty_value(3),
    "dataset_nonfinite": _penalty_value(4),
    "chi2_missing": _penalty_value(5),
    "nonfinite_total": _penalty_value(6),
    "generic": _penalty_value(7),
}

HUGE_CHI2 = CHI2_PENALTIES["generic"]


class ModelConstructionError(RuntimeError):
    """Raised when we cannot build a cosmological model from the inputs."""
    pass

def _with_defaults(params: Dict[str, float]) -> Dict[str, float]:
    """
    Copy the input dictionary and inject shared default parameters.
    """
    import traceback
    import sys
    
    # Log the input parameters
    if "Or0" in params and abs(float(params["Or0"]) - 1.0) < 0.1:  # If Or0 is suspiciously close to 1.0
        print("\n" + "="*80, file=sys.stderr)
        print("WARNING: Suspicious Or0 in _with_defaults", file=sys.stderr)
        print(f"Input params: {params}", file=sys.stderr)
        print("Call stack:", file=sys.stderr)
        traceback.print_stack(limit=10, file=sys.stderr)
        print("\n" + "="*80, file=sys.stderr)
    
    params = params.copy()
    params.setdefault("Or0", OMEGA_R0)
    
    # Log if we're about to return a suspicious Or0
    if abs(float(params["Or0"]) - 1.0) < 0.1:  # If Or0 is suspiciously close to 1.0
        print("\n" + "="*80, file=sys.stderr)
        print("WARNING: Returning suspicious Or0 from _with_defaults", file=sys.stderr)
        print(f"Final params: {params}", file=sys.stderr)
        print("Call stack:", file=sys.stderr)
        traceback.print_stack(limit=10, file=sys.stderr)
        print("\n" + "="*80, file=sys.stderr)
    
    params.setdefault("Ok0", OMEGA_K0)
    return params


def build_model(model_type: str, params: Dict[str, float]):
    """
    Construct a fresh LCDM or PBUF instance from raw parameters.
    """
    import traceback
    import sys
    
    # Log initial parameters if they look suspicious
    if "Or0" in params and abs(float(params.get("Or0", 0)) - 1.0) < 0.1:
        print("\n" + "="*80, file=sys.stderr)
        print("WARNING: Suspicious Or0 in build_model input", file=sys.stderr)
        print(f"Input params: {params}", file=sys.stderr)
        print("Call stack:", file=sys.stderr)
        traceback.print_stack(limit=10, file=sys.stderr)
        print("\n" + "="*80, file=sys.stderr)
    
    model_type = model_type.lower()
    params = _with_defaults(params)
    
    # Log after _with_defaults
    if abs(float(params.get("Or0", 0)) - 1.0) < 0.1:
        print("\n" + "="*80, file=sys.stderr)
        print("WARNING: Suspicious Or0 after _with_defaults", file=sys.stderr)
        print(f"Params after _with_defaults: {params}", file=sys.stderr)
        print("Call stack:", file=sys.stderr)
        traceback.print_stack(limit=10, file=sys.stderr)
        print("\n" + "="*80, file=sys.stderr)
    
    try:
        H0 = float(params["H0"])
        Om0 = float(params["Om0"])
    except KeyError as exc:
        raise ModelConstructionError(f"Missing required parameter: {exc.args[0]}") from exc
    
    Or0 = float(params.get("Or0", OMEGA_R0))
    
    # Log if we have a suspicious Or0 value
    if abs(Or0 - 1.0) < 0.1:  # If Or0 is suspiciously close to 1.0
        print("\n" + "="*80, file=sys.stderr)
        print("WARNING: Suspicious Or0 value in build_model", file=sys.stderr)
        print(f"Or0 = {Or0}, H0 = {H0}, Om0 = {Om0}", file=sys.stderr)
        print("Full params:", params, file=sys.stderr)
        print("Call stack:", file=sys.stderr)
        traceback.print_stack(limit=15, file=sys.stderr)
        print("\n" + "="*80, file=sys.stderr)
    
    Ok0 = float(params.get("Ok0", OMEGA_K0))
    h = H0 / 100.0

    omega_b_param = params.get("omega_b")
    obh2_param = params.get("Obh2")
    if omega_b_param is not None:
        omega_b = float(omega_b_param)
    elif obh2_param is not None:
        try:
            omega_b = float(obh2_param) / (h**2)
        except ZeroDivisionError as exc:  # pragma: no cover - defensive
            raise ModelConstructionError("H0 must be non-zero to compute Ω_b from Obh2.") from exc
    else:
        try:
            omega_b = _infer_baryons(H0)
        except ValueError as exc:
            raise ModelConstructionError(str(exc)) from exc

    if omega_b <= 0.0 or not np.isfinite(omega_b):
        raise ModelConstructionError(f"Ω_b must be positive and finite, got {omega_b}")

    if model_type == "lcdm":
        Ol0 = 1.0 - (Om0 + Or0 + Ok0)
        if Ol0 <= 0:
            raise ModelConstructionError(f"ΩΛ must be positive, got {Ol0}")
        try:
            return LCDM(
                omega_m=Om0,
                omega_lambda=Ol0,
                h=h,
                omega_k=Ok0,
                omega_r=Or0,
                omega_b=omega_b,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelConstructionError(str(exc)) from exc

    if model_type == "pbuf":
        try:
            alpha = float(params["alpha"])
            Rmax = float(params["Rmax"])
            k_sat = float(params["k_sat"])
            eps0 = float(params.get("eps0", 0.7))
            n_alpha = float(params.get("n_alpha", 0.0))
            n_eps = float(params.get("n_eps", 0.0))
            n_R = float(params.get("n_R", 0.0))
        except KeyError as exc:
            raise ModelConstructionError(f"PBUF parameter missing: {exc.args[0]}") from exc
        if alpha < 0 or Rmax <= 0 or k_sat <= 0 or eps0 <= 0:
            raise ModelConstructionError("PBUF parameters out of bounds.")
        try:
            return PBUF(
                omega_m=Om0,
                h=h,
                alpha=alpha,
                Rmax=Rmax,
                k_sat=k_sat,
                eps0=eps0,
                n_alpha=n_alpha,
                n_eps=n_eps,
                n_R=n_R,
                omega_k=Ok0,
                omega_r=Or0,
                omega_b=omega_b,
                T_cmb=DEFAULT_T_CMB,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelConstructionError(str(exc)) from exc

    raise ModelConstructionError(f"Unsupported model_type '{model_type}'")



def _infer_baryons(H0: float) -> float:
    """
    Infer present-day baryon density Ω_b from H0.

    Uses Planck baseline Ω_b h² ≈ 0.02237.
    Returns Ω_b = (Ω_b h²) / h² = 0.02237 / (H0 / 100)².
    """
    if H0 <= 0:
        raise ValueError("H0 must be positive to infer baryons.")
    return 0.02237 / ((H0 / 100.0) ** 2)


def _wrap_dataset(func: Callable, model_type: str, params: Dict[str, float]) -> float:
    """
    Helper to instantiate a new model for each dataset evaluation.
    """
    model = build_model(model_type, params)
    value = float(func(model))
    if not np.isfinite(value):
        return CHI2_PENALTIES["dataset_nonfinite"]
    return value


def evaluate_cmb(model_type: str, params: Dict[str, float]) -> float:
    return _wrap_dataset(chi_squared_cmb, model_type, params)


def evaluate_sn(model_type: str, params: Dict[str, float]) -> float:
    return _wrap_dataset(chi_squared_sn, model_type, params)


def evaluate_bao_iso(model_type: str, params: Dict[str, float]) -> float:
    return _wrap_dataset(chi_squared_bao_iso, model_type, params)


def evaluate_bao_aniso(model_type: str, params: Dict[str, float]) -> float:
    return _wrap_dataset(chi_squared_bao_aniso, model_type, params)


def evaluate_cc(model_type: str, params: Dict[str, float]) -> float:
    return _wrap_dataset(chi_squared_cc, model_type, params)


def _wrap_dataset_dict(func: Callable, model_type: str, params: Dict[str, float]) -> float:
    """
    Helper to instantiate a new model for each dataset evaluation (dict return version).
    """
    def model_func(p):
        return build_model(model_type, p)

    result = func(model_func, params)
    if isinstance(result, dict):
        if "chi2" not in result:
            return CHI2_PENALTIES["chi2_missing"]
        chi2_value = result["chi2"]
    else:
        chi2_value = float(result)

    if not np.isfinite(chi2_value):
        return CHI2_PENALTIES["dataset_nonfinite"]
    return chi2_value


def evaluate_sn_pantheon(model_type: str, params: Dict[str, float]) -> float:
    return _wrap_dataset_dict(chi2_sn_pantheon_abs, model_type, params)


def evaluate_sn_pantheon_abs(model_type: str, params: Dict[str, float]) -> float:
    """
    Alias evaluator for explicit absolute-magnitude Pantheon runs.
    """
    return evaluate_sn_pantheon(model_type, params)


def evaluate_sn_sh0es(model_type: str, params: Dict[str, float]) -> float:
    return _wrap_dataset_dict(chi2_sn_sh0es, model_type, params)


def evaluate_rsd(model_type: str, params: Dict[str, float]) -> float:
    params_copy = dict(params or {})
    sigma8_0 = params_copy.pop("sigma8_0", None)
    model = build_model(model_type, params_copy)
    model_type_lower = model_type.lower()
    if model_type_lower == "pbuf":
        value = float(chi_squared_rsd(model))
    elif sigma8_0 is not None:
        value = float(chi_squared_rsd(model, sigma8_0=sigma8_0))
    else:
        value = float(chi_squared_rsd(model))
    if not np.isfinite(value):
        return CHI2_PENALTIES["dataset_nonfinite"]
    return value


DATASET_EVALUATORS: Dict[str, Callable[[str, Dict[str, float]], float]] = {
    "cmb": evaluate_cmb,
    "sn_pantheon": evaluate_sn_pantheon,
    "sn_pantheon_abs": evaluate_sn_pantheon_abs,
    "sn_sh0es": evaluate_sn_sh0es,
    "bao_iso": evaluate_bao_iso,
    "bao_aniso": evaluate_bao_aniso,
    "cc": evaluate_cc,
    "rsd": evaluate_rsd,
}


def list_available_datasets() -> Iterable[str]:
    return tuple(DATASET_EVALUATORS.keys())


__all__ = [
    "DATASET_EVALUATORS",
    "HUGE_CHI2",
    "CHI2_PENALTIES",
    "ModelConstructionError",
    "build_model",
    "evaluate_bao_aniso",
    "evaluate_bao_iso",
    "evaluate_cc",
    "evaluate_cmb",
    "evaluate_rsd",
    "evaluate_sn_pantheon",
    "evaluate_sn_pantheon_abs",
    "evaluate_sn_sh0es",
    "list_available_datasets",
]
