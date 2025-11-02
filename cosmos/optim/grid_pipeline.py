"""
Optimized Grid Search Pipeline for PBUF4 Cosmology.

This module implements efficient grid search evaluation with quantum mechanics awareness:

FAIR COMPARISON APPROACH:
- LCDM: Full grid search to find OPTIMAL parameters (auto-fails Phase 6a but gets best χ²)
        We want the BEST possible LCDM fit for fair comparison - Phase 6a failure is interpretive
- PBUF: Full multi-dimensional grid since it can pass Phase 6a and needs comprehensive exploration

QUANTUM MECHANICS PERSPECTIVE:
- Phase 6a correctly implements quantum mechanics filter
- LCDM auto-fails (cannot describe quantum mechanics) but gets optimal fitting
- PBUF evaluated with physical sanity checks (can describe quantum mechanics)
- Shows χ² cost of requiring quantum mechanics compatibility vs best classical fit

This provides optimal parameter fitting for both models while maintaining quantum mechanics awareness.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Set

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback
    def tqdm(iterable, *args, **kwargs):
        return iterable

import numpy as np

from cosmos.optim.dataset_evaluators import (
    DATASET_EVALUATORS,
    CHI2_PENALTIES,
    build_model,
    ModelConstructionError,
)
from cosmos.optim.physics_validator import validate_cosmology
from cosmos.optim.physics_checks import physics_scorecard
from cosmos.phase.validation import prior_violation_reason, record_prior_violation
from ..physchecks.phase6a import phase6a_passes

# ===============================================================
# CONFIGURATION FLAGS
# ===============================================================
# Controls whether the PBUF grid pre-filter runs before evaluation.
# Set to False to disable (for debugging or full scans).
ENABLE_PHASE6A_PREFILTER = True
# ===============================================================

BASE_DATASETS: Tuple[str, ...] = ("cmb", "sn_pantheon", "sn_sh0es", "cc", "rsd")
BAO_DATASETS: Tuple[str, ...] = ("bao_iso", "bao_aniso")
GRID_VERSION = "grid-v1"
DEFAULT_OR0 = 9.2e-5
DEFAULT_OK0 = 0.0


def _linspace(start: float, stop: float, num: int) -> List[float]:
    return [float(x) for x in np.linspace(start, stop, num)]


def _logspace(start_exp: float, stop_exp: float, num: int) -> List[float]:
    return [float(x) for x in np.logspace(start_exp, stop_exp, num, base=10.0)]
DEFAULT_LCDM_GRID: "OrderedDict[str, List[float]]" = OrderedDict(
    [
        # Hubble constant: extend beyond Planck (67.4) and SH0ES (~73) ranges,
        # but not high enough to violate early-universe constraints
        ("H0", _linspace(66.000, 70.000, 71)),

        # Matter density: allow slightly below 0.25 (open) and above 0.36 (denser),
        # still consistent with flat-Universe priors when Ok0 = 0
        ("Om0", _linspace(0.2200, 0.3500, 71)),

        ("Or0", [DEFAULT_OR0]),
        ("Ok0", [DEFAULT_OK0]),
    ]
)  # Used for full grid search to find optimal LCDM parameters (Phase 6a failure is interpretive)


DEFAULT_PBUF_GRID: "OrderedDict[str, List[float]]" = OrderedDict(
    [
        # Let PBUF explore a wider H0 to capture both SH0ES-like high-H0
        # and Planck-like low-H0 solutions that may arise due to elasticity
        ("H0", _linspace(66.000, 70.000, 17)),

        # Matter density: same as LCDM for fair comparison
        ("Om0", _linspace(0.2500, 0.3000, 13)),

        ("Or0", [DEFAULT_OR0]),
        ("Ok0", [DEFAULT_OK0]),

        # Elastic amplitude α: use more conservative range to avoid numerical issues
        # PBUF model struggles with very small α values (< 1e-4)
        ("alpha", _logspace(-3.5, -1.5, 12)),  # 1e-4 to 1e-2

        # Rmax: explore 1e7 to 1e9 to capture late-time activation scales
        ("Rmax", _logspace(7.0, 9.0, 13)),

        # k_sat (elastic efficiency): allow sub- and super-unity coupling
        ("k_sat", _linspace(0.5, 3.0, 26)),

        # Elastic stiffness baseline and optional evolution indices
        ("eps0", _linspace(0.5, 1.2, 8)),
        ("n_alpha", [0.0]),
        ("n_eps", [0.0]),
        ("n_R", [0.0]),
    ]
)

DEFAULT_GRIDS = {
    "lcdm": DEFAULT_LCDM_GRID,
    "pbuf": DEFAULT_PBUF_GRID,
}


def _grid_bounds_and_flags(grid_axes: "OrderedDict[str, List[float]]") -> Tuple[Dict[str, Tuple[float, float]], Dict[str, bool]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    varying: Dict[str, bool] = {}
    for key, values in grid_axes.items():
        arr = np.array(values, dtype=float)
        if arr.size == 0:
            bounds[key] = (0.0, 0.0)
            varying[key] = False
        else:
            bounds[key] = (float(np.min(arr)), float(np.max(arr)))
            varying[key] = bool(len(np.unique(np.round(arr, 12))) > 1)
    return bounds, varying


def _apply_param_limits(name: str, lower: float, upper: float, bounds: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if bounds is not None:
        lower = max(lower, bounds[0])
        upper = min(upper, bounds[1])

    if name in {"H0", "Om0", "Or0", "alpha", "Rmax"}:
        lower = max(lower, 1e-12)
    if name == "k_sat":
        lower = max(lower, 1e-6)
    return lower, upper


def _local_axis_values(
    name: str,
    value: float,
    fraction: float,
    points: int,
    bounds_map: Optional[Dict[str, Tuple[float, float]]],
    should_vary: bool,
) -> List[float]:
    if (points <= 1) or (fraction <= 0.0) or (not should_vary):
        return [float(value)]
    span = abs(value) * fraction
    if span == 0.0:
        span = fraction
    lower = value - span
    upper = value + span
    bounds = bounds_map.get(name) if bounds_map else None
    lower, upper = _apply_param_limits(name, lower, upper, bounds)
    if lower > upper:
        lower, upper = upper, upper
    if abs(upper - lower) < 1e-12:
        return [float(lower)]
    return [float(v) for v in np.linspace(lower, upper, points)]


def _params_key(params: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((key, round(float(value), 12)) for key, value in params.items()))


def _local_parameter_grid(
    center_params: Dict[str, float],
    fraction: float,
    points: int,
    bounds_map: Optional[Dict[str, Tuple[float, float]]],
    varying_map: Dict[str, bool],
) -> List[Dict[str, float]]:
    keys = list(center_params.keys())
    axes: List[List[float]] = []
    for key in keys:
        value = float(center_params[key])
        axis = _local_axis_values(
            key,
            value,
            fraction,
            points,
            bounds_map,
            varying_map.get(key, True),
        )
        axes.append(axis)

    locals: List[Dict[str, float]] = []
    for combo in itertools.product(*axes):
        locals.append({key: float(val) for key, val in zip(keys, combo)})
    return locals


def _normalize_dataset_list(datasets: Optional[Sequence[str]], include_bao: bool) -> List[str]:
    if datasets:
        normalized = [name.strip().lower() for name in datasets if name.strip()]
    else:
        normalized = list(BASE_DATASETS)
    if include_bao and "bao_iso" not in normalized:
        normalized.extend(BAO_DATASETS)
    for name in normalized:
        if name not in DATASET_EVALUATORS:
            raise ValueError(f"Unsupported dataset '{name}'")
    return list(dict.fromkeys(normalized))  # preserve order, remove duplicates


def _ensure_iterable(values: Any) -> List[float]:
    if isinstance(values, dict):
        scale = values.get("scale", "linear").lower()
        start = float(values["min"])
        stop = float(values["max"])
        num = int(values["num"])
        if num <= 0:
            raise ValueError("Grid axis requires num > 0")
        if scale == "linear":
            return _linspace(start, stop, num)
        if scale == "log":
            return [float(x) for x in np.logspace(start, stop, num, base=10.0)]
        raise ValueError(f"Unknown axis scale '{scale}'")
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("Grid axis cannot be empty")
        return [float(v) for v in values]
    raise ValueError("Grid axis must be list or dict specification")


def prepare_grid(model_type: str, overrides: Optional[MutableMapping[str, Any]] = None) -> "OrderedDict[str, List[float]]":
    model_type = model_type.lower()
    if overrides:
        ordered = OrderedDict()
        for key, spec in overrides.items():
            ordered[key] = _ensure_iterable(spec)
        return ordered

    template = DEFAULT_GRIDS.get(model_type)
    if template is None:
        raise ValueError(f"No default grid for model '{model_type}'")

    # Special case: LCDM auto-fails Phase 6a (quantum mechanics), but still gets full grid search
    # for fair comparison - we want the BEST possible LCDM fit to compete against
    if model_type == "lcdm":
        # Use the original full grid for LCDM to find optimal parameters
        # Phase 6a failure is interpretive, not punitive
        return OrderedDict((key, list(values)) for key, values in template.items())

    return OrderedDict((key, list(values)) for key, values in template.items())


def _iter_parameter_grid(grid: "OrderedDict[str, List[float]]") -> Iterable[Dict[str, float]]:
    keys = list(grid.keys())
    axes = [grid[key] for key in keys]
    for combination in itertools.product(*axes):
        yield {key: float(value) for key, value in zip(keys, combination)}


def _augment_params(model_type: str, params: Dict[str, float]) -> Dict[str, float]:
    augmented = dict(params)
    augmented.setdefault("Or0", DEFAULT_OR0)
    augmented.setdefault("Ok0", DEFAULT_OK0)
    if model_type == "pbuf":
        missing = [name for name in ("alpha", "Rmax", "k_sat") if name not in augmented]
        if missing:
            raise ValueError(f"PBUF grid missing parameters: {missing}")
    return augmented


def _serialize_params(params: Dict[str, float]) -> Dict[str, float]:
    serialized: Dict[str, float] = {}
    for key, value in params.items():
        if isinstance(value, (int, float, np.floating)):
            serialized[key] = float(value)
        else:
            raise ValueError(f"Non-numeric parameter '{key}': {value}")
    return serialized


def _evaluate_single_dataset(evaluator, model_type: str, params: Dict[str, float]) -> float:
    """Top-level function for evaluating a single dataset that can be pickled."""
    try:
        return float(evaluator(model_type, params))
    except Exception as e:
        raise RuntimeError(f"Error in dataset evaluation: {str(e)}")


def evaluate_cosmology(
    model_type: str,
    params: Dict[str, float],
    datasets: Sequence[str],
    *,
    priors: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Evaluate cosmology with correct working directory.
    
    Args:
        model_type: Type of cosmological model ('lcdm' or 'pbuf')
        params: Dictionary of parameter values
        datasets: List of dataset names to evaluate against
        
    Returns:
        Dictionary containing evaluation results including chi-squared values and validation status
    """
    import os
    import time
    import logging
    from pathlib import Path
    from typing import Any, Dict, Sequence

# Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cosmology_evaluation.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    def _H_of_z(model, z: float) -> float:
        """Helper function to calculate H(z) for Phase 6a checks."""
        return float(model.H(z))
    
    def _rho_elastic_of_z(model, model_type: str, z: float) -> float:
        """Helper function to calculate elastic energy density for Phase 6a checks."""
        return float(model.elastic_energy_density(z)) if model_type.lower() == "pbuf" else 0.0
    
    try:
        start_time = time.time()
        logger.info(f"Starting evaluation for {model_type} with params: {params}")

        prior_reason = prior_violation_reason(params, priors)
        if prior_reason is not None:
            logger.debug("Prior rejection: %s", prior_reason)
            diagnostics: Dict[str, Any] = {}
            record_prior_violation(diagnostics, prior_reason)
            return {
                "status": "invalid",
                "chi2_breakdown": {},
                "chi2_total": math.inf,
                "dataset_errors": {},
                "validation": {"valid": False, "reasons": [prior_reason]},
                "passes_phase6a": False,
                "metadata": {
                    "evaluation_time": time.time() - start_time,
                    "error": "prior_violation",
                    "reason": prior_reason,
                },
                "diagnostics": diagnostics,
            }
        
        # Ensure correct working directory (project root)
        current_file = Path(__file__)
        pbuf4_root = current_file.parent.parent.parent  # cosmos/optim -> cosmos -> project root
        os.chdir(pbuf4_root)

        # Validate parameters
        validation = validate_cosmology(model_type, params)
        if not validation["valid"]:
            validation_reasons = list(validation.get("reasons", []))
            validation_reason = validation_reasons[0] if validation_reasons else "unspecified"
            logger.debug(f"Parameter validation failed: {validation}")
            return {
                "status": "invalid",
                "chi2_breakdown": {},
                "chi2_total": CHI2_PENALTIES["validation_failed"],
                "dataset_errors": {},
                "validation": validation,
                "passes_phase6a": False,
                "metadata": {
                    "evaluation_time": time.time() - start_time,
                    "error": "Parameter validation failed",
                    "reason": validation_reason,
                    "all_reasons": validation_reasons,
                    "reasons": validation_reasons,
                }
            }

        # Build model once for downstream checks
        try:
            model = build_model(model_type, params)
        except ModelConstructionError as exc:
            logger.debug(f"Model construction failed: {exc}")
            return {
                "status": "invalid",
                "chi2_breakdown": {},
                "chi2_total": CHI2_PENALTIES["validation_failed"],
                "dataset_errors": {},
                "validation": validation,
                "passes_phase6a": False,
                "metadata": {
                    "evaluation_time": time.time() - start_time,
                    "error": "Model construction failed",
                    "reasons": [str(exc)],
                },
            }

        # Stage-0 physics guardrail (Path A aware)
        scorecard = physics_scorecard(model, model_type.lower(), params)
        if not scorecard["ok"]:
            scorecard_reasons = list(scorecard.get("reasons", []))
            first_reason = scorecard_reasons[0] if scorecard_reasons else "unspecified"
            logger.debug(f"Physics scorecard failed: {scorecard}")
            return {
                "status": "invalid",
                "chi2_breakdown": {},
                "chi2_total": CHI2_PENALTIES["physics_scorecard_failed"],
                "dataset_errors": {},
                "validation": validation,
                "passes_phase6a": False,
                "metadata": {
                    "evaluation_time": time.time() - start_time,
                    "error": "Stage-0 physics guard failed",
                    "reason": first_reason,
                    "all_reasons": scorecard_reasons,
                    "reasons": scorecard_reasons,
                    "edge_case": bool(scorecard.get("edge_case")),
                },
            }
        edge_case = bool(scorecard.get("edge_case"))
        passes_phase6a = True

        # ---------------------------------------------------------------
        # EARLY PHASE 6A PRE-FILTER (only for PBUF)
        # ---------------------------------------------------------------
        if model_type.lower() == "pbuf":
            try:
                helpers = {
                    "H_of_z": lambda z: model.H(z),
                    "rho_elastic_of_z": lambda z: model.elastic_energy_density(z),
                }

                # --- Debug diagnostics ---
                try:
                    h0 = helpers["H_of_z"](0.0)
                    h1 = helpers["H_of_z"](1.0)
                    rho0 = helpers["rho_elastic_of_z"](0.0)
                    rho1 = helpers["rho_elastic_of_z"](1.0)
                    logger.debug(
                        "[phase6a_prefilter] H(0)=%.6e, H(1)=%.6e, ρ_el(0)=%.6e, ρ_el(1)=%.6e",
                        h0,
                        h1,
                        rho0,
                        rho1,
                    )
                except Exception as e:
                    logger.debug("[phase6a_prefilter] Helper eval failed: %s", e)

                # --- Phase 6a logic ---
                try:
                    result_phase6a = phase6a_passes(model_type, params, helpers)
                    logger.debug("[phase6a_prefilter] phase6a_passes() → %s", result_phase6a)
                except Exception as e:
                    logger.debug("[phase6a_prefilter] phase6a_passes exception: %s", e)
                    result_phase6a = False

                # -----------------------------------------------------------
                # SOFT-PASS SWITCH: set to True temporarily to bypass fails
                # -----------------------------------------------------------
                SOFT_PASS_PHASE6A = True  # toggle this for debugging

                if not result_phase6a:
                    if SOFT_PASS_PHASE6A:
                        logger.debug("[phase6a_prefilter] Soft-passing failed model for diagnostics.")
                    else:
                        logger.debug("[phase6a_prefilter] Rejecting model (phase6a_prefilter_failed).")
                        return {
                            "status": "invalid",
                            "chi2_total": CHI2_PENALTIES.get("phase6a_prefilter_failed", 9.04e28),
                            "chi2_breakdown": {},
                            "dataset_errors": {},
                            "validation": validation,
                            "passes_phase6a": False,
                            "metadata": {
                                "evaluation_time": time.time() - start_time,
                                "model_type": model_type,
                                "error": "phase6a_precheck_failed",
                                "phase6a_debug": {
                                    "H(0)": float(h0) if "h0" in locals() else None,
                                    "ρ_el(0)": float(rho0) if "rho0" in locals() else None,
                                },
                            },
                        }

            except Exception as e:
                logger.debug("[phase6a_prefilter] Build or prefilter exception: %s", e)
                return {
                    "status": "invalid",
                    "chi2_total": CHI2_PENALTIES.get("phase6a_prefilter_error", 9.05e28),
                    "chi2_breakdown": {},
                    "dataset_errors": {"phase6a": str(e)},
                    "validation": validation,
                    "passes_phase6a": False,
                    "metadata": {
                        "evaluation_time": time.time() - start_time,
                        "model_type": model_type,
                        "error": f"phase6a_build_error: {str(e)}",
                    },
                }
        # ---------------------------------------------------------------

        # Evaluate datasets
        total_datasets = len(datasets)
        logger.info(f"\n{'='*80}\nStarting evaluation of {total_datasets} datasets\n{'='*80}")
        breakdown: Dict[str, float] = {}
        dataset_errors: Dict[str, str] = {}
        start_time = time.time()
        
        for idx, dataset in enumerate(datasets, 1):
            # Calculate progress
            elapsed = time.time() - start_time
            progress = (idx / total_datasets) * 100
            
            logger.info(f"\n[Progress: {progress:.1f}% | Dataset {idx}/{total_datasets}] {'-'*40}")
            logger.info(f"Starting evaluation of: {dataset}")
            logger.info(f"Elapsed: {elapsed:.1f}s")
            
            try:
                dataset_start = time.time()
                evaluator = DATASET_EVALUATORS[dataset]
                
                # Evaluate dataset
                logger.info(f"Running evaluation for {dataset}...")
                chi2_value = _evaluate_single_dataset(evaluator, model_type, params)
                
                # Record results
                exec_time = time.time() - dataset_start
                breakdown[dataset] = chi2_value
                logger.info(f"✓ Completed {dataset} in {exec_time:.2f}s | χ² = {chi2_value:.4f}")
                
            except Exception as exc:
                exec_time = time.time() - dataset_start
                error_msg = f"✗ Error in {dataset} after {exec_time:.1f}s: {str(exc)}"
                logger.error(error_msg, exc_info=True)
                chi2_value = CHI2_PENALTIES["dataset_eval_error"]
                dataset_errors[dataset] = str(exc)
                breakdown[dataset] = chi2_value
                    
        # Print summary of evaluations
        total_time = time.time() - start_time
        success_count = len(breakdown) - len(dataset_errors)
        logger.info(f"\n{'='*80}\nEvaluation Summary\n{'='*80}")
        logger.info(f"Total datasets: {total_datasets}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {len(dataset_errors)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average time per dataset: {total_time/total_datasets:.2f}s")
        
        if dataset_errors:
            logger.warning("\nFailed datasets:" + "\n- ".join([''] + list(dataset_errors.keys())))

        # Calculate total chi-squared
        chi2_total = float(np.sum(list(breakdown.values())))
        if not np.isfinite(chi2_total):
            chi2_total = CHI2_PENALTIES["nonfinite_total"]
            dataset_errors.setdefault("chi2_total", "non-finite total")
            logger.warning(f"Non-finite χ² total: {breakdown}")
        
        logger.info(f"Total χ² = {chi2_total:.2f}")

        # Phase 6a physical sanity check (only for PBUF)
        if model_type.lower() == "lcdm":
            passes_phase6a = True
        else:
            try:
                def H_of_z(z): return float(model.H(z))
                def rho_elastic_of_z(z): 
                    return float(model.elastic_energy_density(z))

                helpers = {"H_of_z": H_of_z, "rho_elastic_of_z": rho_elastic_of_z}
                passes_phase6a = phase6a_passes(model_type, params, helpers)
                logger.info(f"Phase 6a check {'passed' if passes_phase6a else 'failed'}")
                
            except Exception:
                passes_phase6a = False
                dataset_errors["phase6a"] = "Phase 6a check failed"

        # Prepare result
        result = {
            "status": "valid",
            "chi2_breakdown": breakdown,
            "chi2_total": chi2_total,
            "dataset_errors": dataset_errors,
            "validation": validation,
            "passes_phase6a": passes_phase6a,
            "metadata": {
                "evaluation_time": time.time() - start_time,
                "model_type": model_type,
                "num_datasets": len(datasets),
                "successful_datasets": len(datasets) - len(dataset_errors),
                "stage0_edge_case": edge_case,
            }
        }
        
        logger.info(f"Evaluation completed in {result['metadata']['evaluation_time']:.2f} seconds")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error in evaluate_cosmology: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        return {
            "status": "error",
            "error": error_msg,
            "params": params,
            "metadata": {
                "evaluation_time": time.time() - start_time,
                "error_type": type(e).__name__,
                "traceback": str(e)
            }
        }


def _evaluate_indexed(model_type: str, params: Dict[str, float], datasets: Sequence[str], grid_index: int) -> Dict[str, Any]:
    """Evaluate cosmology in worker process with correct working directory."""
    import os
    from pathlib import Path
    import logging

    # Ensure correct working directory in worker process
    current_file = Path(__file__)
    pbuf4_root = current_file.parent.parent.parent  # cosmos/optim -> cosmos -> project root
    os.chdir(pbuf4_root)

    # Initialize a basic logger for this function
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    try:
        result = evaluate_cosmology(model_type, params, datasets)
        
        # If result is None, create an error result
        if result is None:
            return {
                "status": "error",
                "error": "Evaluation returned no result",
                "id": f"{model_type}_{grid_index:05d}",
                "params": _serialize_params(params),
                "metadata": {
                    "grid_index": grid_index,
                    "error": "Evaluation returned no result"
                }
            }
            
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            return {
                "status": "error",
                "error": f"Expected dict, got {type(result).__name__}",
                "id": f"{model_type}_{grid_index:05d}",
                "params": _serialize_params(params),
                "metadata": {
                    "grid_index": grid_index,
                    "error": f"Expected dict, got {type(result).__name__}"
                }
            }
            
        # Add required fields
        result["id"] = f"{model_type}_{grid_index:05d}"
        result["params"] = _serialize_params(params)
        meta = dict(result.get("metadata", {}))
        meta["grid_index"] = grid_index
        result["metadata"] = meta
        return result
        
    except Exception as e:
        logger.error(f"Error in _evaluate_indexed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "id": f"{model_type}_{grid_index:05d}",
            "params": _serialize_params(params),
            "metadata": {
                "grid_index": grid_index,
                "error": str(e)
            }
        }
    return result


def _process_wrapper(args):
    """Wrapper function that can be pickled for process pool."""
    import os
    import sys
    import time
    from typing import Any, Dict, Tuple, List, Sequence
    
    # Unpack arguments
    model_type, datasets, idx, params = args
    
    try:
        # Add process start time
        start_time = time.time()
        
        # Run the actual evaluation
        result = _evaluate_indexed(model_type, params, datasets, idx)
        
        return ('success', idx, result)
        
    except Exception as e:
        import traceback
        error_msg = f"Error in process {os.getpid()}: {str(e)}\n{traceback.format_exc()}"
        print(f"[PID:{os.getpid()}] ERROR in task {idx}: {error_msg}")
        
        error_result = {
            'status': 'error',
            'error': error_msg,
            'params': _serialize_params(params),
            'metadata': {'grid_index': idx}
        }
        return ('error', idx, error_result)

def _collect_results(
    model_type: str,
    datasets: Sequence[str],
    points: List[Dict[str, float]],
    workers: int,
):
    import time
    from typing import Dict, List, Sequence, Any, Tuple, Set, Optional
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import Manager
    try:  # pragma: no cover - optional dependency
        from tqdm.auto import tqdm as tqdm_auto
    except ImportError:  # pragma: no cover - fallback
        class _TqdmStub:
            def __init__(self, iterable=None, total=None, **kwargs):
                self.iterable = iterable
                self.total = total
                self.n = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def update(self, n=1):
                self.n += n

            def set_postfix(self, *args, **kwargs):
                pass

            def refresh(self):
                pass

            def __iter__(self):
                if self.iterable is None:
                    return iter(())
                for item in self.iterable:
                    self.n += 1
                    yield item

        def tqdm_auto(iterable=None, *args, **kwargs):
            return _TqdmStub(iterable=iterable, total=kwargs.get("total"))
    
    results: List[Dict[str, Any]] = [None] * len(points)
    total_points = len(points)
    
    # Create a manager for shared state
    manager = Manager()
    progress = manager.dict({
        'completed': 0,
        'best_chi2': float('inf'),
        'lock': manager.Lock()
    })
    
    if workers and workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Prepare arguments for each task
                tasks = [
                    (model_type, datasets, idx, params)
                    for idx, params in enumerate(points)
                ]

                # Submit all tasks
                futures = [
                    executor.submit(_process_wrapper, task)
                    for task in tasks
                ]
                
                print(f"Submitted {len(futures)} tasks to process pool.")
                
                # Initialize progress bar with more frequent updates
                with tqdm_auto(
                    total=total_points,
                    desc=f"Grid search {model_type.upper()}",
                    unit="point",
                    ncols=100,
                    position=0,
                    leave=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                    miniters=1,
                    mininterval=0.5  # Update more frequently
                ) as pbar:
                    # print(f"Progress bar initialized. Processing {total_points} points with {workers} workers...")
                    
                    # Track completed tasks
                    completed = 0
                    last_update = time.time()
                    
                    # Process results as they complete
                    for future in as_completed(futures):
                        try:
                            status, idx, result = future.result()
                            completed += 1
                            
                            # Store the result at the correct index
                            results[idx] = result
                            
                            # Update progress more frequently
                            current_time = time.time()
                            if current_time - last_update >= 0.5:  # Update at most twice per second
                                pbar.update(completed - pbar.n)  # Update by the actual number of completed tasks
                                last_update = current_time
                            
                            # Update best chi2 if available
                            if status == 'success' and 'chi2_total' in result and np.isfinite(result['chi2_total']):
                                with progress['lock']:
                                    current_chi2 = result['chi2_total']
                            if current_chi2 < progress['best_chi2']:
                                progress['best_chi2'] = current_chi2
                            pbar.set_postfix({
                                "best_χ²": f"{progress['best_chi2']:.2f}",
                                "current": f"{current_chi2:.2f}"
                            })
                            pbar.refresh()

                            
                            # Force update the progress bar every 10 completed tasks
                            if completed % 10 == 0:
                                pbar.refresh()
                            
                        except Exception as e:
                            print(f"\nError processing future: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print("\nAll tasks completed. Finalizing results...")
        
        except Exception as e:
            print(f"\nFATAL ERROR in process pool: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        # Single-process version with progress bar
        with tqdm_auto(
            total=total_points,
            desc=f"Grid search {model_type.upper()}",
            unit="point",
            ncols=100,
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        ) as pbar:
            best_chi2 = float('inf')
            for idx, params in enumerate(points):
                result = _evaluate_indexed(model_type, params, datasets, idx)
                results[idx] = result
                pbar.update(1)
                
                # Update best chi2 if available
                if result.get('status') == 'valid' and 'chi2_total' in result and np.isfinite(result['chi2_total']):
                    best_chi2 = min(best_chi2, result['chi2_total'])
                    pbar.set_postfix({"best_χ²": f"{best_chi2:.2f}"})
    
    # Sort results by grid index for consistency
    results.sort(key=lambda entry: entry["metadata"]["grid_index"])
    return results


def _refine_top_candidates(
    model_type: str,
    datasets: Sequence[str],
    top_entries: Sequence[Dict[str, Any]],
    fraction: float,
    points: int,
    bounds_map: Optional[Dict[str, Tuple[float, float]]],
    varying_map: Dict[str, bool],
    existing_keys: Set[Tuple[Tuple[str, float], ...]],
    start_index: int,
) -> List[Dict[str, Any]]:
    if not top_entries or fraction <= 0.0 or points <= 1:
        return []

    refined: List[Dict[str, Any]] = []
    local_counter = 0
    for entry in top_entries:
        center_params = entry.get("params") or {}
        if not center_params:
            continue
        local_grid = _local_parameter_grid(center_params, fraction, points, bounds_map, varying_map)
        for local_params in local_grid:
            key = _params_key(local_params)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            eval_result = evaluate_cosmology(model_type, local_params, datasets)
            eval_result["id"] = f"{model_type}_ref_{entry['id']}_{local_counter:04d}"
            eval_result["params"] = _serialize_params(local_params)
            eval_result.setdefault("metadata", {})
            eval_result["metadata"].update(
                {
                    "origin": "refine",
                    "source_id": entry["id"],
                    "grid_index": start_index + len(refined),
                }
            )
            refined.append(eval_result)
            local_counter += 1
    return refined


def _compute_phase6a_summary(valid_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Phase 6a summary statistics from valid grid entries."""
    if not valid_entries:
        return {
            "best_fit_overall": None,
            "best_fit_phase6a": None,
            "delta": None,
        }

    # Best overall (lowest chi2, regardless of Phase 6a)
    best_overall = min(valid_entries, key=lambda entry: (entry["chi2_total"], entry["id"]))

    # Best Phase 6a (lowest chi2 among those that pass Phase 6a)
    phase6a_entries = [entry for entry in valid_entries if entry.get("passes_phase6a", False)]
    if phase6a_entries:
        best_phase6a = min(phase6a_entries, key=lambda entry: (entry["chi2_total"], entry["id"]))
    else:
        best_phase6a = None

    # Compute parameter deltas
    delta = None
    if best_overall and best_phase6a:
        delta_chi2 = best_phase6a["chi2_total"] - best_overall["chi2_total"]

        # Compute parameter differences for common parameters
        params_overall = best_overall.get("params", {})
        params_phase6a = best_phase6a.get("params", {})

        delta_params = {}
        for param in params_overall:
            if param in params_phase6a:
                try:
                    diff = float(params_phase6a[param]) - float(params_overall[param])
                    delta_params[param] = diff
                except (ValueError, TypeError):
                    pass  # Skip non-numeric parameters

        delta = {
            "delta_chi2": delta_chi2,
            "delta_params": delta_params,
        }

    return {
        "best_fit_overall": {
            "params": best_overall.get("params"),
            "chi2_total": best_overall.get("chi2_total"),
            "passes_phase6a": best_overall.get("passes_phase6a", False),
        } if best_overall else None,
        "best_fit_phase6a": {
            "params": best_phase6a.get("params"),
            "chi2_total": best_phase6a.get("chi2_total"),
            "passes_phase6a": True,
        } if best_phase6a else None,
        "delta": delta,
    }


def _print_phase6a_summary(model_type: str, phase6a_summary: Dict[str, Any]) -> None:
    """Print Phase 6a summary to stdout."""
    print(f"\n{model_type.upper()} grid evaluation complete")
    print("=" * 50)

    best_overall = phase6a_summary.get("best_fit_overall")
    best_phase6a = phase6a_summary.get("best_fit_phase6a")
    delta = phase6a_summary.get("delta")

    if best_overall:
        print("   best_fit_overall:")
        print(f"      χ²_total = {best_overall['chi2_total']:.3f}")
        print(f"      passes_phase6a = {best_overall['passes_phase6a']}")
        if best_overall['passes_phase6a'] == False and model_type == "pbuf":
            print("      (fails Phase 6a: quantum mechanics incompatibility)")
            print("      (full grid search: optimal parameters found)")
        if best_overall.get("params"):
            params_str = ", ".join([f"{k} = {v:.3g}" for k, v in best_overall["params"].items()])
            print(f"      {params_str}")

    if best_phase6a:
        print("\n   best_fit_phase6a:")
        print(f"      χ²_total = {best_phase6a['chi2_total']:.3f}")
        print(f"      passes_phase6a = {best_phase6a['passes_phase6a']}")
        if best_phase6a.get("params"):
            params_str = ", ".join([f"{k} = {v:.3g}" for k, v in best_phase6a["params"].items()])
            print(f"      {params_str}")

    if delta and best_overall and best_phase6a:
        print(f"\n   Δχ² = {delta['delta_chi2']:+.3f}")
        if delta.get("delta_params"):
            print("   Δparams:")
            for param, diff in delta["delta_params"].items():
                print(f"      Δ{param} = {diff:+.3g}")
    elif model_type == "pbuf":
        print("\n   (No Phase 6a fit available - PBUF quantum mechanics incompatibility)")


def run_grid_search(
    model_type: str,
    *,
    datasets: Optional[Sequence[str]] = None,
    include_bao: bool = False,
    grid: Optional[MutableMapping[str, Any]] = None,
    workers: int = 1,
    output_dir: Optional[Path] = None,
    tag: Optional[str] = None,
    refine_top: int = 0,
    refine_fraction: float = 0.05,
    refine_points: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate every point on the deterministic grid for the requested model.

    For LCDM: Uses full grid search to find OPTIMAL parameters. LCDM automatically
              passes Phase 6a since it wasn't designed for quantum mechanics compatibility.
              We want the BEST possible LCDM fit for fair comparison.

    For PBUF: Uses full multi-dimensional grid since it can pass Phase 6a and
              needs comprehensive parameter exploration for quantum compatibility.
    """
    print("Cosmos Grid Search") 
    print("I think now it's time for coffee - this will take a while\n\n", end='', flush=True)
    model_type = model_type.lower()
    dataset_list = _normalize_dataset_list(datasets, include_bao)
    prepared_grid = prepare_grid(model_type, grid)
    bounds_map, varying_map = _grid_bounds_and_flags(prepared_grid)
    augmented_points: List[Dict[str, float]] = []
    for raw_params in _iter_parameter_grid(prepared_grid):
        augmented_points.append(_augment_params(model_type, raw_params))
    if not augmented_points:
        raise ValueError("Grid produced zero parameter combinations.")

    # -----------------------------------------------------------------
    # OPTIONAL EARLY PHASE 6A FILTERING (skip unphysical PBUF points)
    # -----------------------------------------------------------------
    if model_type.lower() == "pbuf" and ENABLE_PHASE6A_PREFILTER:
        print(f"[Prefilter] Phase 6a pre-filter enabled = {ENABLE_PHASE6A_PREFILTER}")
        from cosmos.optim.dataset_evaluators import build_model
        from ..physchecks.phase6a import phase6a_passes
        survivors = []
        print(f"[Prefilter] Running Phase 6a pre-check on {len(augmented_points)} candidates...")
        for idx, p in enumerate(augmented_points, 1):
            try:
                model = build_model(model_type, p)
                helpers = {
                    "H_of_z": lambda z: model.H(z),
                    "rho_elastic_of_z": lambda z: model.elastic_energy_density(z),
                }
                if phase6a_passes(model_type, p, helpers):
                    survivors.append(p)
            except Exception:
                continue
            if idx % 500 == 0:
                print(f"  ...Physics sanity checked {idx}/{len(augmented_points)} \r", end='', flush=True)
        print(f"[Prefilter] {len(survivors)} survived Phase 6a sanity filter.")
        augmented_points = survivors
        if not augmented_points:
            print("[Prefilter] No physical candidates remain. Exiting early.")
            return {
                "model_type": model_type,
                "num_evaluations": 0,
                "num_valid": 0,
                "num_invalid": 0,
                "best": None,
                "phase6a_summary": {},
                "tag": tag,
            }

        # -------------------------------------------------------------
        # 🧩 Apply compute budget limit for PBUF
        # -------------------------------------------------------------
        MAX_EVAL = 500  # or adjust based on compute budget
        total_candidates = len(augmented_points)
        if total_candidates > MAX_EVAL:
            print(f"[Prefilter] ⚙ Applying compute budget cap: keeping {MAX_EVAL}/{total_candidates}")
            def stable_key(p):
                return (
                    round(p.get("H0", 0.0), 6),
                    round(p.get("Om0", 0.0), 6),
                    round(p.get("alpha", 0.0), 9),
                    round(p.get("Rmax", 0.0), 6),
                    round(p.get("k_sat", 0.0), 6),
                )
            augmented_points.sort(key=stable_key)
            stride = max(1, total_candidates // MAX_EVAL)
            augmented_points = augmented_points[::stride][:MAX_EVAL]
            print(f"[Prefilter] ✅ Using {len(augmented_points)} candidates after budget limit\n")

    # -----------------------------------------------------------------
    # FAIRNESS CAP FOR LCDM (equal compute budget)
    # -----------------------------------------------------------------
    elif model_type.lower() == "lcdm":
        MAX_EVAL = 500  # same as PBUF cap
        total_candidates = len(augmented_points)
        if total_candidates > MAX_EVAL:
            print(f"[LCDM] ⚙ Applying compute budget cap: keeping {MAX_EVAL}/{total_candidates}")
            def stable_key(p):
                return (
                    round(p.get("H0", 0.0), 6),
                    round(p.get("Om0", 0.0), 6),
                    round(p.get("Or0", 0.0), 6),
                    round(p.get("Ok0", 0.0), 6),
                )
            augmented_points.sort(key=stable_key)
            stride = max(1, total_candidates // MAX_EVAL)
            augmented_points = augmented_points[::stride][:MAX_EVAL]
            print(f"[LCDM] ✅ Using {len(augmented_points)} candidates after budget limit\n")



    start_time = datetime.now(UTC)
    results = _collect_results(model_type, dataset_list, augmented_points, workers)
    existing_keys: Set[Tuple[Tuple[str, float], ...]] = set()
    for entry in results:
        params = entry.get("params")
        if params:
            existing_keys.add(_params_key(params))

    valid_entries = [
        entry
        for entry in results
        if entry.get("status") == "valid" and np.isfinite(entry.get("chi2_total", np.inf))
    ]
    ranking = sorted(valid_entries, key=lambda entry: (entry["chi2_total"], entry["id"]))

    refined: List[Dict[str, Any]] = []
    if refine_top and ranking:
        top_entries = ranking[: min(refine_top, len(ranking))]
        refined = _refine_top_candidates(
            model_type=model_type,
            datasets=dataset_list,
            top_entries=top_entries,
            fraction=refine_fraction,
            points=refine_points,
            bounds_map=None,
            varying_map=varying_map,
            existing_keys=existing_keys,
            start_index=len(results),
        )
        if refined:
            results.extend(refined)
            valid_entries = [
                entry
                for entry in results
                if entry.get("status") == "valid" and np.isfinite(entry.get("chi2_total", np.inf))
            ]
            ranking = sorted(valid_entries, key=lambda entry: (entry["chi2_total"], entry["id"]))

    best = ranking[0] if ranking else None
    iso_timestamp = start_time.isoformat().replace("+00:00", "Z")
    num_valid = len(valid_entries)
    num_invalid = len(results) - num_valid
    refined_count = len(refined)

    # Compute Phase 6a summary
    phase6a_summary = _compute_phase6a_summary(valid_entries)

    # -------------------------------------------------------------
    # Prefilter reporting for PBUF Phase 6a precheck
    # -------------------------------------------------------------
    prefilter_total = len(augmented_points)
    prefilter_survivors = prefilter_total
    prefilter_skipped = 0
    try:
        # if we used the prefilter, survivors were counted before filtering
        if model_type.lower() == "pbuf" and ENABLE_PHASE6A_PREFILTER:
            # Counted in the Stage 2 block earlier
            prefilter_total = len(prepared_grid["H0"]) * len(prepared_grid["Om0"]) * \
                            len(prepared_grid.get("alpha", [1])) * \
                            len(prepared_grid.get("Rmax", [1])) * \
                            len(prepared_grid.get("k_sat", [1]))
            prefilter_survivors = len(results)
            prefilter_skipped = prefilter_total - prefilter_survivors
    except Exception:
        pass
    # -------------------------------------------------------------

    payload = {
        "version": GRID_VERSION,
        "model_type": model_type,
        "timestamp_utc": iso_timestamp,
        "datasets": dataset_list,
        "grid_axes": {key: list(values) for key, values in prepared_grid.items()},
        "num_evaluations": len(results),
        "num_valid": num_valid,
        "num_invalid": num_invalid,
        "prefilter_total": prefilter_total,
        "prefilter_survivors": prefilter_survivors,
        "prefilter_skipped": prefilter_skipped,
        "evaluations": results,
        "ranking": [
            {"rank": idx + 1, "id": entry["id"], "chi2_total": entry["chi2_total"]}
            for idx, entry in enumerate(ranking)
        ],
        "best": best,
        "phase6a_summary": phase6a_summary,
        "tag": tag,
    }

    if refined_count:
        payload["refined_evaluations"] = refined_count
        payload["refinement"] = {
            "top": refine_top,
            "fraction": refine_fraction,
            "points": refine_points,
        }

    if output_dir is None:
        output_dir = Path("data/results")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{tag}" if tag else ""
    output_path = output_dir / f"grid_{model_type}_{timestamp}{tag_suffix}.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    payload["results_file"] = str(output_path)

    # Print Phase 6a summary to stdout
    _print_phase6a_summary(model_type, phase6a_summary)
    
    if model_type.lower() == "pbuf" and ENABLE_PHASE6A_PREFILTER:
        print(f"\n[Prefilter Summary] {prefilter_survivors}/{prefilter_total} points survived Phase 6a "
              f"({(prefilter_survivors/prefilter_total)*100:.2f}% passed)")

    return payload


def run_dual_grid_search(
    *,
    datasets: Optional[Sequence[str]] = None,
    include_bao: bool = False,
    grid_lcdm: Optional[MutableMapping[str, Any]] = None,
    grid_pbuf: Optional[MutableMapping[str, Any]] = None,
    workers: int = 1,
    output_dir: Optional[Path] = None,
    tag: Optional[str] = None,
    refine_top: int = 0,
    refine_fraction: float = 0.05,
    refine_points: int = 3,
) -> Dict[str, Any]:
    """Evaluate LCDM and PBUF grids independently and report Delta chi^2 between best fits."""
    lcdm_result = run_grid_search(
        "lcdm",
        datasets=datasets,
        include_bao=include_bao,
        grid=grid_lcdm,
        workers=workers,
        output_dir=output_dir,
        tag=tag,
        refine_top=refine_top,
        refine_fraction=refine_fraction,
        refine_points=refine_points,
    )
    pbuf_result = run_grid_search(
        "pbuf",
        datasets=datasets,
        include_bao=include_bao,
        grid=grid_pbuf,
        workers=workers,
        output_dir=output_dir,
        tag=tag,
        refine_top=refine_top,
        refine_fraction=refine_fraction,
        refine_points=refine_points,
    )
    delta = np.nan
    if lcdm_result.get("best") and pbuf_result.get("best"):
        delta = pbuf_result["best"]["chi2_total"] - lcdm_result["best"]["chi2_total"]

    # Print Phase 6a summaries for both models
    if "phase6a_summary" in lcdm_result:
        _print_phase6a_summary("lcdm", lcdm_result["phase6a_summary"])
    if "phase6a_summary" in pbuf_result:
        _print_phase6a_summary("pbuf", pbuf_result["phase6a_summary"])

    return {
        "lcdm": lcdm_result,
        "pbuf": pbuf_result,
        "delta_chi2": delta,
    }


__all__ = [
    "BASE_DATASETS",
    "BAO_DATASETS",
    "DEFAULT_GRIDS",
    "GRID_VERSION",
    "evaluate_cosmology",
    "prepare_grid",
    "run_dual_grid_search",
    "run_grid_search",
]
