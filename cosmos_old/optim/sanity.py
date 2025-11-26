"""Integration layer that wires models, datasets, and sanity checks together."""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import numpy as np

from cosmos.datasets import get_dataset
from cosmos.fits.bao_aniso import run_bao_aniso_fit
from cosmos.fits.bao_iso import run_bao_iso_fit
from cosmos.fits.cc import run_cc_fit
from cosmos.fits.cmb import run_fit as run_cmb_fit
from cosmos.fits.cmb.sanity import check_cmb_sanity
from cosmos.fits.galaxy_pk import run_galaxy_pk_fit
from cosmos.fits.lensing_cross import load_lensing_cross_dataset, run_lensing_cross_fit
from cosmos.fits.rsd import run_rsd_fit
from cosmos.fits.wl import load_wl_s8_dataset, run_wl_s8_fit
from cosmos.optim.sanity_base import SanityResult
from fits.sh0es.sh0es_prior import run_sh0es_prior
from fits.sn.sn_pantheon import run_sn_pantheon_fit

DatasetSummary = Dict[str, Any]
ModelParams = Dict[str, float]

HUGE_CHI2 = 1.0e9


def evaluate_candidate(
    model_name: str,
    params: ModelParams,
    datasets: Sequence[str] | None = None,
) -> tuple[float, dict]:
    """
    Evaluate a candidate parameter set and guard against insane configurations.
    Returns a χ² value plus metadata about dataset summaries and sanity feedback.
    """

    normalized_params = _normalize_params(params)
    dataset_names, alias_map = _normalize_dataset_names(datasets)
    model = _instantiate_model(model_name, normalized_params)

    dataset_summaries, dataset_errors = _compute_dataset_summaries(model, dataset_names, alias_map)
    extras = {
        "sanity_failed": False,
        "sanity_reasons": [],
        "dataset_summaries": dataset_summaries,
    }
    if dataset_errors:
        extras["sanity_failed"] = True
        extras["sanity_reasons"].extend(dataset_errors)
        return HUGE_CHI2, extras

    cmb_solver = None
    if "cmb" in dataset_summaries:
        cmb_solver = _make_cached_cmb_solver(dataset_summaries["cmb"])

    sanity = run_model_sanity(
        model_name,
        normalized_params,
        model,
        lcdm_model_factory=_lcdm_model_factory,
        cmb_solver=cmb_solver,
    )
    extras["sanity_reasons"].extend(sanity.reasons)
    if not sanity.ok:
        extras["sanity_failed"] = True
        print(f"[SANITY FAIL][{model_name}] {sanity.reasons}")
        return HUGE_CHI2, extras

    for dataset_name in dataset_names:
        summary = dataset_summaries.get(dataset_name)
        if summary is None:
            extras["sanity_failed"] = True
            reason = f"{dataset_name} summary missing after evaluation"
            extras["sanity_reasons"].append(reason)
            print(f"[SANITY FAIL][{dataset_name}] {reason}")
            return HUGE_CHI2, extras

        dataset_sanity = run_dataset_sanity(dataset_name, summary)
        extras["sanity_reasons"].extend(dataset_sanity.reasons)
        if not dataset_sanity.ok:
            extras["sanity_failed"] = True
            print(f"[SANITY FAIL][{dataset_name}] {dataset_sanity.reasons}")
            return HUGE_CHI2, extras

    total_chi2 = 0.0
    for dataset_name in dataset_names:
        chi2_val = dataset_summaries.get(dataset_name, {}).get("chi2")
        if chi2_val is None:
            extras["sanity_failed"] = True
            extras["sanity_reasons"].append(f"{dataset_name} summary lacks chi2")
            return HUGE_CHI2, extras
        total_chi2 += float(chi2_val)

    if total_chi2 < 0.0:
        raise ValueError("Computed a negative χ², which should not happen.")

    return total_chi2, extras


def run_model_sanity(
    model_name: str,
    params: ModelParams,
    model: object,
    *,
    lcdm_model_factory: Callable[..., object] | None = None,
    cmb_solver: Callable[[ModelParams, object], dict] | None = None,
) -> SanityResult:
    result = SanityResult()

    if model_name == "pbuf":
        from cosmos.models.pbuf.sanity import check_pbuf_sanity

        result.merge(check_pbuf_sanity(params, model, lcdm_model_factory=lcdm_model_factory))
    elif model_name == "lcdm":
        from cosmos.models.lcdm.sanity import check_lcdm_sanity

        result.merge(check_lcdm_sanity(params, model, cmb_solver=cmb_solver))
    return result


def run_dataset_sanity(dataset_name: str, summary: DatasetSummary) -> SanityResult:
    result = SanityResult()

    chi2 = summary.get("chi2")
    if chi2 is None or not np.isfinite(chi2):
        result.add_error(f"{dataset_name} sanity: non-finite chi2 ({chi2})")
        return result
    if float(chi2) < 0.0:
        result.add_error(f"{dataset_name} sanity: negative chi2 ({chi2})")
        return result

    if dataset_name == "cmb":
        result.merge(_check_predictions(dataset_name, summary))
        result.merge(check_cmb_sanity(summary))
        return result

    result.merge(_check_predictions(dataset_name, summary))

    if dataset_name in {"sn"}:
        if not _has_finite_vector(summary.get("mu_model")):
            result.add_error("sn sanity: mu_model contains non-finite entries or is missing")
    elif dataset_name == "bao_iso":
        if not _has_finite_vector(summary.get("DV_over_rd_model")):
            result.add_error("bao_iso sanity: DV_over_rd_model invalid")
    elif dataset_name == "bao_aniso":
        if not _has_finite_vector(summary.get("bao_aniso_model")):
            result.add_error("bao_aniso sanity: bao_aniso_model invalid")
    elif dataset_name == "cc":
        if not _has_finite_vector(summary.get("H_model")):
            result.add_error("cc sanity: H_model invalid")
    elif dataset_name == "rsd":
        if not _has_finite_vector(summary.get("fs8_model")):
            result.add_error("rsd sanity: fs8_model invalid")
    elif dataset_name == "wl_s8":
        if not _has_finite_vector(summary.get("S8_model")):
            result.add_error("wl_s8 sanity: S8_model invalid")
    elif dataset_name == "galaxy_pk":
        if not _has_finite_vector(summary.get("galaxy_pk_model_vector")):
            result.add_error("galaxy_pk sanity: galaxy_pk_model_vector invalid")
    elif dataset_name in {"lensing_cross", "lensing_x"}:
        if not _has_finite_vector(summary.get("A_model")):
            result.add_error("lensing_x sanity: A_model invalid")
        weights = summary.get("weights")
        if weights is not None:
            weights_arr = np.asarray(weights, dtype=float)
            if np.any(weights_arr <= 0.0):
                result.add_error("lensing_x sanity: non-positive weights detected")
    return result


def _normalize_params(params: ModelParams) -> ModelParams:
    return {key: float(value) for key, value in params.items()}


def _normalize_dataset_names(raw: Sequence[str] | None) -> tuple[list[str], dict[str, set[str]]]:
    if not raw:
        return ["cmb"], {"cmb": {"cmb"}}

    alias_lookup = {
        "lensing_cross": "lensing_x",
        "lensingx": "lensing_x",
        "lensing-x": "lensing_x",
    }
    normalized: list[str] = []
    aliases: dict[str, set[str]] = {}
    for entry in raw:
        if not entry:
            continue
        cleaned = entry.strip().lower()
        if not cleaned:
            continue
        canonical = alias_lookup.get(cleaned, cleaned)
        aliases.setdefault(canonical, set()).add(cleaned)
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        normalized = ["cmb"]
    aliases.setdefault("cmb", {"cmb"}) if "cmb" in normalized else None
    return normalized, aliases


def _instantiate_model(model_name: str, params: ModelParams) -> object:
    if model_name == "pbuf":
        from cosmos.models.pbuf.microphysics import ensure_thermal_table
        from cosmos.models.pbuf.model import PBUFModel

        return PBUFModel(thermal_table=ensure_thermal_table(), **params)
    if model_name == "lcdm":
        from cosmos.models.lcdm.model import LCDMModel

        return LCDMModel(**params)
    raise ValueError(f"Unsupported model '{model_name}'.")


def _compute_dataset_summaries(
    model: object,
    dataset_names: Sequence[str],
    alias_map: dict[str, set[str]],
) -> tuple[Dict[str, DatasetSummary], list[str]]:
    summaries: Dict[str, DatasetSummary] = {}
    errors: list[str] = []
    for dataset_name in dataset_names:
        try:
            summaries[dataset_name] = _compute_single_dataset_summary(dataset_name, model)
        except Exception as exc:  # pragma: no cover - defensive against fit explosions
            errors.append(f"{dataset_name} fit failed: {exc}")
    for canonical, aliases in alias_map.items():
        summary = summaries.get(canonical)
        if summary is None:
            continue
        for alias in aliases:
            if alias not in summaries:
                summaries[alias] = summary
    return summaries, errors


def _compute_single_dataset_summary(dataset_name: str, model: object) -> DatasetSummary:
    if dataset_name == "cmb":
        return _compute_cmb_summary(model)
    if dataset_name == "sn":
        dataset = get_dataset("sn")
        chi2, extras = run_sn_pantheon_fit(model, dataset)
        return _merge_summary(chi2, extras)
    if dataset_name == "sh0es":
        dataset = get_dataset("sh0es")
        chi2, extras = run_sh0es_prior(model, dataset)
        return _merge_summary(chi2, extras)
    if dataset_name == "bao_iso":
        dataset = get_dataset("bao_iso")
        chi2, extras = run_bao_iso_fit(model, dataset)
        return _merge_summary(chi2, extras)
    if dataset_name == "bao_aniso":
        dataset = get_dataset("bao_aniso")
        chi2, extras = run_bao_aniso_fit(model, dataset)
        return _merge_summary(chi2, extras)
    if dataset_name == "cc":
        dataset = get_dataset("cc")
        chi2, extras = run_cc_fit(model, dataset)
        return _merge_summary(chi2, extras)
    if dataset_name == "rsd":
        dataset = get_dataset("rsd")
        chi2, extras = run_rsd_fit(model, dataset)
        return _merge_summary(chi2, extras)
    if dataset_name == "wl_s8":
        dataset = load_wl_s8_dataset()
        chi2, extras = run_wl_s8_fit(model, dataset)
        return _merge_summary(chi2, extras)
    if dataset_name == "galaxy_pk":
        dataset = get_dataset("galaxy_pk")
        chi2, extras = run_galaxy_pk_fit(model, dataset)
        return _merge_summary(chi2, extras)
    if dataset_name in {"lensing_cross", "lensing_x"}:
        dataset = load_lensing_cross_dataset()
        chi2, extras = run_lensing_cross_fit(model, dataset)
        return _merge_summary(chi2, extras)

    raise ValueError(f"Dataset '{dataset_name}' is not supported for sanity runs.")


def _compute_cmb_summary(model: object) -> DatasetSummary:
    chi2, extras = run_cmb_fit(model)
    output = extras.pop("cmb_output", None)
    summary = _merge_summary(chi2, extras)
    if output is not None:
        summary.update(
            {
                "R": float(output.R),
                "l_A": float(output.l_A),
                "theta_star": float(output.theta_star),
                "z_star": float(output.z_star),
                "D_M": float(output.D_M_Mpc),
                "D_A": float(output.D_A_Mpc),
                "r_s": float(output.r_s_Mpc),
                "Omega_b_h2": float(output.Omega_b_h2),
                "H0": float(getattr(model.params, "H0", getattr(model, "H0", np.nan))),
            }
        )
    return summary


def _merge_summary(chi2: float, extras: Dict[str, Any]) -> DatasetSummary:
    summary: DatasetSummary = {"chi2": float(chi2)}
    summary.update(extras or {})
    return summary


def _make_cached_cmb_solver(summary: DatasetSummary) -> Callable[[ModelParams, object], dict]:
    def solver(params: ModelParams, model: object) -> dict:
        return {
            "Omega_b_h2": summary["Omega_b_h2"],
            "r_s": summary["r_s"],
            "H0": summary["H0"],
        }

    return solver


def _lcdm_model_factory(**kwargs: float) -> object:
    from cosmos.models.lcdm.model import LCDMModel

    return LCDMModel(**kwargs)


def _check_predictions(dataset_name: str, summary: DatasetSummary) -> SanityResult:
    result = SanityResult()
    predictions = summary.get("predictions")
    observed = summary.get("observed")
    residuals = summary.get("residuals")

    if predictions is None or observed is None:
        result.add_error(f"{dataset_name} sanity: missing predictions/observed vectors")
        return result

    pred_arr = np.asarray(predictions, dtype=float)
    obs_arr = np.asarray(observed, dtype=float)
    if pred_arr.shape != obs_arr.shape:
        result.add_error(f"{dataset_name} sanity: predictions/observed shape mismatch {pred_arr.shape} vs {obs_arr.shape}")
    if not np.all(np.isfinite(pred_arr)):
        result.add_error(f"{dataset_name} sanity: predictions contain non-finite values")
    if not np.all(np.isfinite(obs_arr)):
        result.add_error(f"{dataset_name} sanity: observed contains non-finite values")
    if residuals is not None:
        res_arr = np.asarray(residuals, dtype=float)
        if not np.all(np.isfinite(res_arr)):
            result.add_error(f"{dataset_name} sanity: residuals contain non-finite values")
    return result


def _has_finite_vector(value: Any | None) -> bool:
    if value is None:
        return False
    arr = np.asarray(value, dtype=float)
    return np.all(np.isfinite(arr))
