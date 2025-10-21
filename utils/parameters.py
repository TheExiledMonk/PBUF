"""
Utilities for harmonising parameter payloads across proof JSON artefacts.

This module centralises the canonical parameter ordering and provides
helpers to translate between the global calibration block (sourced from
the latest CMB fit) and dataset-specific local adjustments.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, Mapping, MutableMapping

from config.constants import NEFF, TCMB
from utils.io import read_latest_result

# ---------------------------------------------------------------------------
# Canonical parameter templates
# ---------------------------------------------------------------------------
_CANONICAL_ORDER: Dict[str, tuple[str, ...]] = {
    "PBUF": (
        "H0",
        "Om0",
        "Obh2",
        "alpha",
        "Rmax",
        "eps0",
        "n_eps",
        "k_sat",
        "ns",
        "recomb_method",
        "Tcmb",
        "Neff",
        "Ok0",
    ),
    "LCDM": (
        "H0",
        "Om0",
        "Obh2",
        "ns",
        "recomb_method",
        "Tcmb",
        "Neff",
        "Ok0",
    ),
}

_DEFAULT_CALIBRATIONS: Dict[str, Dict[str, float | str]] = {
    "PBUF": {
        "H0": 67.4,
        "Om0": 0.315,
        "Obh2": 0.02237,
        "alpha": 5.0e-4,
        "Rmax": 1.0e9,
        "eps0": 0.7,
        "n_eps": 0.0,
        "k_sat": 1.0,
        "ns": 0.9649,
        "recomb_method": "PLANCK18",
        "Tcmb": TCMB,
        "Neff": NEFF,
        "Ok0": 0.0,
    },
    "LCDM": {
        "H0": 67.4,
        "Om0": 0.315,
        "Obh2": 0.02237,
        "ns": 0.9649,
        "recomb_method": "PLANCK18",
        "Tcmb": TCMB,
        "Neff": NEFF,
        "Ok0": 0.0,
    },
}


def _normalise_model(model: str) -> str:
    key = model.upper()
    if key not in _CANONICAL_ORDER:
        raise ValueError(f"Unsupported model '{model}'. Expected one of {sorted(_CANONICAL_ORDER)}.")
    return key


def extract_global_dict(parameters_field) -> Dict[str, float | str]:
    """
    Convert a ``parameters`` payload into a flat dictionary.

    Supports legacy dict payloads as well as the new
    ``{\"global\": [{\"name\": ..., \"value\": ...}, ...]}`` schema.
    """

    if not parameters_field:
        return {}

    if isinstance(parameters_field, Mapping):
        if "global" in parameters_field:
            global_block = parameters_field["global"]
            if isinstance(global_block, Mapping):
                return dict(global_block)
            if isinstance(global_block, list):
                return {
                    str(entry["name"]): entry.get("value")
                    for entry in global_block
                    if isinstance(entry, Mapping) and "name" in entry
                }
        return dict(parameters_field)  # legacy schema

    if isinstance(parameters_field, list):
        return {
            str(entry["name"]): entry.get("value")
            for entry in parameters_field
            if isinstance(entry, Mapping) and "name" in entry
        }

    return {}


def canonical_parameters(model: str) -> OrderedDict[str, float | str]:
    """
    Return the canonical global parameter block for ``model``.

    Values are pulled from the most recent CMB calibration if available,
    otherwise the baked-in defaults are used.
    """

    model_key = _normalise_model(model)
    values: MutableMapping[str, float | str] = dict(_DEFAULT_CALIBRATIONS[model_key])

    latest = read_latest_result(model=model_key, kind="CMB")
    if latest:
        raw = extract_global_dict(latest.get("parameters"))
        for name in values:
            if name in raw:
                values[name] = raw[name]

    order = _CANONICAL_ORDER[model_key]
    return OrderedDict((name, values[name]) for name in order)


def build_parameter_payload(
    model: str,
    fitted: Mapping[str, float | str] | None = None,
    *,
    free_names: Iterable[str] = (),
    extra_locals: Mapping[str, float | str] | None = None,
    canonical: Mapping[str, float | str] | None = None,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-12,
) -> Dict[str, list[Dict[str, float | str]]]:
    """
    Construct the standardised parameter payload for proof JSON outputs.

    Parameters
    ----------
    model : str
        Cosmological model identifier (``'PBUF'`` or ``'LCDM'``).
    fitted : mapping, optional
        Dictionary of the parameters used in the dataset-specific fit.
    free_names : iterable of str, optional
        Explicit list of parameters that were varied locally; they are
        always tagged with ``scope='local'`` irrespective of numerical
        equality with the global calibration.
    extra_locals : mapping, optional
        Additional parameters to inject into the local block (e.g.
        sigma8 for growth-rate fits).
    canonical : mapping, optional
        Precomputed canonical block to avoid recomputation.
    rel_tol, abs_tol : float
        Numerical tolerances used to decide whether a fitted parameter
        deviates from the global calibration.
    """

    model_key = _normalise_model(model)
    canonical_block = OrderedDict(canonical) if canonical is not None else canonical_parameters(model_key)

    global_entries = [
        {"name": name, "value": canonical_block[name], "scope": "global"}
        for name in canonical_block
    ]

    local_map: OrderedDict[str, float | str] = OrderedDict()
    fitted = fitted or {}
    free_set = {str(name) for name in free_names}

    for name, value in fitted.items():
        key = str(name)
        if key not in canonical_block:
            local_map[key] = value
            continue
        canonical_value = canonical_block[key]
        if key in free_set:
            local_map[key] = value
            continue
        try:
            numeric_value = float(value)  # type: ignore[arg-type]
            numeric_canonical = float(canonical_value)  # type: ignore[arg-type]
            if not math.isclose(numeric_value, numeric_canonical, rel_tol=rel_tol, abs_tol=abs_tol):
                local_map[key] = value
        except (TypeError, ValueError):
            if value != canonical_value:
                local_map[key] = value

    if extra_locals:
        for name, value in extra_locals.items():
            local_map[str(name)] = value

    local_entries = [
        {"name": name, "value": value, "scope": "local"}
        for name, value in local_map.items()
    ]

    return {"global": global_entries, "local": local_entries}


def flatten_payload(payload: Mapping[str, list[Mapping[str, float | str]]]) -> Dict[str, float | str]:
    """
    Convert a canonical payload back into a simple dictionary combining
    global and local entries. Local values override global ones.
    """

    global_values = {}
    for entry in payload.get("global", []):
        if isinstance(entry, Mapping) and "name" in entry:
            global_values[str(entry["name"])] = entry.get("value")

    for entry in payload.get("local", []):
        if isinstance(entry, Mapping) and "name" in entry:
            global_values[str(entry["name"])] = entry.get("value")

    return global_values

