"""Shared helpers for applying parameter-space priors."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional


def prior_violation_reason(
    params: Mapping[str, float],
    priors: Optional[Mapping[str, Mapping[str, Any]]],
) -> Optional[str]:
    """Return a human-readable reason when *params* violate *priors*."""

    if not priors:
        return None
    for param, spec in priors.items():
        if param not in params:
            continue
        try:
            value = float(params[param])
        except (TypeError, ValueError):
            return f"{param} not numeric"

        minimum = spec.get("min")
        maximum = spec.get("max")
        prior_type = str(spec.get("type", "uniform")).lower()

        if prior_type in {"log_uniform", "loguniform"} and value <= 0.0:
            return f"{param}={value:.6g} non-positive for log-uniform prior"

        if minimum is not None:
            try:
                min_val = float(minimum)
            except (TypeError, ValueError):
                min_val = None
            if min_val is not None and value < min_val:
                return f"{param}={value:.6g} below prior minimum {min_val:.6g}"

        if maximum is not None:
            try:
                max_val = float(maximum)
            except (TypeError, ValueError):
                max_val = None
            if max_val is not None and value > max_val:
                return f"{param}={value:.6g} above prior maximum {max_val:.6g}"

    return None


def record_prior_violation(
    diagnostics: MutableMapping[str, Any],
    reason: str,
) -> None:
    """Attach the rejection *reason* to `diagnostics['priors_rejected']`."""

    if "priors_rejected" in diagnostics:
        entry = diagnostics["priors_rejected"]
        if isinstance(entry, list):
            if reason not in entry:
                entry.append(reason)
            return
        if isinstance(entry, str):
            if entry != reason:
                diagnostics["priors_rejected"] = [entry, reason]
            return
    diagnostics["priors_rejected"] = [reason]


__all__ = ["prior_violation_reason", "record_prior_violation"]
