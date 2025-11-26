"""Sample PBUF candidates to catalog which Phase-6a checks fail."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cosmos2.config import load_bounds_for_model
from cosmos2.models.model_factory import create_model
from cosmos2.pbuf.microphysics import ensure_thermal_table


def _sample_candidate(rng: random.Random, bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    params: dict[str, float] = {}
    for name, interval in bounds.items():
        lower, upper = float(interval[0]), float(interval[1])
        if lower > upper:
            raise RuntimeError(f"Inconsistent bound for {name}: {lower} > {upper}")
        params[name] = lower if upper == lower else rng.uniform(lower, upper)
    return params


def main() -> None:
    raw_bounds = load_bounds_for_model("pbuf", ["cmb"])
    table = ensure_thermal_table()
    metadata = dict(getattr(table, "metadata", {}) or {})
    lut = {
        "T": np.asarray(table.T, dtype=float),
        "eps": np.asarray(table.eps, dtype=float),
        "alpha": np.asarray(table.alpha, dtype=float),
        "dln_eps": np.asarray(table.dln_eps, dtype=float),
        "dln_alpha": np.asarray(table.dln_alpha, dtype=float),
        "g_star": np.asarray(table.g_star, dtype=float),
        "g_starS": np.asarray(table.g_starS, dtype=float),
        "a": np.asarray(table.a, dtype=float),
        "metadata": metadata,
    }
    rng = random.Random(0xC0FFEE)

    reasons = Counter()
    params_by_reason: dict[str, dict[str, float]] = {}
    samples = 400
    failures = 0

    for index in range(samples):
        params = _sample_candidate(rng, raw_bounds)
        model = create_model("pbuf", lut=lut, **params)
        ok = model.is_valid()
        if ok:
            continue
        failures += 1
        reason_list: list[str] = []
        if not getattr(model, "_phase_ok", True):
            reason_list.append("phase6a/phase7a_kernel")
        if getattr(model, "_phase7a_external_ok", True) is False:
            reason_list.append("phase7a_external")
        if not reason_list:
            reason_list.append("unknown_failure")
        for reason in reason_list:
            reasons[reason] += 1
            params_by_reason.setdefault(reason, dict(params))
        if len(reasons) >= 3:
            break

    print(f"Sampled {samples} random candidates, {failures} failed Phase-6a.")
    for reason, count in reasons.most_common():
        print(f"{count:3d}: {reason}")
        sample_params = params_by_reason.get(reason)
        if sample_params:
            sorted_params = ", ".join(f"{k}={v!r}" for k, v in sorted(sample_params.items()))
            print(f"     example params: {sorted_params}")

    if reasons:
        print("\nTop reasons:")
        for reason, count in reasons.most_common(2):
            params_snapshot = params_by_reason.get(reason)
            print(f"- {reason} ({count} hits)")
            if params_snapshot:
                print(f"  params: {params_snapshot}")


if __name__ == "__main__":
    main()
