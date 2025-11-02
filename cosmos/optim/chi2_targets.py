from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class Chi2TargetRule:
    target: float
    tolerance: float
    weight: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "Chi2TargetRule":
        try:
            target = float(payload["target"])
            tolerance = float(payload.get("tolerance", 1.0))
            weight = float(payload.get("weight", 1.0))
        except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid χ² target rule: {payload!r}") from exc
        if tolerance <= 0.0:
            tolerance = 1.0
        if weight <= 0.0:
            weight = 1.0
        return cls(target=target, tolerance=tolerance, weight=weight)


class Chi2TargetRegistry:
    """
    Holds χ² expectations for a specific model/dataset bundle and produces
    objective scores for optimiser evaluations that prioritise hitting the target
    rather than blindly minimising χ².
    """

    def __init__(self, rules: Mapping[str, Chi2TargetRule], *, delta: float = 1.0) -> None:
        self._rules: Dict[str, Chi2TargetRule] = dict(rules)
        self.delta = float(delta) if delta > 0.0 else 1.0

    def is_empty(self) -> bool:
        return not self._rules

    def score(self, evaluation: Mapping[str, object], fallback: Optional[float]) -> Optional[float]:
        """
        Compute a dimensionless distance-to-target score from an optimiser evaluation.

        When no matching datasets are present in the evaluation breakdown, the fallback
        χ² value is returned so traditional minimisation semantics continue to work.
        """
        breakdown = evaluation.get("chi2_breakdown")
        if not isinstance(breakdown, Mapping):
            return fallback

        total_weight = 0.0
        score_accumulator = 0.0
        for dataset, rule in self._rules.items():
            value_obj = breakdown.get(dataset)
            if value_obj is None:
                continue
            try:
                actual = float(value_obj)
            except (TypeError, ValueError):
                continue
            total_weight += rule.weight
            distance = abs(actual - rule.target) / rule.tolerance
            score_accumulator += rule.weight * distance

        if total_weight <= 0.0:
            return fallback
        return score_accumulator / total_weight

    def describe(self) -> Dict[str, Dict[str, float]]:
        return {
            dataset: {
                "target": rule.target,
                "tolerance": rule.tolerance,
                "weight": rule.weight,
            }
            for dataset, rule in self._rules.items()
        }


def load_chi2_targets(path: Path, model: str) -> Optional[Chi2TargetRegistry]:
    """
    Load χ² target configuration from JSON file for a specific model.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid χ² target configuration: {path}") from exc

    model_payload = payload.get(model)
    if not isinstance(model_payload, Mapping):
        return None

    rules: Dict[str, Chi2TargetRule] = {}
    for dataset, rule_payload in model_payload.items():
        if isinstance(rule_payload, Mapping):
            try:
                rules[dataset] = Chi2TargetRule.from_mapping(rule_payload)
            except ValueError:
                continue
    if not rules:
        return None

    delta = payload.get("delta")
    return Chi2TargetRegistry(rules, delta=float(delta) if isinstance(delta, (int, float)) else 1.0)
