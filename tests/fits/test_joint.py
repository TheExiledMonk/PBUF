"""Unit tests for the joint fit orchestrator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict

import pytest

from cosmos.fits.joint import build_joint_chi2_evaluator
from cosmos.fits.registry import FIT_REGISTRY

class _FakeModel:
    def __init__(self, *, valid: bool):
        self._valid = valid

    def is_valid(self) -> bool:
        return self._valid


def _make_factory(valid: bool) -> Callable[[Dict[str, float]], _FakeModel]:
    def factory(_: Dict[str, float]) -> _FakeModel:
        return _FakeModel(valid=valid)

    return factory


def _write_config(tmp_path: Path, fits: list[str], weights: Dict[str, float] | None = None) -> Path:
    payload: Dict[str, object] = {"fits": fits}
    if weights:
        payload["fit_weights"] = weights
    path = tmp_path / "joint_config.json"
    path.write_text(json.dumps(payload))
    return path


def _register_dummy_fits(monkeypatch) -> None:
    def fit_a(_: _FakeModel) -> tuple[float, dict]:
        return 5.0, {}

    def fit_b(_: _FakeModel) -> tuple[float, dict]:
        return 2.0, {}

    monkeypatch.setitem(FIT_REGISTRY, "fit_a", fit_a)
    monkeypatch.setitem(FIT_REGISTRY, "fit_b", fit_b)


def test_joint_chi2_sums_weighted_fits(tmp_path: Path, monkeypatch) -> None:
    _register_dummy_fits(monkeypatch)
    config_path = _write_config(tmp_path, fits=["fit_a", "fit_b"], weights={"fit_a": 1.0, "fit_b": 2.0})
    evaluator = build_joint_chi2_evaluator(_make_factory(valid=True), config_path)

    chi2 = evaluator({})
    assert chi2 == pytest.approx(5.0 + 2.0 * 2.0)


def test_joint_chi2_respects_fit_subset(tmp_path: Path, monkeypatch) -> None:
    _register_dummy_fits(monkeypatch)
    config_path = _write_config(tmp_path, fits=["fit_a"])
    evaluator = build_joint_chi2_evaluator(_make_factory(valid=True), config_path)

    chi2 = evaluator({})
    assert chi2 == pytest.approx(5.0)


def test_joint_chi2_returns_inf_for_invalid_model(tmp_path: Path, monkeypatch) -> None:
    _register_dummy_fits(monkeypatch)
    config_path = _write_config(tmp_path, fits=["fit_a", "fit_b"])
    evaluator = build_joint_chi2_evaluator(_make_factory(valid=False), config_path)

    assert math.isinf(evaluator({}))


def test_joint_chi2_returns_inf_for_non_finite_fit(tmp_path: Path, monkeypatch) -> None:
    def bad_fit(_: _FakeModel) -> tuple[float, dict]:
        return math.nan, {}

    monkeypatch.setitem(FIT_REGISTRY, "fit_nan", bad_fit)
    config_path = _write_config(tmp_path, fits=["fit_nan"])
    evaluator = build_joint_chi2_evaluator(_make_factory(valid=True), config_path)

    assert math.isinf(evaluator({}))
