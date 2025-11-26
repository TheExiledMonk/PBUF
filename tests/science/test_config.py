"""Unit tests for the science run configuration loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos.science_runner.config import ScienceRunConfig


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_joint_config_merging(tmp_path: Path) -> None:
    joint_payload = {"fits": ["cmb", "sn"], "fit_weights": {"cmb": 1.0, "sn": 2.0}}
    joint_path = tmp_path / "joint.json"
    joint_path.write_text(json.dumps(joint_payload), encoding="utf-8")

    config_payload = {
        "run_name": "joint_merge",
        "models": ["lcdm"],
        "mode": "fit",
        "fits_config": str(joint_path),
        "parameter_bounds_inline": {"H0": [65, 75]},
        "output": {"base_dir": str(tmp_path / "science_runs")},
    }
    config_path = tmp_path / "science.json"
    _write_config(config_path, config_payload)

    config = ScienceRunConfig.from_path(config_path)
    assert config.fits_list == ["cmb", "sn"]
    assert config.fit_weights == {"cmb": 1.0, "sn": 2.0}
    assert config.joint_config_payload["fit_weights"] == {"cmb": 1.0, "sn": 2.0}

    config.set_fits(["sn"])
    assert config.fits_list == ["sn"]
    assert config.joint_config_payload["fits"] == ["sn"]

    with pytest.raises(ValueError):
        config.set_fits([])


def test_parameter_bounds_per_model(tmp_path: Path) -> None:
    payload = {
        "run_name": "bounds_test",
        "models": ["lcdm", "pbuf"],
        "mode": "fit",
        "fits_override": ["cmb"],
        "parameter_bounds": {
            "lcdm": {
                "H0": [65, 75],
                "Omega_m0": [0.25, 0.35],
            },
            "pbuf": {
                "H0": [60, 80],
                "Rmax": [5e6, 1e8],
            },
        },
        "output": {"base_dir": str(tmp_path / "science_runs")},
    }
    config_path = tmp_path / "bounds.json"
    _write_config(config_path, payload)

    config = ScienceRunConfig.from_path(config_path)
    lcdm_bounds = config.parameter_bounds_for_model("LCDM")
    pbuf_bounds = config.parameter_bounds_for_model("pbuf")

    assert lcdm_bounds["H0"] == (65.0, 75.0)
    assert lcdm_bounds["Omega_m0"] == (0.25, 0.35)
    assert "Rmax" not in lcdm_bounds
    assert pbuf_bounds["H0"] == (60.0, 80.0)
    assert pbuf_bounds["Rmax"] == (5000000.0, 100000000.0)

    payload_data = config.parameter_bounds_payload
    assert payload_data["global"] == {}
    assert payload_data["models"]["lcdm"]["H0"] == [65.0, 75.0]
    assert payload_data["models"]["pbuf"]["Rmax"] == [5000000.0, 100000000.0]
