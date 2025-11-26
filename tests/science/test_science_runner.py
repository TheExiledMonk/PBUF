"""Unit tests for the configurable science runner."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from science_runner import _interactive_confirm

import numpy as np
import pytest

from cosmos.interfaces import CMBOutput
from cosmos.science_runner.config import ScienceRunConfig
from cosmos.science_runner.runner import ScienceRunner


class FakeModel:
    def __init__(self, **params: Any) -> None:
        self._params = {key: float(value) for key, value in params.items()}
        self._params.setdefault("H0", 70.0)
        self._params.setdefault("Omega_m0", 0.3)

    @property
    def parameters(self) -> dict[str, float]:
        return dict(self._params)

    def cmb(self, data: Any = None) -> CMBOutput:
        return CMBOutput(
            R=1.0,
            l_A=1.0,
            Omega_b_h2=0.022,
            theta_star=0.01,
            z_star=1100.0,
            D_M_Mpc=14000.0,
            D_A_Mpc=7000.0,
            r_s_Mpc=150.0,
        )

    def omega_m0(self) -> float:
        return self._params.get("Omega_m0", 0.3)

    def sigma8(self) -> float:
        return self._params.get("sigma8", 0.8)

    def distance_modulus(self, z: Any) -> Any:
        arr = np.asarray(z, dtype=float)
        return np.full_like(arr, fill_value=self._params["H0"], dtype=float)

    def DM(self, z: Any) -> Any:
        return np.full_like(np.asarray(z, dtype=float), 3000.0, dtype=float)

    def DA(self, z: Any) -> Any:
        return np.full_like(np.asarray(z, dtype=float), 1500.0, dtype=float)

    def DH(self, z: Any) -> Any:
        return np.full_like(np.asarray(z, dtype=float), 100.0, dtype=float)

    def Hubble(self, z: Any) -> Any:
        return np.full_like(np.asarray(z, dtype=float), self._params["H0"], dtype=float)

    def fs8(self, z: Any) -> Any:
        return np.full_like(np.asarray(z, dtype=float), 0.7, dtype=float)

    def S8(self, gamma: float = 0.5) -> float:
        return self.sigma8() * (self.omega_m0() / 0.3) ** gamma

    def sound_horizon(self) -> float:
        return 147.0

    def is_valid(self) -> bool:
        return True


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    def optimise(
        self,
        *,
        objective: Any,
        bounds: dict[str, tuple[float, float]],
        fixed_parameters: dict[str, float],
        initial_parameters: dict[str, float],
        phase6a: Any,
    ) -> dict[str, Any]:
        best: dict[str, float] = {}
        for key, (lower, upper) in bounds.items():
            best[key] = initial_parameters.get(key, (lower + upper) / 2.0)
        best = {key: float(value) for key, value in best.items()}
        self.calls.append(best)
        return {
            "engine": "fake",
            "best_params": best,
            "best_chi2": objective(best),
            "trace": [best],
        }


class FakePlotter:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def generate(self, *, predictions: dict[str, Any], model_dir: Path) -> None:
        self.calls.append(model_dir)


class FakeReporter:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def generate(
        self,
        *,
        run_dir: Path,
        model_dir: Path,
        model_name: str,
        run_meta: dict[str, Any],
        best_params: dict[str, float],
        best_chi2: float,
        chi2_breakdown: dict[str, float],
        fit_outputs: dict[str, Any],
        predictions: dict[str, Any],
        report_formats: Sequence[str],
    ) -> None:
        self.calls.append(model_dir)


@pytest.fixture(autouse=True)
def patch_science_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_make_model_factory(model_name: str, *, datasets: Sequence[str] | None = None) -> Any:
        return lambda params: FakeModel(**params)

    def fake_engine_factory(name: str, settings: dict[str, Any]) -> FakeEngine:
        return FakeEngine()

    def fake_joint(_factory: Any, _path: Path) -> Any:
        def chi2(params: dict[str, float]) -> float:
            return sum(float(value) for value in params.values())

        return chi2

    def fake_fit(model: FakeModel) -> tuple[float, dict[str, float]]:
        total = sum(model.parameters.values())
        return float(total), {"summary": float(total)}

    monkeypatch.setattr("cosmos.science_runner.factories.make_model_factory", fake_make_model_factory)
    monkeypatch.setattr("cosmos.science_runner.factories.make_engine", fake_engine_factory)
    monkeypatch.setattr("cosmos.science_runner.runner.make_model_factory", fake_make_model_factory)
    monkeypatch.setattr("cosmos.science_runner.runner.make_engine", fake_engine_factory)
    monkeypatch.setattr(
        "cosmos.science_runner.runner.build_joint_chi2_evaluator", lambda factory, path: fake_joint(factory, path)
    )
    monkeypatch.setattr("cosmos.science_runner.runner.FIT_REGISTRY", {"cmb": fake_fit})


def _write_config(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _run_config(tmp_path: Path, config_data: dict[str, Any]) -> Path:
    config_file = tmp_path / "science_config.json"
    _write_config(config_file, config_data)
    config = ScienceRunConfig.from_path(config_file)
    plotter = FakePlotter()
    reporter = FakeReporter()
    runner = ScienceRunner(config, plotter=plotter, reporter=reporter)
    runner.execute()
    return config.output.base_dir


def _find_run_directory(base_dir: Path) -> Path:
    runs = sorted(p for p in base_dir.iterdir() if p.is_dir())
    assert runs, "No science run folders were created"
    return runs[-1]


def test_scout_mode_creates_outputs(tmp_path: Path) -> None:
    base_dir = tmp_path / "science_runs"
    config = {
        "run_name": "scout_test",
        "models": ["lcdm"],
        "mode": "scout",
        "fits_override": ["cmb"],
        "parameter_bounds_inline": {
            "H0": [65, 75],
            "Omega_m0": [0.25, 0.35],
            "Omega_k0": [-0.05, 0.05],
        },
        "fixed_parameters": {"Omega_r0": 9e-5},
        "initial_parameters": {"H0": 70, "Omega_m0": 0.3, "Omega_k0": 0.0},
        "output": {
            "base_dir": str(base_dir),
            "generate_plots": False,
            "generate_reports": False,
            "save_space": False,
        },
    }
    output_base = _run_config(tmp_path, config)
    run_dir = _find_run_directory(output_base)
    assert (run_dir / "config_used.json").exists()
    assert (run_dir / "run_meta.json").exists()
    assert (run_dir / "history_entry.json").exists()
    assert (run_dir / "lcdm" / "best_fit.json").exists()
    assert (run_dir / "lcdm" / "chi2_breakdown.json").exists()
    assert (run_dir / "joint_config_used.json").exists()
    assert (run_dir / "datasets_used.json").exists()
    assert (run_dir / "engine_settings.json").exists()
    assert (run_dir / "chi2_history.json").exists()
    chi2_history = json.loads((run_dir / "chi2_history.json").read_text(encoding="utf-8"))
    assert chi2_history and chi2_history[-1]["fit"] == "cmb"
    model_dir = run_dir / "lcdm"
    assert (model_dir / "fits" / "cmb.json").exists()
    history = json.loads((output_base / "history.json").read_text(encoding="utf-8"))
    assert history and history[-1]["model"] == "lcdm"


def test_fit_mode_generates_reports_and_respects_save_space(tmp_path: Path) -> None:
    base_dir = tmp_path / "science_runs_fit"
    config = {
        "run_name": "fit_test",
        "models": ["lcdm"],
        "mode": "fit",
        "fits_override": ["cmb"],
        "engine": "basin",
        "parameter_bounds_inline": {
            "H0": [66, 74],
            "Omega_m0": [0.26, 0.34],
            "Omega_k0": [-0.04, 0.04],
        },
        "initial_parameters": {"H0": 68, "Omega_m0": 0.3},
        "output": {
            "base_dir": str(base_dir),
            "generate_plots": True,
            "generate_reports": True,
            "save_space": True,
        },
    }
    config_file = tmp_path / "science_fit.json"
    _write_config(config_file, config)
    science_config = ScienceRunConfig.from_path(config_file)
    plotter = FakePlotter()
    reporter = FakeReporter()
    runner = ScienceRunner(science_config, plotter=plotter, reporter=reporter)
    runner.execute()
    run_dir = _find_run_directory(science_config.output.base_dir)
    model_dir = run_dir / "lcdm"
    assert not (model_dir / "parameters_trace.json").exists()
    assert plotter.calls
    assert reporter.calls
    engine_trace = model_dir / "engine_trace.json"
    assert engine_trace.exists()
    trace_payload = json.loads(engine_trace.read_text(encoding="utf-8"))
    assert "trace_meta" in trace_payload
    assert "trace" not in trace_payload


def test_batch_mode_supports_multiple_models(tmp_path: Path) -> None:
    base_dir = tmp_path / "science_runs_multi"
    config = {
        "run_name": "multi_model",
        "models": ["lcdm", "pbuf"],
        "mode": "scout",
        "fits_override": ["cmb"],
        "parameter_bounds_inline": {
            "H0": [65, 75],
            "Omega_m0": [0.25, 0.35],
            "Omega_k0": [-0.05, 0.05],
        },
        "initial_parameters": {"H0": 70, "Omega_m0": 0.3},
        "output": {"base_dir": str(base_dir), "generate_plots": False, "generate_reports": False},
    }
    output_base = _run_config(tmp_path, config)
    run_dir = _find_run_directory(output_base)
    assert (run_dir / "lcdm").is_dir()
    assert (run_dir / "pbuf").is_dir()


def test_interactive_bypasses_non_tty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "run_name": "interactive_test",
        "models": ["lcdm"],
        "mode": "fit",
        "fits_override": ["cmb"],
        "parameter_bounds_inline": {"H0": [65, 75]},
        "output": {"base_dir": str(tmp_path / "science_runs"), "generate_plots": False, "generate_reports": False},
    }
    config_file = tmp_path / "interactive.json"
    _write_config(config_file, config)
    science_config = ScienceRunConfig.from_path(config_file)
    fake_stdin = io.StringIO("")
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    assert _interactive_confirm(science_config)
