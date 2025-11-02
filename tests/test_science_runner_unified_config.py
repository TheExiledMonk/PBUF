from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cosmos.optim.coord_optimizer.basin_walker import CoordinateBasinWalker
from cosmos.optim.science_runner import ScienceRunner, load_config


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "science.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_config_full_structure(tmp_path: Path) -> None:
    config_payload = {
        "run_id": "v10_full_run",
        "models": ["lcdm", "pbuf"],
        "scenarios": [
            {"id": "geom", "datasets": ["cmb", "bao_aniso", "cc"]},
        ],
        "walker": {"converge": True, "max_rebalances": 2, "reseed_on_plateau": True},
        "priors": {"pbuf": {"k_sat": {"type": "uniform", "min": 0.95, "max": 1.0}}},
        "targets": {"cmb": [1.0, 0.5]},
    }
    path = _write_config(tmp_path, config_payload)

    cfg = load_config(path)

    assert cfg["run_id"] == "v10_full_run"
    assert cfg["walker"]["converge"] is True
    assert cfg["walker"]["max_rebalances"] == 2
    assert "priors" in cfg and "pbuf" in cfg["priors"]
    assert cfg["targets"]["cmb"] == [1.0, 0.5]


def test_load_config_defaults(tmp_path: Path) -> None:
    payload = {
        "run_id": "mini",
        "models": ["lcdm"],
        "scenarios": [{"id": "s", "datasets": ["cmb"]}],
    }
    path = _write_config(tmp_path, payload)

    cfg = load_config(path)

    assert cfg["walker"]["converge"] is False
    assert cfg["checkpointing"]["enabled"] is False
    assert cfg["reporting"]["record_dof"] is False


class RecordingExecutor:
    def __init__(self, runtime: float = 1.0) -> None:
        self.calls: list[tuple[str, str]] = []
        self.runtime = runtime

    def __call__(self, scenario: dict, model: str, config: dict) -> dict:
        self.calls.append((scenario["id"], model))
        breakdown = {dataset: 10.0 for dataset in scenario.get("datasets", [])}
        return {
            "scenario_id": scenario["id"],
            "model": model,
            "fiducial_breakdown": breakdown,
            "metadata": {"runtime_seconds": self.runtime},
        }


def _base_config(tmp_path: Path) -> dict:
    return {
        "run_id": "unittest",
        "models": ["lcdm", "pbuf"],
        "scenarios": [
            {"id": "geom", "datasets": ["cmb", "bao_aniso"]},
        ],
        "checkpointing": {"enabled": True, "resume": True},
        "reporting": {"store_per_dataset_partitions": True},
        "output_root": str(tmp_path / "runs"),
    }


def test_science_runner_checkpoint_resume(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    executor = RecordingExecutor(runtime=2.5)
    runner = ScienceRunner(config, executor=executor)

    summary = runner.run(fresh=True)
    assert executor.calls == [("geom", "lcdm"), ("geom", "pbuf")]
    assert summary["runtime_seconds"] > 0
    assert runner.run_dir is not None
    checkpoints = list((runner.run_dir / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 2

    # Resume with a fresh executor; no new calls should be made
    resume_executor = RecordingExecutor(runtime=1.0)
    resume_runner = ScienceRunner(config, executor=resume_executor)
    resume_runner.run(resume_dir=runner.run_dir)
    assert resume_executor.calls == []


def test_science_runner_reporting_partitions(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    executor = RecordingExecutor(runtime=3.0)
    runner = ScienceRunner(config, executor=executor)

    summary = runner.run(fresh=True)

    assert "per_dataset" in summary
    cmb_entries = summary["per_dataset"]["cmb"]
    assert cmb_entries[0]["chi2"] == pytest.approx(10.0)
    assert cmb_entries[0]["runtime_seconds"] == pytest.approx(3.0)


def test_walker_forces_convergence_when_requested() -> None:
    walker = CoordinateBasinWalker(
        model_type="lcdm",
        datasets=["cmb"],
        verbose=False,
        walker_settings={"converge": True},
    )

    calls = {"count": 0}

    def fake_run_with_convergence(self) -> dict:
        calls["count"] += 1
        return {"convergence": {"converged": True}}

    walker.run_with_convergence = fake_run_with_convergence.__get__(walker, CoordinateBasinWalker)

    result = walker.run()
    assert calls["count"] == 1
    assert result["convergence"]["converged"] is True


def test_priors_reject_parameter_sets() -> None:
    walker = CoordinateBasinWalker(
        model_type="lcdm",
        datasets=["cmb"],
        priors={"H0": {"type": "uniform", "min": 60.0, "max": 70.0}},
    )

    evaluation = walker._evaluate({"H0": 75.0, "Om0": 0.3})
    assert evaluation["status"] == "invalid"
    diagnostics = evaluation.get("diagnostics", {})
    assert diagnostics["priors_rejected"]
