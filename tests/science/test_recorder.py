"""Unit tests for the science run recorder."""

from __future__ import annotations

import json
from pathlib import Path

from cosmos.science_runner.recorder import RunRecorder


def test_run_recorder_saves_fit_and_trace(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path / "runs")
    run_dir = recorder.prepare_run_directory("recorder_test", "0001")
    recorder.write_json(run_dir, "sample.json", {"ok": True})
    assert (run_dir / "sample.json").exists()

    model_dir = run_dir / "lcdm"
    recorder.save_fit_output(model_dir, "cmb", 10.5, {"residual": [0.1, 0.2]})
    fit_payload = json.loads((model_dir / "fits" / "cmb.json").read_text(encoding="utf-8"))
    assert fit_payload["chi2"] == 10.5
    assert fit_payload["extras"]["residual"] == [0.1, 0.2]

    recorder.save_engine_trace(
        model_dir,
        engine_name="basin",
        trace=[{"chi2": 5.0}],
        trace_meta={"iterations": 1, "final_step": {"chi2": 5.0}, "converged": True},
        save_space=True,
    )
    trace_payload = json.loads((model_dir / "engine_trace.json").read_text(encoding="utf-8"))
    assert trace_payload["trace_meta"]["iterations"] == 1
    assert "trace" not in trace_payload

    recorder.save_engine_trace(
        model_dir,
        engine_name="basin",
        trace=[{"chi2": 5.0}],
        trace_meta={"iterations": 1, "final_step": {"chi2": 5.0}, "converged": True},
        save_space=False,
    )
    trace_payload = json.loads((model_dir / "engine_trace.json").read_text(encoding="utf-8"))
    assert trace_payload["trace"]

    recorder.record_model_failure(model_dir, "fit crashed")
    failure_payload = json.loads((model_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure_payload["failure_reason"] == "fit crashed"

    recorder.write_text(run_dir, "notes.txt", "hello")
    assert (run_dir / "notes.txt").read_text(encoding="utf-8") == "hello"

    recorder.save_metadata(run_dir, {"run_meta": "ok"})
    assert (run_dir / "metadata.json").exists()
