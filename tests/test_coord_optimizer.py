import json
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "tqdm" not in sys.modules:
    dummy_tqdm_module = types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable)
    sys.modules["tqdm"] = dummy_tqdm_module

import pytest

from cosmos.optim.coord_optimizer import CoordinateBasinWalker, DEFAULT_PBUF_REFERENCE
from cosmos.optim.coord_optimizer.observers import RecordingObserver
from reports.basin_plotter import generate_basin_plots


@pytest.fixture
def reference():
    return dict(DEFAULT_PBUF_REFERENCE)


def test_make_scan_range_h0(reference):
    walker = CoordinateBasinWalker(
        model_type="pbuf",
        datasets=["cmb"],
        reference_params=reference,
        second_pass_params=(),
        param_order=("H0",),
        max_workers=1,
    )

    coarse = walker._make_scan_range("H0", reference, pass_id=1)
    assert coarse[0] == pytest.approx(66.0)
    assert coarse[-1] == pytest.approx(74.0)
    assert len(coarse) > 1

    tight_params = dict(reference)
    tight_params["H0"] = 67.3
    refine = walker._make_scan_range("H0", tight_params, pass_id=2)
    assert refine[0] == pytest.approx(66.8, rel=1e-3, abs=1e-3)
    assert refine[-1] == pytest.approx(67.8, rel=1e-3, abs=1e-3)
    assert all(refine[i] < refine[i + 1] for i in range(len(refine) - 1))


def test_lcdm_defaults():
    walker = CoordinateBasinWalker(
        model_type="lcdm",
        datasets=["cmb"],
        second_pass_params=(),
        max_workers=1,
    )
    assert walker.param_order == ("H0", "Om0")
    assert "alpha" not in walker.reference_params
    assert "k_sat" not in walker.reference_params


class StubWalker(CoordinateBasinWalker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._evaluations = {}

    def _evaluate(self, params):
        key = tuple(sorted(params.items()))
        if key not in self._evaluations:
            value = float(params["H0"])
            delta = value - 67.3
            chi2 = 1500.0 + delta * delta * 10.0
            passes_phase6a = abs(delta) <= 0.2
            self._evaluations[key] = {
                "status": "valid",
                "chi2_total": chi2,
                "passes_phase6a": passes_phase6a,
                "chi2_breakdown": {"cmb": chi2},
            }
        return self._evaluations[key]


class EdgeWalker(CoordinateBasinWalker):
    def _evaluate(self, params):
        h0 = float(params["H0"])
        delta = h0 - 66.8
        chi2 = 1800.0 + 50.0 * delta * delta
        return {
            "status": "valid",
            "chi2_total": chi2,
            "passes_phase6a": True,
            "chi2_breakdown": {"cmb": chi2},
        }


def test_scan_axis_selects_best(reference):
    scan_presets = {
        "H0": {
            "coarse": {"type": "list", "values": [67.0, 67.3, 67.6]},
            "refine": {"type": "list", "values": [67.2, 67.3, 67.4]},
        }
    }

    walker = StubWalker(
        model_type="pbuf",
        datasets=["cmb"],
        reference_params=reference,
        scan_presets=scan_presets,
        param_order=("H0",),
        second_pass_params=(),
        delta_chi2=0.1,
        max_workers=1,
    )

    params = dict(reference)
    summary = walker._scan_axis("H0", params, [67.0, 67.3, 67.6], pass_id=1)

    assert summary["best"] == pytest.approx(67.3)
    assert summary["chi2_min"] == pytest.approx(1500.0)
    assert summary["left_edge"] == pytest.approx(67.3)
    assert summary["right_edge"] == pytest.approx(67.3)
    assert summary["curve"][1]["valid"] is True
    assert summary["curve"][0]["valid"] is False


def test_run_generates_payload(reference, tmp_path: Path):
    scan_presets = {
        "H0": {
            "coarse": {"type": "list", "values": [67.0, 67.3, 67.6]},
            "refine": {"type": "list", "values": [67.2, 67.3, 67.4]},
        }
    }

    walker = StubWalker(
        model_type="pbuf",
        datasets=["cmb"],
        reference_params=reference,
        scan_presets=scan_presets,
        param_order=("H0",),
        second_pass_params=(),
        delta_chi2=5.0,
        max_workers=1,
    )

    result = walker.run()

    assert result["version"] == "coord-opt-v2"
    assert result["fiducial_params"]["H0"] == pytest.approx(67.3)
    assert result["datasets_used"] == ["cmb"]
    assert math.isclose(result["fiducial_chi2"], 1500.0)

    axis_scans = result["axis_scans"]
    assert len(axis_scans) == 1
    assert axis_scans[0]["best"] == pytest.approx(67.3)
    assert len(axis_scans[0]["curve"]) == 3

    output_path = tmp_path / "basin.json"
    walker.run_and_save(output_path)
    saved = json.loads(output_path.read_text())
    assert saved["fiducial_params"]["H0"] == pytest.approx(67.3)


def test_recording_observer_and_plot_generation(reference, tmp_path: Path):
    scan_presets = {
        "H0": {
            "coarse": {"type": "list", "values": [67.0, 67.3, 67.6]},
            "refine": {"type": "list", "values": [67.2, 67.3, 67.4]},
        }
    }
    observer = RecordingObserver(tmp_path, auto_run_subdir=False)

    walker = StubWalker(
        model_type="pbuf",
        datasets=["cmb"],
        reference_params=reference,
        scan_presets=scan_presets,
        param_order=("H0",),
        second_pass_params=(),
        delta_chi2=5.0,
        max_workers=1,
        observers=[observer],
    )

    walker.run()
    trace_path = observer.last_trace_path or (tmp_path / observer.filename)
    assert trace_path.exists()
    trace_data = json.loads(trace_path.read_text())
    assert trace_data.get("scans"), "Trace should include scan entries"

    plot_dir = tmp_path / "plots"
    outputs = generate_basin_plots(trace_path, plot_dir)
    assert outputs, "Expected basin plot outputs"
    for path in outputs.values():
        assert path.exists(), f"Expected plot file at {path}"


def test_cli_coordinate_command(monkeypatch, tmp_path, capsys):
    captured = {}

    class DummyWalker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["kwargs"] = kwargs
            self.second_pass_params = kwargs.get("second_pass_params")
            self.max_workers = kwargs.get("max_workers") or 1
            self.max_cycles = kwargs.get("max_cycles", 6)

        def run(self):
            captured["run_called"] = True
            return {
                "version": "coord-opt-v2",
                "model_type": self.kwargs["model_type"],
                "datasets_used": list(self.kwargs["datasets"]),
                "phase6a_enforced": self.kwargs["enforce_phase6a"],
                "fiducial_params": self.kwargs["reference_params"],
                "fiducial_chi2": 123.0,
                "axis_scans": [
                    {"param": "H0", "pass": 1, "best": 67.3, "left_edge": 67.1, "right_edge": 67.5}
                ],
            }

        def find_island_center(self, *args, **kwargs):
            pytest.fail("find_island_center should not be called when island_samples=0")

    monkeypatch.setattr("cosmos.optim.coord_optimizer.CoordinateBasinWalker", DummyWalker)
    from cosmos.optim import coord_optimizer as coord_mod

    monkeypatch.setitem(
        coord_mod.DEFAULT_REFERENCES,
        "pbuf",
        {"H0": 67.4, "Om0": 0.267137, "alpha": 0.02, "Rmax": 1.0e8, "k_sat": 0.97},
    )

    from types import SimpleNamespace
    import cli

    args = SimpleNamespace(
        model="pbuf",
        datasets=None,
        include_bao=True,
        phase6a=True,
        delta_chi2=12.5,
        output=str(tmp_path / "scan.json"),
        seed_json=None,
        skip_second_pass=True,
        quiet=False,
        no_progress=True,
        converge=False,
        max_cycles=6,
        max_workers=None,
        improvement_tol=1.0e-2,
        eps0=None,
        island_samples=0,
        island_delta=20.0,
        island_seed=None,
        basin_record_dir=None,
        basin_plot_dir=None,
    )

    cli.fit_coordinate_optimizer(args)

    kwargs = captured["kwargs"]
    assert kwargs["datasets"] == ["cmb", "sn_pantheon", "bao_iso", "bao_aniso"]
    assert kwargs["enforce_phase6a"] is True
    assert kwargs["delta_chi2"] == 12.5
    assert kwargs["second_pass_params"] == tuple()
    assert kwargs["verbose"] is True
    assert kwargs["progress"] is False
    assert kwargs["max_workers"] is None
    assert kwargs["max_cycles"] == 6
    assert kwargs["improvement_tol"] == pytest.approx(1.0e-2)
    assert kwargs["observers"] is None

    assert captured.get("run_called") is True
    saved_path = tmp_path / "scan.json"
    assert saved_path.exists()

    output_text = capsys.readouterr().out
    assert "Coordinate optimizer completed" in output_text
    assert "Δχ² tolerance" in output_text
    assert "Second pass" in output_text
    assert "Seed:" in output_text
    assert "Fiducial χ²_total" in output_text


def test_edge_rescan_expands_range(reference):
    scan_presets = {
        "H0": {
            "coarse": {"type": "list", "values": [66.9, 67.1, 67.3]},
            "refine": {
                "type": "linear_relative",
                "radius": 0.05,
                "step": 0.05,
                "clip_min": 66.5,
                "clip_max": 67.5,
            },
        }
    }

    walker = EdgeWalker(
        model_type="pbuf",
        datasets=["cmb"],
        reference_params=reference,
        scan_presets=scan_presets,
        param_order=("H0",),
        second_pass_params=(),
        max_workers=1,
    )

    result = walker.run()
    axis_scans = result["axis_scans"]
    assert len(axis_scans) >= 2
    first_scan = axis_scans[0]
    second_scan = axis_scans[1]
    assert first_scan.get("edge_hit") == "left"
    assert second_scan.get("edge_rescan") == "left"
    assert second_scan["scan_values"][0] < first_scan["scan_values"][0]
