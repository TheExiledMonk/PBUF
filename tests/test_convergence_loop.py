import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos.optim.coord_optimizer import DEFAULT_PBUF_REFERENCE, CoordinateBasinWalker


class ConvergenceStubWalker(CoordinateBasinWalker):
    def _evaluate(self, params):
        value = float(params["H0"])
        delta = value - 67.3
        chi2 = 1500.0 + delta * delta * 100.0
        return {
            "status": "valid",
            "chi2_total": chi2,
            "passes_phase6a": True,
            "chi2_breakdown": {"cmb": chi2},
        }


def test_convergence_cycles_reduce_chi2():
    reference = dict(DEFAULT_PBUF_REFERENCE)
    reference["H0"] = 67.1
    scan_presets = {
        "H0": {
            "coarse": {"type": "list", "values": [66.9, 67.1, 67.2]},
            "refine": {
                "type": "linear_relative",
                "radius": 0.05,
                "step": 0.05,
                "clip_min": 66.5,
                "clip_max": 68.0,
            },
        }
    }

    walker = ConvergenceStubWalker(
        model_type="pbuf",
        datasets=["cmb"],
        reference_params=reference,
        scan_presets=scan_presets,
        param_order=("H0",),
        second_pass_params=("H0",),
        max_workers=1,
        improvement_tol=0.05,
        max_cycles=5,
    )

    result = walker.run_with_convergence()
    convergence = result["convergence"]
    history = convergence["chi2_history"]
    numeric_history = [value for value in history if value is not None]

    assert convergence["converged"] is True
    assert convergence["cycles_completed"] == len(history)
    assert len(history) >= 2
    for first, second in zip(numeric_history, numeric_history[1:]):
        assert second <= first + 1e-9
    assert numeric_history[-1] <= numeric_history[0] + 1e-9
    assert result["fiducial_chi2"] == pytest.approx(numeric_history[-1])

    cycle_deltas = [cycle.get("delta_chi2") for cycle in convergence["cycles"] if "delta_chi2" in cycle]
    assert cycle_deltas
    assert all(delta is None or delta >= -1e-9 for delta in cycle_deltas)

    cycles_reported = {entry.get("cycle") for entry in result["axis_scans"] if "cycle" in entry}
    assert {0, 1}.issubset(cycles_reported)
