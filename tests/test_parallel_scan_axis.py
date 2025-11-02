import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos.optim.coord_optimizer import DEFAULT_PBUF_REFERENCE, CoordinateBasinWalker


class ParallelStubWalker(CoordinateBasinWalker):
    def _evaluate(self, params):
        value = float(params["H0"])
        delta = value - 67.3
        chi2 = 1500.0 + delta * delta * 50.0
        status = "valid"
        passes_phase6a = True
        if value > 67.5:
            status = "invalid"
            passes_phase6a = False
        return {
            "status": status,
            "chi2_total": chi2,
            "passes_phase6a": passes_phase6a,
            "chi2_breakdown": {"cmb": chi2},
        }


def test_parallel_scan_matches_serial():
    reference = dict(DEFAULT_PBUF_REFERENCE)
    scan_presets = {
        "H0": {
            "coarse": {"type": "list", "values": [67.0, 67.3, 67.6]},
            "refine": {"type": "list", "values": [67.2, 67.3, 67.4]},
        }
    }
    scan_values = [67.0, 67.3, 67.6]

    serial_walker = ParallelStubWalker(
        model_type="pbuf",
        datasets=["cmb"],
        reference_params=reference,
        scan_presets=scan_presets,
        param_order=("H0",),
        second_pass_params=("H0",),
        max_workers=1,
        progress=False,
    )
    parallel_walker = ParallelStubWalker(
        model_type="pbuf",
        datasets=["cmb"],
        reference_params=reference,
        scan_presets=scan_presets,
        param_order=("H0",),
        second_pass_params=("H0",),
        max_workers=2,
        progress=False,
    )

    summary_serial = serial_walker._scan_axis("H0", dict(reference), scan_values, pass_id=1)
    summary_parallel = parallel_walker._scan_axis("H0", dict(reference), scan_values, pass_id=1)

    assert summary_serial["best"] == pytest.approx(summary_parallel["best"])
    assert summary_serial["chi2_min"] == pytest.approx(summary_parallel["chi2_min"])
    assert summary_serial["curve"] == summary_parallel["curve"]
    assert summary_parallel["num_valid"] == summary_serial["num_valid"]
    assert summary_parallel["curve"][-1]["valid"] is False
    assert summary_parallel["curve"][-1]["passes_phase6a"] is False
