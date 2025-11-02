import math
from pathlib import Path

import pytest

from cosmos.optim.coord_optimizer import CoordinateBasinWalker, DEFAULT_LCDM_REFERENCE


class IslandWalker(CoordinateBasinWalker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_cache = {}

    def _evaluate(self, params):
        key = tuple(sorted(params.items()))
        if key in self._eval_cache:
            return self._eval_cache[key]

        h0 = float(params["H0"])
        om0 = float(params["Om0"])
        delta_h0 = h0 - 70.0
        delta_om0 = om0 - 0.30
        chi2 = 1000.0 + 200.0 * delta_h0 * delta_h0 + 400.0 * delta_om0 * delta_om0
        valid = abs(delta_h0) <= 1.2 and abs(delta_om0) <= 0.05
        status = "valid" if valid else "invalid"
        payload = {
            "status": status,
            "chi2_total": chi2,
            "passes_phase6a": True,
            "chi2_breakdown": {"cmb": chi2},
        }
        self._eval_cache[key] = payload
        return payload


@pytest.fixture
def lcdm_reference():
    reference = dict(DEFAULT_LCDM_REFERENCE)
    reference.update({"H0": 70.0, "Om0": 0.30})
    return reference


def test_find_island_center_selects_interior(lcdm_reference):
    scan_presets = {
        "H0": {
            "coarse": {"type": "list", "values": [69.0, 70.0, 71.0]},
            "refine": {"type": "list", "values": [69.5, 69.8, 70.0, 70.2, 70.5]},
        },
        "Om0": {
            "coarse": {"type": "list", "values": [0.27, 0.30, 0.33]},
            "refine": {"type": "list", "values": [0.28, 0.30, 0.32]},
        },
    }

    walker = IslandWalker(
        model_type="lcdm",
        datasets=["cmb"],
        reference_params=lcdm_reference,
        scan_presets=scan_presets,
        param_order=("H0", "Om0"),
        second_pass_params=("H0", "Om0"),
        max_workers=1,
        progress=False,
    )

    result = walker.run()
    island = walker.find_island_center(result, num_samples=60, chi2_delta=5.0, seed=123)

    assert island["num_core"] <= island["num_viable"] <= island["num_samples"]
    center = island["center_params"]
    assert center["H0"] == pytest.approx(70.0, abs=0.05)
    assert center["Om0"] == pytest.approx(0.30, abs=0.01)
    assert island["center_chi2"] == pytest.approx(1000.0, abs=0.5)
    assert island["core_stats"]["chi2_min"] >= 1000.0


def test_find_island_center_requires_axis_data(lcdm_reference):
    walker = IslandWalker(
        model_type="lcdm",
        datasets=["cmb"],
        reference_params=lcdm_reference,
        param_order=("H0", "Om0"),
        second_pass_params=("H0", "Om0"),
        max_workers=1,
        progress=False,
    )
    with pytest.raises(ValueError):
        walker.find_island_center({"axis_scans": []}, num_samples=10, chi2_delta=5.0)
