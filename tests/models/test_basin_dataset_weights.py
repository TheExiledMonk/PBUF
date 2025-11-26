import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos.models.lcdm.basin_utils import LCDMBasinModel


def _lcdm_params(h0: float = 67.0) -> dict[str, float]:
    return {
        "H0": h0,
        "Omega_m0": 0.3,
        "Omega_b0": 0.05,
        "Omega_r0": 9.0e-5,
        "Omega_k0": 0.0,
    }


def test_lcdm_basin_model_respects_dataset_weights():
    params = _lcdm_params()
    base = LCDMBasinModel()
    weighted = LCDMBasinModel(dataset_weights={"cmb": 2.0})

    base_chi2 = base.evaluate(params, ["cmb"])
    weighted_chi2 = weighted.evaluate(params, ["cmb"])

    assert pytest.approx(weighted_chi2) == 2.0 * base_chi2

