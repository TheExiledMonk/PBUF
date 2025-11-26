import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmos2.data.registry import get_dataset as get_cosmos2_dataset  # noqa: E402


@pytest.mark.parametrize(
    "name",
    ["cmb", "sn", "bao_iso", "bao_aniso", "cc", "rsd", "wl_s8", "lensing_cross", "galaxy_pk", "sh0es"],
)
def test_standardized_dataset_parity(name: str):
    data_path = Path("data/standardized") / f"{name}.npz"
    if not data_path.exists():
        pytest.skip(f"Standardized dataset {data_path} missing")

    cosmos2_payload = get_cosmos2_dataset(name)
    assert isinstance(cosmos2_payload, dict)
    raw = np.load(data_path, allow_pickle=True)
    for key in raw.files:
        assert key in cosmos2_payload
        expected = raw[key]
        received = cosmos2_payload[key]
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(received, expected)
        else:
            assert received == expected
