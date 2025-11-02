from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cosmos.fits.bao.aniso.data_loader import load_bao_aniso_data
import data_interface.bao_raw_to_standardized as bao_std


def test_bao_aniso_covariance_sign_flip(tmp_path, monkeypatch):
    csv_path = tmp_path / "bao_aniso.csv"
    df = pd.DataFrame(
        {
            "z": [0.5],
            "DM_over_rd": [10.0],
            "sigma_DM_over_rd": [0.5],
            "Hz_rd_over_c": [0.2],
            "sigma_Hz_rd_over_c": [0.01],
            "rho": [0.3],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = load_bao_aniso_data(csv_path)
    cov_block = loaded["cov_list"][0]

    sigma_dm = 0.5
    sigma_dh = 0.01 / (0.2**2)
    expected_cov = -0.3 * sigma_dm * sigma_dh

    assert cov_block.shape == (2, 2)
    np.testing.assert_allclose(cov_block[0, 1], expected_cov, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cov_block[1, 0], expected_cov, rtol=1e-12, atol=1e-12)

    cache_path = tmp_path / "bao_aniso_standardized.npz"
    monkeypatch.setattr(bao_std, "RAW_PATH_ANISO", csv_path)
    monkeypatch.setattr(bao_std, "CACHE_PATH_ANISO", cache_path)
    dataset = bao_std.bao_aniso_raw_to_standardized(csv_path)

    assert dataset["cov"] is not None
    np.testing.assert_allclose(dataset["cov"][:2, :2], cov_block)

    diag_errors = np.sqrt(np.diag(dataset["cov"]))
    np.testing.assert_allclose(dataset["err"], diag_errors)
    assert dataset["meta"]["ordering"].startswith("obs interleaved as [D_M/r_d, D_H/r_d")
