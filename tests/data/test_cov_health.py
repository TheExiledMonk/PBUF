import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")


SAMPLES_DIR = Path("pipelines/data/samples/pantheon_plus").resolve()


def prepare_dataset(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw" / "pantheon_plus"
    raw_dir.mkdir(parents=True)
    for sample in SAMPLES_DIR.iterdir():
        shutil.copy2(sample, raw_dir / sample.name)

    derived_dir = tmp_path / "derived"

    spec = importlib.util.spec_from_file_location(
        "pbuf_prepare_sn_data", Path("pipelines/data/prepare_sn_data.py").resolve()
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main(
        [
            "--raw",
            str(raw_dir),
            "--derived",
            str(derived_dir),
            "--compose-cov",
            "stat,sys",
        ]
    )
    return derived_dir


def test_covariance_condition_number_reasonable(tmp_path):
    derived_dir = prepare_dataset(tmp_path)
    cov_path = derived_dir / "supernova_index.cov.npy"
    cov = np.load(cov_path)
    cond = np.linalg.cond(cov)
    assert cond < 1e5
