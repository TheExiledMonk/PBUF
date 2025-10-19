import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from dataio.loaders import load_sn_pantheon

pytest.importorskip("pandas")

SAMPLES_DIR = Path("pipelines/data/samples/pantheon_plus").resolve()


def _run_preparer(tmp_path: Path) -> Path:
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
            "--z-prefer",
            "z_cmb",
            "--compose-cov",
            "stat,sys",
            "--release-tag",
            "pantheon_plus_mock",
        ]
    )
    return derived_dir


def test_load_sn_pantheon_schema(tmp_path):
    derived_dir = _run_preparer(tmp_path)
    config_entry = {
        "prepared": {
            "table": derived_dir / "supernova_index.csv",
            "parquet": derived_dir / "supernova_index.parquet",
            "cov": derived_dir / "supernova_index.cov.npy",
            "meta": derived_dir / "supernova_index.meta.json",
        },
        "z_prefer": "z_cmb",
        "release_tag": "pantheon_plus_mock",
    }

    dataset = load_sn_pantheon(config_entry)
    assert set(dataset.keys()) == {"z", "y", "mu", "sigma", "sigma_mu", "cov", "tags", "index_map", "meta"}
    assert dataset["z"].shape == (3,)
    assert dataset["cov"].shape == (3, 3)
    assert dataset["index_map"].dtype.kind in {"i", "u"}
    assert dataset["meta"]["transform_version"] == "sn_prepare_v1"

    diag = np.diag(dataset["cov"])
    idx = dataset["index_map"][0]
    assert diag[idx] > 0
