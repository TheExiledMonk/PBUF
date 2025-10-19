import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")

SAMPLES_DIR = Path("pipelines/data/samples/pantheon_plus").resolve()


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "pbuf_prepare_sn_data", Path("pipelines/data/prepare_sn_data.py").resolve()
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def test_prepare_sn_data_creates_artifacts(tmp_path):
    raw_dir = tmp_path / "raw" / "pantheon_plus"
    raw_dir.mkdir(parents=True)
    for sample in SAMPLES_DIR.iterdir():
        shutil.copy2(sample, raw_dir / sample.name)

    derived_dir = tmp_path / "derived"

    module = _load_prepare_module()
    argv = [
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
    module.main(argv)

    table_path = derived_dir / "supernova_index.csv"
    cov_path = derived_dir / "supernova_index.cov.npy"
    meta_path = derived_dir / "supernova_index.meta.json"

    assert table_path.exists()
    assert cov_path.exists()
    assert meta_path.exists()

    first_hash = sha256sum(table_path)
    module.main(argv)  # rerun for determinism
    second_hash = sha256sum(table_path)
    assert first_hash == second_hash

    cov = np.load(cov_path)
    assert cov.shape[0] == cov.shape[1] == 3
    assert np.allclose(cov, cov.T)

    meta = json.loads(meta_path.read_text())
    assert meta["transform_version"] == "sn_prepare_v1"
    assert meta["z_prefer"] == "z_cmb"
    assert len(meta["records"]) >= 3
