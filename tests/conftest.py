import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sn_result(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "results"
    argv = ["fit_sn.py", "--model", "lcdm", "--out", str(output_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    spec = importlib.util.spec_from_file_location("pbuf_pipelines.fit_sn", PROJECT_ROOT / "pipelines" / "fit_sn.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()
    json_path = next(output_dir.glob("LCDM/*/fit_results.json"))
    result = json.loads(json_path.read_text())
    return result, json_path


@pytest.fixture
def cmb_result(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "cmb_results"
    argv = [
        "fit_cmb.py",
        "--model",
        "pbuf",
        "--priors",
        "planck2018",
        "--out",
        str(output_dir),
        "--grid-ksat",
        "12,0.5,2.0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    spec = importlib.util.spec_from_file_location("pbuf_pipelines.fit_cmb", PROJECT_ROOT / "pipelines" / "fit_cmb.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()
    json_path = next(output_dir.glob("CMB_PBUF_*/fit_results.json"))
    result = json.loads(json_path.read_text())
    return result, json_path


@pytest.fixture
def joint_result(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "joint_results"
    argv = [
        "fit_joint.py",
        "--model",
        "pbuf",
        "--datasets",
        "sn,cmb",
        "--sn-dataset",
        "mock_supernovae",
        "--cmb-priors",
        "planck2018",
        "--out",
        str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    spec = importlib.util.spec_from_file_location("pbuf_pipelines.fit_joint", PROJECT_ROOT / "pipelines" / "fit_joint.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()
    json_path = next(output_dir.glob("JOINT_PBUF_*/fit_results.json"))
    result = json.loads(json_path.read_text())
    return result, json_path
