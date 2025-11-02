import math
import pathlib
import sys
from typing import Dict, Any

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmos.optim.parameter_defaults import (
    PBUF_PARAMETER_DEFAULTS,
    SIGMA8_PLANCK,
)


def _base_pbuf_params() -> Dict[str, Any]:
    params = dict(PBUF_PARAMETER_DEFAULTS)
    params.setdefault("Ok0", 0.0)
    params.setdefault("Or0", 9.2e-5)
    params["Ol0"] = 0.0
    params["Obh2"] = 0.02237
    return params


def _recording_pbuf(module, monkeypatch):
    original = module.PBUF
    captured: Dict[str, Any] = {}

    def _wrapper(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs.copy()
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "PBUF", _wrapper)
    return captured


@pytest.mark.parametrize(
    ("module_path", "fn_name", "loader_module", "loader_func", "chi2_attr"),
    [
        (
            "cosmos.fits.rsd.observables",
            "chi2_rsd",
            "data_interface.rsd_loader",
            "load_rsd_data",
            "chi_squared_rsd",
        ),
        (
            "cosmos.fits.cc.observables",
            "chi2_cc",
            "data_interface.cc_loader",
            "load_cc_data",
            "chi_squared_cc",
        ),
        (
            "cosmos.fits.sn.observables",
            "chi2_sn",
            "data_interface.sn_loader",
            "load_sn_data",
            "chi_squared_sn",
        ),
        (
            "cosmos.fits.bao.iso.observables",
            "chi2_bao_iso",
            "data_interface.bao_loader",
            "load_bao_iso_data",
            "chi_squared_bao_iso",
        ),
        (
            "cosmos.fits.bao.aniso.observables",
            "chi2_bao_aniso",
            "data_interface.bao_loader",
            "load_bao_data",
            "chi_squared_bao_aniso",
        ),
    ],
)
def test_pbuf_observable_wrappers_forward_elastic_params(
    module_path,
    fn_name,
    loader_module,
    loader_func,
    chi2_attr,
    monkeypatch,
):
    module = pytest.importorskip(module_path)
    captured = _recording_pbuf(module, monkeypatch)

    # Stub the heavy dataset pipeline.
    loader = pytest.importorskip(loader_module)
    if module_path.endswith("rsd.observables"):
        growth = pytest.importorskip("cosmos.helper.growth")
        monkeypatch.setattr(
            growth,
            "growth_factor",
            lambda z, model, sigma8_0=SIGMA8_PLANCK, z_max=20.0, n_points=800: np.ones_like(np.atleast_1d(z), dtype=float),
        )
        monkeypatch.setattr(
            growth,
            "growth_rate",
            lambda z, model: np.ones_like(np.atleast_1d(z), dtype=float),
        )
    if module_path.endswith("bao.aniso.observables"):
        stub = {"z": np.array([0.3]), "obs": np.array([0.0, 0.0]), "err": np.array([0.1, 0.1]), "cov": None}
    else:
        stub = {"z": np.array([0.3]), "obs": np.array([0.0]), "err": np.array([0.1]), "cov": None}
    monkeypatch.setattr(loader, loader_func, lambda: stub)
    std = pytest.importorskip("data_interface.standardize")
    monkeypatch.setattr(std, "ensure_standard_dataset", lambda data, *_: data, raising=False)
    monkeypatch.setattr(module, chi2_attr, lambda *args, **kwargs: 0.0, raising=False)

    params = _base_pbuf_params()
    params.update(
        {
            "eps0": 0.93,
            "n_alpha": 0.17,
            "n_eps": -0.42,
            "n_R": 0.08,
            "k_sat": 0.87,
        }
    )

    getattr(module, fn_name)(params, model_type="pbuf")

    assert "kwargs" in captured, f"PBUF was not constructed in {module_path}"
    kwargs = captured["kwargs"]
    assert math.isclose(kwargs["eps0"], params["eps0"])
    assert math.isclose(kwargs["n_alpha"], params["n_alpha"])
    assert math.isclose(kwargs["n_eps"], params["n_eps"])
    assert math.isclose(kwargs["n_R"], params["n_R"])
    assert math.isclose(kwargs["k_sat"], params["k_sat"])


def test_fit_rsd_optimizer_forwards_elastic_params(monkeypatch):
    module = pytest.importorskip("cosmos.fits.rsd.optimizer")
    captured: Dict[str, Any] = {}

    class _DummyModel:
        pass

    def _recording_pbuf(*args, **kwargs):
        captured["kwargs"] = kwargs.copy()
        return _DummyModel()

    monkeypatch.setattr(module, "PBUF", _recording_pbuf)
    monkeypatch.setattr(module, "chi_squared_rsd", lambda *args, **kwargs: 0.0)

    def _fake_minimize(func, x0, *args, **kwargs):
        func(x0)

        class _Result:
            success = True
            x = x0
            fun = 0.0
            message = "stub"
            nfev = 1

        return _Result()

    monkeypatch.setattr(module, "minimize", _fake_minimize)

    init = {
        "H0": 67.1,
        "Om0": 0.31,
        "alpha": 5.0e-3,
        "Rmax": 1.5e8,
        "k_sat": 0.92,
        "eps0": 0.81,
        "n_alpha": 0.21,
        "n_eps": -0.12,
        "n_R": 0.07,
    }
    bounds = {
        "H0": (60.0, 75.0),
        "Om0": (0.2, 0.4),
        "alpha": (1e-6, 1e-1),
        "Rmax": (1e6, 1e12),
        "k_sat": (0.1, 3.0),
        "eps0": (0.2, 1.5),
        "n_alpha": (-1.0, 1.0),
        "n_eps": (-1.0, 1.0),
        "n_R": (-1.0, 1.0),
    }

    module.fit_rsd(model_type="pbuf", initial_params=init, bounds=bounds, sigma8_0=0.8)

    assert "kwargs" in captured
    kwargs = captured["kwargs"]
    for key in ("eps0", "n_alpha", "n_eps", "n_R", "k_sat"):
        assert math.isclose(kwargs[key], init[key])
