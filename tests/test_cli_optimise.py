from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import types

fake_numba = types.SimpleNamespace(
    njit=lambda *args, **kwargs: (lambda fn: fn),
    jit=lambda *args, **kwargs: (lambda fn: fn),
    guvectorize=lambda *args, **kwargs: (lambda fn: fn),
    vectorize=lambda *args, **kwargs: (lambda fn: fn),
    prange=range,
)
sys.modules.setdefault("numba", fake_numba)
fake_tqdm = types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable)
sys.modules.setdefault("tqdm", fake_tqdm)

import cli.main


def _patch_model_factory(monkeypatch):
    def _factory(_model_name: str):
        return lambda params: {"params": params}

    monkeypatch.setattr(cli.main, "_make_model_factory", _factory)
    monkeypatch.setattr("cosmos2.api.engine.create_model", lambda model_name, **kwargs: {"model": model_name, **kwargs})


def _patch_fit(monkeypatch, counter):
    def _fit(_model):
        counter["count"] += 1
        return 0.5, {}

    monkeypatch.setitem(cli.main.COSMOS2_FIT_REGISTRY, "cmb", _fit)


def test_evaluate_runs(monkeypatch):
    _patch_model_factory(monkeypatch)
    counter = {"count": 0}
    _patch_fit(monkeypatch, counter)

    exit_code = cli.main.main(
        [
            "optimise",
            "--engine",
            "grid_search",
            "--samples",
            "2",
            "--model",
            "lcdm",
            "--datasets",
            "cmb",
        ]
    )
    assert exit_code == 0
    assert counter["count"] > 0


def test_multiple_batches(monkeypatch):
    _patch_model_factory(monkeypatch)
    counter = {"count": 0}
    _patch_fit(monkeypatch, counter)

    exit_code = cli.main.main(
        [
            "optimise",
            "--engine",
            "basin",
            "--scatter",
            "8",
            "--seeds",
            "3",
            "--refine",
            "5",
            "--model",
            "lcdm",
            "--datasets",
            "cmb",
        ]
    )
    assert exit_code == 0
    assert counter["count"] > 5


def test_writes_output(monkeypatch, tmp_path):
    _patch_model_factory(monkeypatch)
    counter = {"count": 0}
    _patch_fit(monkeypatch, counter)

    output_file = tmp_path / "optimisation_result.json"
    exit_code = cli.main.main(
        [
            "optimise",
            "--engine",
            "grid_search",
            "--samples",
            "3",
            "--model",
            "lcdm",
            "--datasets",
            "cmb",
            "--save-result",
            "--output",
            str(output_file),
        ]
    )
    assert exit_code == 0
    data = json.loads(output_file.read_text())
    assert data["result"]["models"]
    assert data["result"]["models"][0]["best_chi2"] >= 0.0
