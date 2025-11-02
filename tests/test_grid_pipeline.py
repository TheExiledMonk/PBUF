import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cosmos.optim.dataset_evaluators import CHI2_PENALTIES
from cosmos.optim.grid_pipeline import evaluate_cosmology, run_grid_search


def test_evaluate_cosmology_deterministic():
    params = {
        "H0": 67.36,
        "Om0": 0.3153,
        "Or0": 9.2e-5,
        "Ok0": 0.0,
    }
    datasets = ["cmb"]

    first = evaluate_cosmology("lcdm", params, datasets)
    second = evaluate_cosmology("lcdm", params, datasets)

    assert first["status"] == "valid"
    assert second["status"] == "valid"
    assert first["chi2_breakdown"]["cmb"] == pytest.approx(second["chi2_breakdown"]["cmb"])
    assert first["chi2_total"] == pytest.approx(second["chi2_total"])


def test_invalid_cosmology_flagged():
    params = {
        "H0": -10.0,
        "Om0": 0.3153,
        "Or0": 9.2e-5,
        "Ok0": 0.0,
    }
    result = evaluate_cosmology("lcdm", params, ["cmb"])
    assert result["status"] == "invalid"
    assert result["chi2_total"] == CHI2_PENALTIES["validation_failed"]
    assert result["validation"]["reasons"]


def test_run_grid_search_writes_results(tmp_path):
    grid = {
        "H0": [67.36],
        "Om0": [0.3153],
        "Or0": [9.2e-5],
        "Ok0": [0.0],
    }

    result = run_grid_search(
        "lcdm",
        datasets=["cmb"],
        grid=grid,
        workers=1,
        output_dir=tmp_path,
        tag="unit",
    )

    assert result["num_evaluations"] == 1
    assert result["num_valid"] == 1
    assert result["num_invalid"] == 0
    assert result["ranking"][0]["id"] == result["best"]["id"]
    results_path = Path(result["results_file"])
    assert results_path.exists()

    payload = json.loads(results_path.read_text())
    assert payload["model_type"] == "lcdm"
    assert payload["num_evaluations"] == 1
    assert payload["num_valid"] == 1
    assert payload["num_invalid"] == 0


def test_run_grid_search_with_refinement(tmp_path):
    grid = {
        "H0": [60, 80],
        "Om0": [0.2, 0.4],
        "Or0": [9.2e-5],
        "Ok0": [0.0],
    }

    result = run_grid_search(
        "lcdm",
        datasets=["cmb"],
        grid=grid,
        workers=1,
        output_dir=tmp_path,
        tag="unit-refine",
        refine_top=1,
        refine_fraction=0.1,
        refine_points=2,
    )

    assert result.get("refined_evaluations", 0) > 0
    assert result["num_evaluations"] > 1
