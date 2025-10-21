from pathlib import Path

from utils.parameters import flatten_payload


def test_fit_cmb_pipeline(cmb_result):
    result, json_path = cmb_result
    assert json_path.exists()
    assert result["model"] == "PBUF"
    for key in ["run_id", "timestamp", "parameters", "metrics", "predictions", "grid_scan"]:
        assert key in result
    params = result["parameters"]
    if isinstance(params, dict) and "global" in params:
        params_flat = flatten_payload(params)
    else:
        params_flat = params
    assert "k_sat" in params_flat
    grid = result["grid_scan"]
    assert len(grid["k_sat"]) == len(grid["chi2"])
    figure_path = result["figures"].get("chi2_vs_k_sat")
    assert figure_path is not None
    assert Path(figure_path).exists()
