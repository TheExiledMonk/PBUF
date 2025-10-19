from pathlib import Path


def test_fit_joint_pipeline(joint_result):
    result, json_path = joint_result
    assert json_path.exists()
    assert result["model"] == "PBUF"
    metrics = result["metrics"]
    assert set(metrics.keys()) == {"total", "sn", "cmb"}
    assert metrics["total"]["chi2"] >= metrics["sn"]["chi2"]
    params = result["parameters"]
    assert "k_sat" in params
    assert "Obh2" in params
    cmb_pred = result["predictions"]["cmb"]
    for key in ["100theta_*", "l_A", "R", "Obh2"]:
        assert key in cmb_pred
    assert "Omegabh2" in cmb_pred
    assert "ns" in cmb_pred
    cmb_priors = result["priors"]["cmb"]
    assert cmb_priors["labels"]
    figures = result.get("figures", {})
    assert "sn_residuals_vs_z" in figures
    for path in figures.values():
        assert Path(path).exists()
