def test_fit_sn_pipeline(sn_result):
    result, json_path = sn_result
    assert json_path.exists()
    for key in ["run_id", "timestamp", "dataset", "model", "parameters", "metrics", "figures"]:
        assert key in result
    assert result["model"] == "LCDM"
