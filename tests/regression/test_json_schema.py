REQUIRED_TOP_LEVEL = {
    "run_id",
    "timestamp",
    "mock",
    "dataset",
    "model",
    "parameters",
    "evolution_policy",
    "metrics",
    "data_vectors",
    "figures",
    "provenance",
}


def test_fit_json_schema(sn_result):
    result, _ = sn_result
    missing = REQUIRED_TOP_LEVEL - set(result.keys())
    assert not missing, f"Missing keys: {missing}"

    assert isinstance(result["data_vectors"]["z"], str)
    assert result["figures"]["residuals_vs_z"].endswith(".png")
