"""
Integration test for parameter module with actual cmb_priors.

This test verifies that the parameter module works correctly with the real
prepare_background_params function.
"""

import pytest
from . import parameter


def test_real_integration_lcdm():
    """Test parameter construction with real prepare_background_params for ΛCDM."""
    result = parameter.build_params("lcdm")
    
    # Verify all expected parameters are present
    expected_base = {"H0", "Om0", "Obh2", "ns", "Neff", "Tcmb", "recomb_method"}
    expected_derived = {"Omh2", "Orh2", "z_recomb", "z_drag", "r_s_drag"}
    expected_meta = {"model_class"}
    
    expected_all = expected_base | expected_derived | expected_meta
    # Allow extra metadata keys that start with underscore
    actual_keys = {k for k in result.keys() if not k.startswith('_')}
    assert actual_keys == expected_all
    
    # Verify model metadata
    assert result["model_class"] == "lcdm"
    
    # Verify derived parameters are reasonable
    assert 800 < result["z_recomb"] < 1500
    assert 800 < result["z_drag"] < 1500
    assert 50 < result["r_s_drag"] < 300
    assert result["z_recomb"] > result["z_drag"]  # Physical consistency
    
    # Verify density consistency
    expected_Omh2 = result["Om0"] * (result["H0"] / 100.0)**2
    assert abs(result["Omh2"] - expected_Omh2) < 1e-4


def test_real_integration_pbuf():
    """Test parameter construction with real prepare_background_params for PBUF."""
    result = parameter.build_params("pbuf")
    
    # Verify all expected parameters are present
    expected_base = {"H0", "Om0", "Obh2", "ns", "Neff", "Tcmb", "recomb_method"}
    expected_pbuf = {"alpha", "Rmax", "eps0", "n_eps", "k_sat"}
    expected_derived = {"Omh2", "Orh2", "z_recomb", "z_drag", "r_s_drag"}
    expected_meta = {"model_class"}
    
    expected_all = expected_base | expected_pbuf | expected_derived | expected_meta
    # Allow extra metadata keys that start with underscore
    actual_keys = {k for k in result.keys() if not k.startswith('_')}
    assert actual_keys == expected_all
    
    # Verify model metadata
    assert result["model_class"] == "pbuf"
    
    # Verify PBUF-specific parameters
    assert result["alpha"] == 5e-4
    assert result["Rmax"] == 1e9
    assert result["eps0"] == 0.7
    assert result["n_eps"] == 0.0
    assert result["k_sat"] == 0.9762


def test_real_integration_with_overrides():
    """Test parameter construction with overrides and real prepare_background_params."""
    overrides = {"H0": 70.0, "Om0": 0.3}
    result = parameter.build_params("lcdm", overrides)
    
    # Verify overrides were applied
    assert result["H0"] == 70.0
    assert result["Om0"] == 0.3
    
    # Verify derived parameters reflect the overrides
    expected_Omh2 = 0.3 * (70.0 / 100.0)**2
    assert abs(result["Omh2"] - expected_Omh2) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])