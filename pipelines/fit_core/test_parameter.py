"""
Unit tests for parameter construction and validation module.

Tests cover default parameter retrieval, override application, parameter validation,
and integration with prepare_background_params() function.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from . import parameter
from . import ParameterDict


class TestGetDefaults:
    """Test default parameter retrieval for both ΛCDM and PBUF models."""
    
    def test_lcdm_defaults(self):
        """Test ΛCDM default parameter retrieval."""
        defaults = parameter.get_defaults("lcdm")
        
        # Verify all required ΛCDM parameters are present
        expected_params = {
            "H0", "Om0", "Obh2", "ns", "Neff", "Tcmb", "recomb_method"
        }
        assert set(defaults.keys()) == expected_params
        
        # Verify specific default values
        assert defaults["H0"] == 67.4
        assert defaults["Om0"] == 0.315
        assert defaults["Obh2"] == 0.02237
        assert defaults["ns"] == 0.9649
        assert defaults["Neff"] == 3.046
        assert defaults["Tcmb"] == 2.7255
        assert defaults["recomb_method"] == "PLANCK18"
        
        # Verify types
        assert isinstance(defaults["H0"], (int, float))
        assert isinstance(defaults["Om0"], (int, float))
        assert isinstance(defaults["recomb_method"], str)
    
    def test_pbuf_defaults(self):
        """Test PBUF default parameter retrieval."""
        defaults = parameter.get_defaults("pbuf")
        
        # Verify all required PBUF parameters are present (includes ΛCDM + PBUF-specific)
        expected_params = {
            "H0", "Om0", "Obh2", "ns", "Neff", "Tcmb", "recomb_method",
            "alpha", "Rmax", "eps0", "n_eps", "k_sat"
        }
        assert set(defaults.keys()) == expected_params
        
        # Verify ΛCDM parameters are identical to ΛCDM defaults
        lcdm_defaults = parameter.get_defaults("lcdm")
        for param in lcdm_defaults:
            assert defaults[param] == lcdm_defaults[param]
        
        # Verify PBUF-specific default values
        assert defaults["alpha"] == 5e-4
        assert defaults["Rmax"] == 1e9
        assert defaults["eps0"] == 0.7
        assert defaults["n_eps"] == 0.0
        assert defaults["k_sat"] == 0.9762
        
        # Verify types
        assert isinstance(defaults["alpha"], (int, float))
        assert isinstance(defaults["Rmax"], (int, float))
        assert isinstance(defaults["k_sat"], (int, float))
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model types."""
        with pytest.raises(ValueError, match="Unknown model type: invalid"):
            parameter.get_defaults("invalid")
        
        with pytest.raises(ValueError, match="Must be 'lcdm' or 'pbuf'"):
            parameter.get_defaults("wcdm")
    
    def test_defaults_are_copies(self):
        """Test that get_defaults returns independent copies."""
        defaults1 = parameter.get_defaults("lcdm")
        defaults2 = parameter.get_defaults("lcdm")
        
        # Modify one copy
        defaults1["H0"] = 999.0
        
        # Verify the other copy is unchanged
        assert defaults2["H0"] == 67.4
        
        # Verify original DEFAULTS dict is unchanged
        defaults3 = parameter.get_defaults("lcdm")
        assert defaults3["H0"] == 67.4


class TestApplyOverrides:
    """Test parameter override application logic."""
    
    def test_basic_override_application(self):
        """Test basic parameter override functionality."""
        base_params = {"H0": 67.4, "Om0": 0.315, "ns": 0.9649}
        overrides = {"H0": 70.0, "Om0": 0.3}
        
        result = parameter.apply_overrides(base_params, overrides)
        
        # Verify overrides were applied
        assert result["H0"] == 70.0
        assert result["Om0"] == 0.3
        
        # Verify non-overridden parameters unchanged
        assert result["ns"] == 0.9649
        
        # Verify original base_params unchanged
        assert base_params["H0"] == 67.4
    
    def test_new_parameter_addition(self):
        """Test adding new parameters via overrides."""
        base_params = {"H0": 67.4, "Om0": 0.315}
        overrides = {"new_param": 1.5}
        
        result = parameter.apply_overrides(base_params, overrides)
        
        assert result["new_param"] == 1.5
        assert len(result) == len(base_params) + 1
    
    def test_string_parameter_override(self):
        """Test overriding string parameters."""
        base_params = {"recomb_method": "PLANCK18", "H0": 67.4}
        overrides = {"recomb_method": "HS96"}
        
        result = parameter.apply_overrides(base_params, overrides)
        
        assert result["recomb_method"] == "HS96"
        assert result["H0"] == 67.4
    
    def test_invalid_override_types(self):
        """Test error handling for invalid override value types."""
        base_params = {"H0": 67.4}
        
        # Test list override (invalid)
        with pytest.raises(ValueError, match="Invalid override value type"):
            parameter.apply_overrides(base_params, {"H0": [1, 2, 3]})
        
        # Test dict override (invalid)
        with pytest.raises(ValueError, match="Invalid override value type"):
            parameter.apply_overrides(base_params, {"H0": {"nested": "dict"}})
        
        # Test None override (invalid)
        with pytest.raises(ValueError, match="Invalid override value type"):
            parameter.apply_overrides(base_params, {"H0": None})
    
    def test_empty_overrides(self):
        """Test behavior with empty overrides."""
        base_params = {"H0": 67.4, "Om0": 0.315}
        
        # Empty dict
        result = parameter.apply_overrides(base_params, {})
        assert result == base_params
        assert result is not base_params  # Should be a copy
        
        # None overrides (handled by build_params)
        result = parameter.apply_overrides(base_params, {})
        assert result == base_params


class TestValidateParams:
    """Test parameter validation logic."""
    
    def test_valid_lcdm_parameters(self):
        """Test validation of valid ΛCDM parameters."""
        valid_params = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "Neff": 3.046,
            "Tcmb": 2.7255,
            "recomb_method": "PLANCK18"
        }
        
        # Should not raise any exceptions
        assert parameter.validate_params(valid_params, "lcdm") is True
    
    def test_valid_pbuf_parameters(self):
        """Test validation of valid PBUF parameters."""
        valid_params = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "Neff": 3.046,
            "Tcmb": 2.7255,
            "recomb_method": "PLANCK18",
            "alpha": 5e-4,
            "Rmax": 1e9,
            "eps0": 0.7,
            "n_eps": 0.0,
            "k_sat": 0.9762
        }
        
        # Should not raise any exceptions
        assert parameter.validate_params(valid_params, "pbuf") is True
    
    def test_missing_required_parameters(self):
        """Test error handling for missing required parameters."""
        # Missing H0 for ΛCDM
        incomplete_lcdm = {
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "Neff": 3.046,
            "Tcmb": 2.7255,
            "recomb_method": "PLANCK18"
        }
        
        with pytest.raises(ValueError, match="Missing required parameters"):
            parameter.validate_params(incomplete_lcdm, "lcdm")
        
        # Missing PBUF-specific parameters
        lcdm_only = parameter.get_defaults("lcdm")
        with pytest.raises(ValueError, match="Missing required parameters"):
            parameter.validate_params(lcdm_only, "pbuf")
    
    def test_physical_bounds_validation(self):
        """Test validation of physical parameter bounds."""
        base_params = parameter.get_defaults("lcdm")
        
        # Test H0 bounds
        invalid_h0_low = base_params.copy()
        invalid_h0_low["H0"] = 10.0  # Below minimum
        with pytest.raises(ValueError, match="Parameter H0=10.0 outside physical bounds"):
            parameter.validate_params(invalid_h0_low, "lcdm")
        
        invalid_h0_high = base_params.copy()
        invalid_h0_high["H0"] = 200.0  # Above maximum
        with pytest.raises(ValueError, match="Parameter H0=200.0 outside physical bounds"):
            parameter.validate_params(invalid_h0_high, "lcdm")
        
        # Test Om0 bounds
        invalid_om0 = base_params.copy()
        invalid_om0["Om0"] = 1.5  # Above maximum
        with pytest.raises(ValueError, match="Parameter Om0=1.5 outside physical bounds"):
            parameter.validate_params(invalid_om0, "lcdm")
        
        # Test PBUF parameter bounds
        pbuf_params = parameter.get_defaults("pbuf")
        invalid_alpha = pbuf_params.copy()
        invalid_alpha["alpha"] = 1.0  # Above maximum
        with pytest.raises(ValueError, match="Parameter alpha=1.0 outside physical bounds"):
            parameter.validate_params(invalid_alpha, "pbuf")
    
    def test_recombination_method_validation(self):
        """Test validation of recombination method parameter."""
        base_params = parameter.get_defaults("lcdm")
        
        # Valid methods
        for method in ["PLANCK18", "HS96", "EH98"]:
            valid_params = base_params.copy()
            valid_params["recomb_method"] = method
            assert parameter.validate_params(valid_params, "lcdm") is True
        
        # Invalid method
        invalid_params = base_params.copy()
        invalid_params["recomb_method"] = "INVALID"
        with pytest.raises(ValueError, match="Invalid recombination method"):
            parameter.validate_params(invalid_params, "lcdm")


class TestBuildParams:
    """Test complete parameter construction with integration."""
    
    @patch('pipelines.fit_core.parameter.cmb_priors.prepare_background_params')
    def test_lcdm_parameter_construction(self, mock_prepare):
        """Test complete ΛCDM parameter construction."""
        # Mock the prepare_background_params function with consistent values
        h = 67.4 / 100.0
        expected_Omh2 = 0.315 * h**2
        mock_derived = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "Omh2": expected_Omh2, "Orh2": 2.469e-5, "z_recomb": 1089.80,
            "z_drag": 1059.57, "r_s_drag": 147.09
        }
        mock_prepare.return_value = mock_derived
        
        result = parameter.build_params("lcdm")
        
        # Verify prepare_background_params was called
        mock_prepare.assert_called_once()
        
        # Verify model metadata was added
        assert result["model_class"] == "lcdm"
        
        # Verify all expected parameters are present
        expected_keys = set(mock_derived.keys()) | {"model_class"}
        assert set(result.keys()) == expected_keys
    
    @patch('pipelines.fit_core.parameter.cmb_priors.prepare_background_params')
    def test_pbuf_parameter_construction(self, mock_prepare):
        """Test complete PBUF parameter construction."""
        # Mock the prepare_background_params function with consistent values
        h = 67.4 / 100.0
        expected_Omh2 = 0.315 * h**2
        mock_derived = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "alpha": 5e-4, "Rmax": 1e9, "eps0": 0.7, "n_eps": 0.0, "k_sat": 0.9762,
            "Omh2": expected_Omh2, "Orh2": 2.469e-5, "z_recomb": 1089.80,
            "z_drag": 1059.57, "r_s_drag": 147.09
        }
        mock_prepare.return_value = mock_derived
        
        result = parameter.build_params("pbuf")
        
        # Verify prepare_background_params was called
        mock_prepare.assert_called_once()
        
        # Verify model metadata was added
        assert result["model_class"] == "pbuf"
        
        # Verify PBUF-specific parameters are present
        pbuf_params = {"alpha", "Rmax", "eps0", "n_eps", "k_sat"}
        assert pbuf_params.issubset(set(result.keys()))
    
    @patch('pipelines.fit_core.parameter.cmb_priors.prepare_background_params')
    def test_parameter_construction_with_overrides(self, mock_prepare):
        """Test parameter construction with overrides applied."""
        # Mock the prepare_background_params function with consistent values
        h = 70.0 / 100.0
        expected_Omh2 = 0.3 * h**2
        mock_derived = {
            "H0": 70.0, "Om0": 0.3, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "HS96",
            "Omh2": expected_Omh2, "Orh2": 2.469e-5, "z_recomb": 1100.0,  # Higher than z_drag
            "z_drag": 1059.57, "r_s_drag": 147.09
        }
        mock_prepare.return_value = mock_derived
        
        overrides = {"H0": 70.0, "Om0": 0.3, "recomb_method": "HS96"}
        result = parameter.build_params("lcdm", overrides)
        
        # Verify overrides were applied before calling prepare_background_params
        call_args = mock_prepare.call_args[0][0]
        assert call_args["H0"] == 70.0
        assert call_args["Om0"] == 0.3
        assert call_args["recomb_method"] == "HS96"
        
        # Verify final result contains overridden values
        assert result["H0"] == 70.0
        assert result["Om0"] == 0.3
        assert result["recomb_method"] == "HS96"
    
    def test_invalid_model_in_build_params(self):
        """Test error handling for invalid model in build_params."""
        with pytest.raises(ValueError, match="Unknown model type: invalid"):
            parameter.build_params("invalid")
    
    @patch('pipelines.fit_core.parameter.cmb_priors.prepare_background_params')
    def test_derived_parameter_validation(self, mock_prepare):
        """Test validation of derived parameters."""
        # Mock prepare_background_params to return invalid derived parameters
        mock_derived = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "Omh2": 0.143, "Orh2": 2.469e-5, "z_recomb": 500.0,  # Invalid: too low
            "z_drag": 1059.57, "r_s_drag": 147.09
        }
        mock_prepare.return_value = mock_derived
        
        with pytest.raises(ValueError, match="Derived parameter z_recomb=500.0 outside expected bounds"):
            parameter.build_params("lcdm")
    
    @patch('pipelines.fit_core.parameter.cmb_priors.prepare_background_params')
    def test_physical_consistency_validation(self, mock_prepare):
        """Test validation of physical consistency relationships."""
        # Mock prepare_background_params with inconsistent z_recomb < z_drag
        mock_derived = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "Omh2": 0.143, "Orh2": 2.469e-5, "z_recomb": 1000.0,
            "z_drag": 1100.0, "r_s_drag": 147.09  # z_drag > z_recomb (invalid)
        }
        mock_prepare.return_value = mock_derived
        
        with pytest.raises(ValueError, match="Recombination redshift .* should be greater than drag epoch"):
            parameter.build_params("lcdm")
    
    @patch('pipelines.fit_core.parameter.cmb_priors.prepare_background_params')
    def test_density_consistency_validation(self, mock_prepare):
        """Test validation of density parameter consistency."""
        # Mock prepare_background_params with inconsistent Omh2
        mock_derived = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "Omh2": 0.200,  # Inconsistent with Om0 * h^2 = 0.315 * 0.674^2 ≈ 0.143
            "Orh2": 2.469e-5, "z_recomb": 1089.80,
            "z_drag": 1059.57, "r_s_drag": 147.09
        }
        mock_prepare.return_value = mock_derived
        
        with pytest.raises(ValueError, match="Inconsistent physical matter density"):
            parameter.build_params("lcdm")


class TestIntegrationWithPrepareBackgroundParams:
    """Test integration with prepare_background_params function."""
    
    def test_prepare_background_params_called_correctly(self):
        """Test that prepare_background_params is called with correct parameters."""
        with patch('pipelines.fit_core.parameter.cmb_priors.prepare_background_params') as mock_prepare:
            # Setup mock return value with consistent Omh2
            mock_derived = parameter.get_defaults("lcdm").copy()
            h = 70.0 / 100.0
            expected_Omh2 = 0.315 * h**2
            mock_derived.update({
                "H0": 70.0,  # Include the override
                "Omh2": expected_Omh2, "Orh2": 2.469e-5, "z_recomb": 1089.80,
                "z_drag": 1059.57, "r_s_drag": 147.09
            })
            mock_prepare.return_value = mock_derived
            
            # Call build_params
            overrides = {"H0": 70.0}
            result = parameter.build_params("lcdm", overrides)
            
            # Verify prepare_background_params was called once
            assert mock_prepare.call_count == 1
            
            # Verify it was called with the correct parameters (defaults + overrides)
            call_args = mock_prepare.call_args[0][0]
            assert call_args["H0"] == 70.0  # Override applied
            assert call_args["Om0"] == 0.315  # Default value
            assert call_args["Obh2"] == 0.02237  # Default value
    
    def test_prepare_background_params_return_value_used(self):
        """Test that return value from prepare_background_params is properly used."""
        with patch('pipelines.fit_core.parameter.cmb_priors.prepare_background_params') as mock_prepare:
            # Setup mock return value with specific derived parameters
            # Use consistent Omh2 value to pass validation
            h = 67.4 / 100.0
            consistent_Omh2 = 0.315 * h**2
            mock_derived = parameter.get_defaults("lcdm").copy()
            mock_derived.update({
                "Omh2": consistent_Omh2,  # Consistent with Om0 and H0
                "Orh2": 1.234e-5,  # Distinctive value
                "z_recomb": 1234.56,  # Distinctive value
                "z_drag": 1111.11,  # Distinctive value
                "r_s_drag": 222.22  # Distinctive value
            })
            mock_prepare.return_value = mock_derived
            
            result = parameter.build_params("lcdm")
            
            # Verify derived parameters are in the result
            assert result["Omh2"] == consistent_Omh2
            assert result["Orh2"] == 1.234e-5
            assert result["z_recomb"] == 1234.56
            assert result["z_drag"] == 1111.11
            assert result["r_s_drag"] == 222.22
    
    def test_missing_derived_parameters_error(self):
        """Test error handling when prepare_background_params returns incomplete data."""
        with patch('pipelines.fit_core.parameter.cmb_priors.prepare_background_params') as mock_prepare:
            # Mock return value missing required derived parameters
            incomplete_derived = parameter.get_defaults("lcdm").copy()
            incomplete_derived.update({
                "Omh2": 0.143,
                "Orh2": 2.469e-5,
                # Missing z_recomb, z_drag, r_s_drag
            })
            mock_prepare.return_value = incomplete_derived
            
            with pytest.raises(ValueError, match="Missing derived parameters"):
                parameter.build_params("lcdm")


class TestParameterValidationEdgeCases:
    """Test edge cases and boundary conditions in parameter validation."""
    
    def test_boundary_values_validation(self):
        """Test validation at parameter boundary values."""
        base_params = parameter.get_defaults("lcdm")
        
        # Test exact boundary values (should pass)
        boundary_params = base_params.copy()
        boundary_params["H0"] = 20.0  # Minimum allowed
        assert parameter.validate_params(boundary_params, "lcdm") is True
        
        boundary_params["H0"] = 150.0  # Maximum allowed
        assert parameter.validate_params(boundary_params, "lcdm") is True
        
        # Test just outside boundaries (should fail)
        invalid_params = base_params.copy()
        invalid_params["H0"] = 19.99  # Just below minimum
        with pytest.raises(ValueError, match="outside physical bounds"):
            parameter.validate_params(invalid_params, "lcdm")
    
    def test_non_numeric_parameter_handling(self):
        """Test handling of non-numeric parameters in bounds checking."""
        base_params = parameter.get_defaults("lcdm")
        
        # String parameters should not be bounds-checked
        string_params = base_params.copy()
        string_params["custom_string"] = "some_value"
        assert parameter.validate_params(string_params, "lcdm") is True
    
    def test_integer_parameter_handling(self):
        """Test that integer parameters are handled correctly."""
        base_params = parameter.get_defaults("lcdm")
        
        # Integer values should be accepted for numeric parameters
        int_params = base_params.copy()
        int_params["H0"] = 67  # Integer instead of float
        int_params["Neff"] = 3  # Integer instead of float
        assert parameter.validate_params(int_params, "lcdm") is True


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])