#!/usr/bin/env python3
"""
Test suite for BAO dataset separation validation.

Tests the implementation of task 5: Add joint fit separation validation
- Implement checks to prevent simultaneous use of "bao" and "bao_ani" datasets
- Add validation warnings when both isotropic and anisotropic BAO are requested
- Create configuration validation for proper dataset separation
- Document best practices for BAO dataset selection

Requirements: 1.1, 1.2
"""

import pytest
import sys
import os
from typing import List, Dict, Any

# Add the parent directory to the path so we can import fit_core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fit_core.bao_aniso_validation import (
    validate_bao_dataset_separation,
    validate_joint_fit_configuration,
    get_bao_dataset_selection_guide,
    print_dataset_separation_warning,
    create_dataset_configuration_validator,
    BAOValidationError
)


class TestBAODatasetSeparation:
    """Test BAO dataset separation validation functionality."""
    
    def test_valid_isotropic_bao_only(self):
        """Test that isotropic BAO only is valid."""
        datasets = ["cmb", "bao", "sn"]
        # Should not raise any exception
        validate_bao_dataset_separation(datasets)
    
    def test_valid_anisotropic_bao_only(self):
        """Test that anisotropic BAO only is valid."""
        datasets = ["bao_ani"]
        # Should not raise any exception
        validate_bao_dataset_separation(datasets)
    
    def test_valid_no_bao_datasets(self):
        """Test that configurations without BAO are valid."""
        datasets = ["cmb", "sn"]
        # Should not raise any exception
        validate_bao_dataset_separation(datasets)
    
    def test_invalid_mixed_bao_datasets(self):
        """Test that mixing bao and bao_ani raises error."""
        datasets = ["cmb", "bao", "bao_ani", "sn"]
        
        with pytest.raises(BAOValidationError) as exc_info:
            validate_bao_dataset_separation(datasets)
        
        error_message = str(exc_info.value)
        assert "Cannot use both isotropic ('bao') and anisotropic ('bao_ani')" in error_message
        assert "standard cosmological practice" in error_message
        assert str(datasets) in error_message
    
    def test_invalid_minimal_mixed_bao(self):
        """Test that even minimal mixing of bao types is invalid."""
        datasets = ["bao", "bao_ani"]
        
        with pytest.raises(BAOValidationError) as exc_info:
            validate_bao_dataset_separation(datasets)
        
        error_message = str(exc_info.value)
        assert "Cannot use both isotropic" in error_message
        assert "anisotropic" in error_message


class TestJointFitConfiguration:
    """Test joint fit configuration validation."""
    
    def test_valid_standard_joint_fit(self):
        """Test standard joint fit configuration."""
        datasets = ["cmb", "bao", "sn"]
        result = validate_joint_fit_configuration(datasets)
        
        assert result["status"] == "valid"
        assert result["dataset_separation"] == "proper"
        assert "Excellent dataset combination" in str(result["recommendations"])
    
    def test_anisotropic_bao_joint_fit_warning(self):
        """Test that anisotropic BAO in joint fits generates warnings."""
        datasets = ["cmb", "bao_ani", "sn"]
        result = validate_joint_fit_configuration(datasets)
        
        assert result["status"] == "valid"
        assert len(result["warnings"]) > 0
        assert "typically analyzed independently" in result["warnings"][0]
        assert "fit_bao_aniso.py" in str(result["recommendations"])
    
    def test_single_dataset_recommendations(self):
        """Test recommendations for single dataset configurations."""
        # Test BAO only
        datasets = ["bao"]
        result = validate_joint_fit_configuration(datasets)
        
        assert result["status"] == "valid"
        assert any("CMB and/or SN" in rec for rec in result["recommendations"])
        
        # Test CMB only
        datasets = ["cmb"]
        result = validate_joint_fit_configuration(datasets)
        
        assert result["status"] == "valid"
        assert any("BAO or SN" in rec for rec in result["recommendations"])
    
    def test_invalid_mixed_bao_configuration(self):
        """Test that mixed BAO configuration is invalid."""
        datasets = ["bao", "bao_ani"]
        
        with pytest.raises(BAOValidationError):
            validate_joint_fit_configuration(datasets)
    
    def test_standalone_anisotropic_bao(self):
        """Test standalone anisotropic BAO configuration."""
        datasets = ["bao_ani"]
        result = validate_joint_fit_configuration(datasets)
        
        assert result["status"] == "valid"
        assert result["dataset_separation"] == "proper"
        # Should not have warnings for standalone anisotropic BAO
        assert len(result["warnings"]) == 0


class TestDatasetSelectionGuide:
    """Test dataset selection guidance functionality."""
    
    def test_get_bao_dataset_selection_guide(self):
        """Test that dataset selection guide provides comprehensive information."""
        guide = get_bao_dataset_selection_guide()
        
        # Check that all expected sections are present
        expected_sections = [
            "isotropic_bao",
            "anisotropic_bao", 
            "joint_fit_recommendations",
            "avoid"
        ]
        
        for section in expected_sections:
            assert section in guide
            assert len(guide[section]) > 0
        
        # Check content quality
        assert "Joint fits with CMB" in guide["isotropic_bao"]
        assert "Dedicated anisotropic BAO analysis" in guide["anisotropic_bao"]
        assert "['cmb', 'bao', 'sn']" in guide["joint_fit_recommendations"]
        assert "['bao', 'bao_ani']" in guide["avoid"]
    
    def test_dataset_selection_guide_completeness(self):
        """Test that guide covers all important use cases."""
        guide = get_bao_dataset_selection_guide()
        
        # Should mention key concepts
        guide_text = " ".join(guide.values())
        
        important_concepts = [
            "joint fits",
            "anisotropic",
            "isotropic", 
            "CMB",
            "SN",
            "Double-counting",
            "standalone"
        ]
        
        for concept in important_concepts:
            assert concept.lower() in guide_text.lower()


class TestDatasetConfigurationValidator:
    """Test dataset configuration validator factory."""
    
    def test_create_validator_function(self):
        """Test that validator function is created correctly."""
        validator = create_dataset_configuration_validator()
        
        # Test valid configurations
        assert validator(["cmb", "bao", "sn"]) == True
        assert validator(["bao_ani"]) == True
        assert validator(["cmb", "sn"]) == True
        
        # Test invalid configurations
        assert validator(["bao", "bao_ani"]) == False
        assert validator(["cmb", "bao", "bao_ani", "sn"]) == False
    
    def test_validator_function_type(self):
        """Test that validator returns correct types."""
        validator = create_dataset_configuration_validator()
        
        result_valid = validator(["cmb", "bao"])
        result_invalid = validator(["bao", "bao_ani"])
        
        assert isinstance(result_valid, bool)
        assert isinstance(result_invalid, bool)
        assert result_valid == True
        assert result_invalid == False


class TestIntegrationWithEngine:
    """Test integration with the fitting engine."""
    
    def test_engine_validation_import(self):
        """Test that engine can import validation functions."""
        try:
            from fit_core.bao_aniso_validation import validate_bao_dataset_separation
            # If import succeeds, validation is available
            assert True
        except ImportError:
            pytest.fail("Engine should be able to import BAO validation functions")
    
    def test_validation_error_propagation(self):
        """Test that validation errors propagate correctly."""
        from fit_core.bao_aniso_validation import validate_bao_dataset_separation
        
        # This should raise BAOValidationError
        with pytest.raises(BAOValidationError):
            validate_bao_dataset_separation(["bao", "bao_ani"])


class TestWarningAndInformationalOutput:
    """Test warning and informational output functionality."""
    
    def test_print_dataset_separation_warning_mixed(self, capsys):
        """Test warning output for mixed BAO datasets."""
        datasets = ["bao", "bao_ani"]
        print_dataset_separation_warning(datasets)
        
        captured = capsys.readouterr()
        assert "DATASET SEPARATION WARNING" in captured.out
        assert "Both isotropic ('bao') and anisotropic ('bao_ani')" in captured.out
        assert "Standard practice:" in captured.out
    
    def test_print_dataset_separation_warning_aniso_joint(self, capsys):
        """Test info output for anisotropic BAO in joint fits."""
        datasets = ["cmb", "bao_ani", "sn"]
        print_dataset_separation_warning(datasets)
        
        captured = capsys.readouterr()
        assert "Anisotropic BAO in joint fit detected" in captured.out
        assert "fit_bao_aniso.py" in captured.out
    
    def test_print_dataset_separation_warning_valid(self, capsys):
        """Test no warning for valid configurations."""
        datasets = ["cmb", "bao", "sn"]
        print_dataset_separation_warning(datasets)
        
        captured = capsys.readouterr()
        # Should not print warnings for valid configurations
        assert "WARNING" not in captured.out
        assert "INFO" not in captured.out


class TestErrorMessages:
    """Test quality and informativeness of error messages."""
    
    def test_error_message_content(self):
        """Test that error messages are informative and actionable."""
        datasets = ["bao", "bao_ani"]
        
        with pytest.raises(BAOValidationError) as exc_info:
            validate_bao_dataset_separation(datasets)
        
        error_message = str(exc_info.value)
        
        # Should explain the problem
        assert "Cannot use both isotropic" in error_message
        assert "anisotropic" in error_message
        
        # Should explain why it's a problem
        assert "standard cosmological practice" in error_message
        assert "same physical scale" in error_message
        
        # Should provide solutions
        assert "Choose either:" in error_message
        assert "'bao' for isotropic" in error_message
        assert "'bao_ani' for anisotropic" in error_message
        
        # Should show current configuration
        assert str(datasets) in error_message
    
    def test_error_message_specificity(self):
        """Test that error messages are specific to the problem."""
        # Test with different invalid configurations
        test_cases = [
            ["bao", "bao_ani"],
            ["cmb", "bao", "bao_ani"],
            ["bao", "bao_ani", "sn"]
        ]
        
        for datasets in test_cases:
            with pytest.raises(BAOValidationError) as exc_info:
                validate_bao_dataset_separation(datasets)
            
            error_message = str(exc_info.value)
            # Should mention the specific datasets that caused the problem
            assert str(datasets) in error_message


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])