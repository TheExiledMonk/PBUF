"""
Tests for advanced optimization features: dry-run, warm-start, and provenance logging.

This module tests the advanced optimization capabilities including dry-run mode,
warm-start functionality, comprehensive provenance logging, and HTML report integration.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from .optimizer import ParameterOptimizer, optimize_cmb_parameters, OptimizationResult
from .parameter_store import OptimizedParameterStore, OptimizationRecord
from . import parameter


class TestDryRunMode:
    """Test dry-run mode functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = OptimizedParameterStore(self.temp_dir)
        self.optimizer = ParameterOptimizer()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dry_run_produces_results_without_side_effects(self):
        """
        Test that dry-run mode performs optimization but doesn't persist results.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Get initial state
        initial_lcdm_defaults = self.store.get_model_defaults("lcdm")
        initial_history_count = len(self.store.get_optimization_history("lcdm"))
        
        # Mock the optimization objective to return predictable results
        with patch.object(self.optimizer, '_build_optimization_objective') as mock_objective:
            # Create a simple quadratic objective that has a minimum
            def simple_objective(params):
                return sum((p - 0.5) ** 2 for p in params)
            
            mock_objective.return_value = simple_objective
            
            # Run optimization in dry-run mode
            result = self.optimizer.optimize_parameters(
                model="lcdm",
                datasets_list=["cmb"],
                optimize_params=["H0", "Om0"],
                dry_run=True
            )
            
            # Verify optimization completed successfully
            assert result.convergence_status == "success"
            assert result.metadata["dry_run_mode"] is True
            assert result.metadata["execution_mode"] == "dry_run"
            assert len(result.optimized_params) == 2
            
            # Verify no side effects occurred
            final_lcdm_defaults = self.store.get_model_defaults("lcdm")
            final_history_count = len(self.store.get_optimization_history("lcdm"))
            
            # Defaults should be unchanged
            assert final_lcdm_defaults == initial_lcdm_defaults
            
            # History should be unchanged
            assert final_history_count == initial_history_count
            
            # Verify dry-run metadata is recorded
            assert "dry_run_mode" in result.metadata
            assert result.metadata["dry_run_mode"] is True
    
    def test_dry_run_parameter_store_update(self):
        """
        Test that parameter store dry-run mode validates without persisting.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Test dry-run update (should not raise errors but not persist)
        optimized_params = {"H0": 70.0, "Om0": 0.3}
        metadata = {"test": "dry_run_metadata"}
        
        # This should not raise an error
        self.store.update_model_defaults("lcdm", optimized_params, metadata, dry_run=True)
        
        # Verify no changes were persisted
        current_defaults = self.store.get_model_defaults("lcdm")
        assert current_defaults["H0"] != 70.0  # Should still be original value
        assert current_defaults["Om0"] != 0.3   # Should still be original value
    
    def test_dry_run_with_invalid_parameters(self):
        """
        Test that dry-run mode properly validates parameters without side effects.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Test dry-run with invalid parameters
        invalid_params = {"invalid_param": 1.0}
        metadata = {"test": "invalid_dry_run"}
        
        with pytest.raises(ValueError, match="not optimizable"):
            self.store.update_model_defaults("lcdm", invalid_params, metadata, dry_run=True)
        
        # Verify no changes occurred
        history = self.store.get_optimization_history("lcdm")
        assert len(history) == 0


class TestWarmStartFunctionality:
    """Test warm-start functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = OptimizedParameterStore(self.temp_dir)
        self.optimizer = ParameterOptimizer()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_warm_start_with_recent_optimization(self):
        """
        Test warm-start functionality with recent optimization results.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Create a recent optimization record
        recent_timestamp = datetime.now(timezone.utc).isoformat()
        optimization_record = OptimizationRecord(
            timestamp=recent_timestamp,
            model="lcdm",
            dataset="cmb",
            optimized_params=["H0", "Om0"],
            final_values={"H0": 70.0, "Om0": 0.3},
            chi2_improvement=5.0,
            convergence_status="success",
            optimizer_info={"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"}
        )
        
        # Add to history manually
        self.store._add_to_history("lcdm", {"H0": 70.0, "Om0": 0.3}, {
            "timestamp": recent_timestamp,
            "dataset": "cmb",
            "chi2_improvement": 5.0,
            "convergence_status": "success",
            "optimizer_info": {"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"}
        })
        
        # Test warm-start parameter retrieval
        warm_start_params = self.store.get_warm_start_params("lcdm", max_age_hours=24.0)
        
        assert warm_start_params is not None
        assert warm_start_params["H0"] == 70.0
        assert warm_start_params["Om0"] == 0.3
    
    def test_warm_start_with_old_optimization(self):
        """
        Test warm-start functionality with old optimization results (should not use).
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Create an old optimization record (older than max age)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        
        self.store._add_to_history("lcdm", {"H0": 70.0, "Om0": 0.3}, {
            "timestamp": old_timestamp,
            "dataset": "cmb",
            "chi2_improvement": 5.0,
            "convergence_status": "success",
            "optimizer_info": {"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"}
        })
        
        # Test warm-start parameter retrieval with 24-hour limit
        warm_start_params = self.store.get_warm_start_params("lcdm", max_age_hours=24.0)
        
        # Should return None because optimization is too old
        assert warm_start_params is None
    
    def test_warm_start_parameter_compatibility_validation(self):
        """
        Test validation of warm-start parameter compatibility.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        warm_start_params = {"H0": 70.0, "Om0": 0.3, "Obh2": 0.022}
        optimize_params = ["H0", "Om0"]  # Missing Obh2, has extra parameter
        
        compatibility = self.optimizer._validate_warm_start_compatibility(
            "lcdm", warm_start_params, optimize_params
        )
        
        assert compatibility["is_compatible"] is True  # Should still be compatible
        assert "H0" in compatibility["parameter_overlap"]
        assert "Om0" in compatibility["parameter_overlap"]
        assert "Obh2" in compatibility["extra_parameters"]
        assert len(compatibility["missing_parameters"]) == 0
    
    def test_warm_start_with_out_of_bounds_parameters(self):
        """
        Test warm-start validation with out-of-bounds parameters.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Create parameters that are outside valid bounds
        out_of_bounds_params = {"H0": 200.0}  # H0 should be bounded [20, 150]
        optimize_params = ["H0"]
        
        compatibility = self.optimizer._validate_warm_start_compatibility(
            "lcdm", out_of_bounds_params, optimize_params
        )
        
        assert compatibility["is_compatible"] is False
        assert any("outside bounds" in warning for warning in compatibility["warnings"])
    
    @patch('pipelines.fit_core.optimizer.parameter.get_defaults')
    def test_get_starting_parameters_with_warm_start(self, mock_get_defaults):
        """
        Test _get_starting_parameters method with warm-start enabled.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Mock default parameters
        mock_get_defaults.return_value = {"H0": 67.4, "Om0": 0.315, "Obh2": 0.02237}
        
        # Mock warm-start parameters
        with patch('pipelines.fit_core.parameter_store.OptimizedParameterStore') as mock_store_class:
            mock_store_instance = mock_store_class.return_value
            mock_store_instance.get_warm_start_params.return_value = {"H0": 70.0, "Om0": 0.3}
            
            # Test with warm-start enabled
            starting_params = self.optimizer._get_starting_parameters(
                model="lcdm",
                starting_values=None,
                warm_start=True,
                warm_start_max_age_hours=24.0
            )
            
            # Should use warm-start values where available
            assert starting_params["H0"] == 70.0  # From warm-start
            assert starting_params["Om0"] == 0.3   # From warm-start
            assert starting_params["Obh2"] == 0.02237  # From defaults (not in warm-start)
    
    def test_get_starting_parameters_explicit_override(self):
        """
        Test that explicit starting values override warm-start values.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        with patch('pipelines.fit_core.optimizer.parameter.get_defaults') as mock_defaults:
            mock_defaults.return_value = {"H0": 67.4, "Om0": 0.315}
            
            with patch('pipelines.fit_core.parameter_store.OptimizedParameterStore') as mock_store_class:
                mock_store_instance = mock_store_class.return_value
                mock_store_instance.get_warm_start_params.return_value = {"H0": 70.0, "Om0": 0.3}
                
                # Test with explicit starting values (should override warm-start)
                starting_params = self.optimizer._get_starting_parameters(
                    model="lcdm",
                    starting_values={"H0": 75.0},  # Explicit override
                    warm_start=True,
                    warm_start_max_age_hours=24.0
                )
                
                # Explicit value should take precedence
                assert starting_params["H0"] == 75.0  # From explicit override
                assert starting_params["Om0"] == 0.3   # From warm-start


class TestProvenanceLogging:
    """Test comprehensive provenance logging functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.optimizer = ParameterOptimizer()
    
    def test_optimizer_provenance_completeness(self):
        """
        Test that optimizer provenance includes all required information.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        optimizer_config = {
            "method": "L-BFGS-B",
            "options": {"maxiter": 1000, "ftol": 1e-9}
        }
        
        provenance = self.optimizer._build_optimizer_provenance(optimizer_config)
        
        # Check required provenance fields
        required_fields = [
            "method", "library", "scipy_version", "numpy_version", 
            "python_version", "platform", "architecture", "processor",
            "config_checksum", "covariance_scaling", "provenance_timestamp"
        ]
        
        for field in required_fields:
            assert field in provenance, f"Missing provenance field: {field}"
        
        # Verify specific values
        assert provenance["method"] == "L-BFGS-B"
        assert provenance["library"] == "scipy"
        assert provenance["covariance_scaling"] == "1.0"
        assert len(provenance["config_checksum"]) == 8  # MD5 hash truncated to 8 chars
    
    def test_optimization_metadata_completeness(self):
        """
        Test that optimization metadata includes all required information.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        datasets_list = ["cmb"]
        initial_chi2 = 100.0
        optimizer_config = {"method": "L-BFGS-B"}
        starting_params = {"H0": 67.4, "Om0": 0.315}
        optimize_params = ["H0", "Om0"]
        
        metadata = self.optimizer._build_optimization_metadata(
            datasets_list, initial_chi2, optimizer_config, 
            starting_params, optimize_params, dry_run=True, warm_start=True
        )
        
        # Check required metadata fields
        required_fields = [
            "datasets", "dataset_checksum", "initial_chi2", "optimization_config",
            "starting_param_checksum", "optimized_param_list", "optimization_timestamp",
            "session_id", "covariance_scaling_applied", "dry_run_mode", "warm_start_enabled",
            "execution_mode", "reproducibility_info"
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"
        
        # Verify specific values
        assert metadata["datasets"] == ["cmb"]
        assert metadata["initial_chi2"] == 100.0
        assert metadata["dry_run_mode"] is True
        assert metadata["warm_start_enabled"] is True
        assert metadata["execution_mode"] == "dry_run"
        assert len(metadata["session_id"]) == 12
        assert len(metadata["dataset_checksum"]) == 8
        assert len(metadata["starting_param_checksum"]) == 8
    
    def test_provenance_logging_output(self, capsys):
        """
        Test that provenance logging produces comprehensive output.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Create a mock optimization result
        result = OptimizationResult(
            model="lcdm",
            optimized_params={"H0": 70.0, "Om0": 0.3},
            starting_params={"H0": 67.4, "Om0": 0.315},
            final_chi2=95.0,
            chi2_improvement=5.0,
            convergence_status="success",
            n_function_evaluations=150,
            optimization_time=2.5,
            bounds_reached=[],
            optimizer_info={
                "method": "L-BFGS-B",
                "library": "scipy",
                "scipy_version": "1.13.0",
                "numpy_version": "1.24.0",
                "python_version": "3.9.0",
                "platform": "Linux-test",
                "config_checksum": "abc12345"
            },
            covariance_scaling=1.0,
            metadata={
                "session_id": "test_session_123",
                "optimization_timestamp": "2025-10-22T14:42:00Z",
                "dry_run_mode": True,
                "warm_start_enabled": True,
                "execution_mode": "dry_run",
                "reproducibility_info": {
                    "parameter_count": 2,
                    "dataset_count": 1
                }
            }
        )
        
        # Test provenance logging
        self.optimizer.log_optimization_provenance(result)
        
        # Capture output
        captured = capsys.readouterr()
        output = captured.out
        
        # Verify key provenance information is logged
        assert "[PROVENANCE] Optimization Session Report" in output
        assert "Session ID: test_session_123" in output
        assert "Timestamp: 2025-10-22T14:42:00Z" in output
        assert "Execution mode: dry_run" in output
        assert "[DRY_RUN] Mode enabled" in output
        assert "[WARM_START] Mode enabled" in output
        assert "Optimizer: L-BFGS-B (scipy v1.13.0)" in output
        assert "Environment: Python 3.9.0 on Linux-test" in output
        assert "Config checksum: abc12345" in output
        assert "[CONVERGENCE] Status: success" in output
        assert "[PARAMETERS] Optimized 2 parameters:" in output
        assert "H0: 67.40000000 → 70.00000000" in output
        assert "[REPRODUCIBILITY] Parameter count: 2" in output


class TestHTMLReportIntegration:
    """Test HTML report integration functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = OptimizedParameterStore(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_optimization_summary_export(self):
        """
        Test that optimization summary export creates valid JSON.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Create test optimization data
        self.store.update_model_defaults("lcdm", {"H0": 70.0}, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chi2_improvement": 3.0,
            "convergence_status": "success"
        })
        
        # Export summary
        summary_path = Path(self.temp_dir) / "test_summary.json"
        self.store.export_optimization_summary(str(summary_path))
        
        # Verify file was created and is valid JSON
        assert summary_path.exists()
        
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        
        # Check required structure
        required_sections = ["export_timestamp", "models", "cross_model_consistency", 
                           "optimization_history", "summary_statistics"]
        for section in required_sections:
            assert section in summary_data
        
        # Check model data
        assert "lcdm" in summary_data["models"]
        assert "pbuf" in summary_data["models"]
    
    def test_html_optimization_section_generation(self):
        """
        Test HTML optimization section data generation.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Add some optimization data
        self.store.update_model_defaults("lcdm", {"H0": 70.0, "Om0": 0.3}, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chi2_improvement": 5.0,
            "convergence_status": "success",
            "optimizer_info": {"method": "L-BFGS-B"}
        })
        
        # Generate HTML data
        html_data = self.store.get_html_optimization_section()
        
        # Verify structure
        assert html_data["optimization_enabled"] is True
        assert "models" in html_data
        assert "cross_model_status" in html_data
        assert "summary_stats" in html_data
        
        # Check LCDM model data
        lcdm_data = html_data["models"]["lcdm"]
        assert lcdm_data["status"] == "optimized"
        assert lcdm_data["chi2_improvement"] == 5.0
        assert lcdm_data["convergence_status"] == "success"
    
    def test_html_optimization_consistency_validation(self):
        """
        Test validation that HTML data matches optimization summary JSON.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Create optimization data
        self.store.update_model_defaults("lcdm", {"H0": 70.0}, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chi2_improvement": 3.0,
            "convergence_status": "success"
        })
        
        # Export summary and generate HTML data
        summary_path = Path(self.temp_dir) / "test_summary.json"
        self.store.export_optimization_summary(str(summary_path))
        html_data = self.store.get_html_optimization_section()
        
        # Validate consistency
        validation_result = self.store.validate_html_optimization_consistency(
            html_data, str(summary_path)
        )
        
        assert validation_result["is_consistent"] is True
        assert validation_result["summary_file_exists"] is True
        assert validation_result["html_data_valid"] is True
        assert len(validation_result["discrepancies"]) == 0
    
    def test_html_optimization_consistency_with_discrepancies(self):
        """
        Test validation detects discrepancies between HTML and JSON data.
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Create optimization data
        self.store.update_model_defaults("lcdm", {"H0": 70.0}, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chi2_improvement": 3.0,
            "convergence_status": "success"
        })
        
        # Export summary
        summary_path = Path(self.temp_dir) / "test_summary.json"
        self.store.export_optimization_summary(str(summary_path))
        
        # Create HTML data with intentional discrepancy
        html_data = {
            "optimization_enabled": True,
            "models": {
                "lcdm": {
                    "status": "optimized",
                    "chi2_improvement": 5.0,  # Different from JSON (3.0)
                    "optimized_params": ["H0"]
                },
                "pbuf": {"status": "default"}
            },
            "cross_model_status": "consistent",
            "summary_stats": {
                "models_optimized": 2,  # Different from actual (1)
                "total_chi2_improvement": 5.0
            }
        }
        
        # Validate consistency (should detect discrepancies)
        validation_result = self.store.validate_html_optimization_consistency(
            html_data, str(summary_path)
        )
        
        assert validation_result["is_consistent"] is False
        assert len(validation_result["discrepancies"]) > 0
        
        # Check specific discrepancies
        discrepancies_text = " ".join(validation_result["discrepancies"])
        assert "χ² improvement mismatch" in discrepancies_text
        assert "Models optimized count mismatch" in discrepancies_text


if __name__ == "__main__":
    pytest.main([__file__])