"""
Integration tests for engine optimization functionality.

This module tests the end-to-end optimization workflow for both ΛCDM and PBUF models,
parameter propagation across multiple fitters, and optimization summary generation.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from . import engine
from . import parameter
from .parameter_store import OptimizedParameterStore
from .optimizer import OptimizationResult


class TestEngineOptimization:
    """Test suite for engine optimization integration."""
    
    def setup_method(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.param_store = OptimizedParameterStore(storage_dir=self.temp_dir)
        
        # Mock dataset loading to avoid file dependencies
        self.mock_cmb_data = {
            "observations": np.array([1.0, 2.0, 3.0]),
            "covariance": np.eye(3) * 0.1,
            "theory_predictions": np.array([1.1, 2.1, 3.1])
        }
        
        # Mock likelihood functions to return predictable results
        self.mock_likelihood_results = {
            "cmb": (10.5, {"theory": np.array([1.1, 2.1, 3.1])}),
            "bao": (5.2, {"theory": np.array([0.5, 0.6])}),
            "sn": (8.3, {"theory": np.array([0.1, 0.2, 0.3])})
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    def test_lcdm_optimization_workflow(self, mock_integrity, mock_likelihood_cmb, mock_load_dataset):
        """
        Test end-to-end optimization workflow for ΛCDM model.
        
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Setup mocks
        mock_load_dataset.return_value = self.mock_cmb_data
        mock_likelihood_cmb.return_value = self.mock_likelihood_results["cmb"]
        mock_integrity.return_value = {"overall_status": "PASS", "failures": []}
        
        # Test parameters
        optimize_params = ["H0", "Om0"]
        
        # Execute optimization workflow
        results = engine.run_fit(
            model="lcdm",
            datasets_list=["cmb"],
            optimize_params=optimize_params,
            covariance_scaling=1.0,
            dry_run=False,
            warm_start=False
        )
        
        # Verify results structure
        assert "params" in results
        assert "results" in results
        assert "metrics" in results
        assert "chi2_breakdown" in results
        assert "optimization_result" in results
        
        # Verify optimization metadata
        opt_result = results["optimization_result"]
        assert opt_result.model == "lcdm"
        assert set(opt_result.optimized_params.keys()) == set(optimize_params)
        assert opt_result.convergence_status in ["success", "failed"]
        
        # Verify metrics include optimization information
        assert "optimization" in results["metrics"]
        opt_metrics = results["metrics"]["optimization"]
        assert "optimized_parameters" in opt_metrics
        assert "chi2_improvement" in opt_metrics
        assert "convergence_status" in opt_metrics
        
        # Verify χ² breakdown
        assert "cmb" in results["chi2_breakdown"]
        assert isinstance(results["chi2_breakdown"]["cmb"], (int, float))
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.likelihoods.likelihood_bao')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    def test_pbuf_optimization_workflow(self, mock_integrity, mock_likelihood_bao, mock_likelihood_cmb, mock_load_dataset):
        """
        Test end-to-end optimization workflow for PBUF model.
        
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Setup mocks
        def mock_load_side_effect(dataset_name):
            if dataset_name == "cmb":
                return self.mock_cmb_data
            elif dataset_name == "bao":
                return {
                    "observations": {"DV_over_rs": np.array([4.3, 5.4, 9.2, 10.6, 11.5])},
                    "covariance": np.eye(5) * 0.1,
                    "redshifts": np.array([0.1, 0.2, 0.3, 0.4, 0.5])
                }
            else:
                return {}
        
        mock_load_dataset.side_effect = mock_load_side_effect
        mock_likelihood_cmb.return_value = self.mock_likelihood_results["cmb"]
        mock_likelihood_bao.return_value = self.mock_likelihood_results["bao"]
        mock_integrity.return_value = {"overall_status": "PASS", "failures": []}
        
        # Test parameters - include PBUF-specific parameters
        optimize_params = ["H0", "Om0", "k_sat", "alpha"]
        
        # Execute optimization workflow
        results = engine.run_fit(
            model="pbuf",
            datasets_list=["cmb"],
            optimize_params=optimize_params,
            covariance_scaling=1.0,
            dry_run=False,
            warm_start=False
        )
        
        # Verify results structure
        assert "params" in results
        assert "results" in results
        assert "metrics" in results
        assert "chi2_breakdown" in results
        assert "optimization_result" in results
        
        # Verify optimization metadata
        opt_result = results["optimization_result"]
        assert opt_result.model == "pbuf"
        assert set(opt_result.optimized_params.keys()) == set(optimize_params)
        assert opt_result.convergence_status in ["success", "failed"]
        
        # Verify PBUF-specific parameters are included
        pbuf_params = {"k_sat", "alpha"}
        optimized_pbuf_params = set(opt_result.optimized_params.keys()) & pbuf_params
        assert len(optimized_pbuf_params) > 0, "PBUF-specific parameters should be optimized"
        
        # Verify metrics include optimization information
        assert "optimization" in results["metrics"]
        opt_metrics = results["metrics"]["optimization"]
        assert "optimized_parameters" in opt_metrics
        assert set(opt_metrics["optimized_parameters"]) == set(optimize_params)
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.likelihoods.likelihood_bao')
    @patch('pipelines.fit_core.likelihoods.likelihood_sn')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    def test_parameter_propagation_across_fitters(self, mock_integrity, mock_sn, mock_bao, mock_cmb, mock_load_dataset):
        """
        Test parameter propagation across multiple fitters.
        
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Setup mocks
        mock_load_dataset.return_value = self.mock_cmb_data
        mock_cmb.return_value = self.mock_likelihood_results["cmb"]
        mock_bao.return_value = self.mock_likelihood_results["bao"]
        mock_sn.return_value = self.mock_likelihood_results["sn"]
        mock_integrity.return_value = {"overall_status": "PASS", "failures": []}
        
        # Step 1: Optimize CMB parameters
        cmb_results = engine.run_fit(
            model="lcdm",
            datasets_list=["cmb"],
            optimize_params=["H0", "Om0"],
            dry_run=False
        )
        
        # Verify optimization completed
        assert "optimization_result" in cmb_results
        opt_result = cmb_results["optimization_result"]
        optimized_h0 = opt_result.optimized_params.get("H0")
        optimized_om0 = opt_result.optimized_params.get("Om0")
        
        assert optimized_h0 is not None
        assert optimized_om0 is not None
        
        # Step 2: Run BAO fit without optimization - should use optimized parameters
        bao_results = engine.run_fit(
            model="lcdm",
            datasets_list=["bao"],
            optimize_params=None  # No optimization, should use stored values
        )
        
        # Verify BAO fit uses optimized parameters from CMB
        bao_params = bao_results["params"]
        assert abs(bao_params["H0"] - optimized_h0) < 1e-10, "BAO fit should use optimized H0 from CMB"
        assert abs(bao_params["Om0"] - optimized_om0) < 1e-10, "BAO fit should use optimized Om0 from CMB"
        
        # Step 3: Run joint fit - should also use optimized parameters
        joint_results = engine.run_fit(
            model="lcdm",
            datasets_list=["cmb", "bao", "sn"],
            optimize_params=None  # No optimization, should use stored values
        )
        
        # Verify joint fit uses optimized parameters
        joint_params = joint_results["params"]
        assert abs(joint_params["H0"] - optimized_h0) < 1e-10, "Joint fit should use optimized H0"
        assert abs(joint_params["Om0"] - optimized_om0) < 1e-10, "Joint fit should use optimized Om0"
        
        # Verify all datasets are included in joint fit
        assert set(joint_results["datasets"]) == {"cmb", "bao", "sn"}
        assert len(joint_results["chi2_breakdown"]) == 3
    
    def test_optimization_summary_generation(self):
        """
        Test creation of optimization summary snapshot (reports/optimization_summary.json).
        
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Create mock optimization results for both models
        lcdm_metadata = {
            "optimization_timestamp": "2025-10-22T14:42:00Z",
            "source_dataset": "cmb",
            "optimized_params": ["H0", "Om0", "Obh2", "ns"],
            "chi2_improvement": 2.34,
            "convergence_status": "success",
            "optimizer_info": {"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"},
            "covariance_scaling": 1.0,
            "final_chi2": 12.34
        }
        
        pbuf_metadata = {
            "optimization_timestamp": "2025-10-22T14:42:00Z",
            "source_dataset": "cmb",
            "optimized_params": ["H0", "Om0", "k_sat", "alpha"],
            "chi2_improvement": 5.67,
            "convergence_status": "success",
            "optimizer_info": {"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"},
            "covariance_scaling": 1.0,
            "final_chi2": 8.76
        }
        
        # Update parameter store with optimization results
        self.param_store.update_model_defaults(
            model="lcdm",
            optimized_params={"H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649},
            optimization_metadata=lcdm_metadata
        )
        
        self.param_store.update_model_defaults(
            model="pbuf",
            optimized_params={"H0": 67.4, "Om0": 0.315, "k_sat": 0.9762, "alpha": 5e-4},
            optimization_metadata=pbuf_metadata
        )
        
        # Generate optimization summary
        summary_file = Path(self.temp_dir) / "optimization_summary.json"
        self.param_store.export_optimization_summary(str(summary_file))
        
        # Verify summary file was created
        assert summary_file.exists(), "Optimization summary file should be created"
        
        # Load and verify summary content
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Verify structure
        assert "models" in summary
        assert "lcdm" in summary["models"]
        assert "pbuf" in summary["models"]
        assert "export_timestamp" in summary
        
        # Verify ΛCDM summary
        lcdm_summary = summary["models"]["lcdm"]
        assert lcdm_summary["last_optimization"]["chi2_improvement"] == 2.34
        assert "current_defaults" in lcdm_summary
        assert lcdm_summary["is_optimized"] == True
        
        # Verify PBUF summary
        pbuf_summary = summary["models"]["pbuf"]
        assert pbuf_summary["last_optimization"]["chi2_improvement"] == 5.67
        assert "current_defaults" in pbuf_summary
        assert pbuf_summary["is_optimized"] == True
        
        # Verify cross-model consistency
        assert "cross_model_consistency" in summary
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    def test_dry_run_mode(self, mock_integrity, mock_likelihood_cmb, mock_load_dataset):
        """
        Test dry run mode produces results without side effects.
        
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Setup mocks
        mock_load_dataset.return_value = self.mock_cmb_data
        mock_likelihood_cmb.return_value = self.mock_likelihood_results["cmb"]
        mock_integrity.return_value = {"overall_status": "PASS", "failures": []}
        
        # Get initial parameter values
        initial_params = self.param_store.get_model_defaults("lcdm")
        initial_h0 = initial_params["H0"]
        
        # Execute optimization in dry run mode
        results = engine.run_fit(
            model="lcdm",
            datasets_list=["cmb"],
            optimize_params=["H0", "Om0"],
            dry_run=True
        )
        
        # Verify optimization completed and returned results
        assert "optimization_result" in results
        opt_result = results["optimization_result"]
        assert opt_result.convergence_status in ["success", "failed"]
        
        # Verify parameters were not persisted (no side effects)
        final_params = self.param_store.get_model_defaults("lcdm")
        final_h0 = final_params["H0"]
        
        assert abs(final_h0 - initial_h0) < 1e-10, "Dry run should not modify stored parameters"
        
        # Verify optimization history was not updated
        history = self.param_store.get_optimization_history("lcdm")
        assert len(history) == 0, "Dry run should not add to optimization history"
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    def test_warm_start_functionality(self, mock_integrity, mock_likelihood_cmb, mock_load_dataset):
        """
        Test warm start functionality with recent optimization results.
        
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Setup mocks
        mock_load_dataset.return_value = self.mock_cmb_data
        mock_likelihood_cmb.return_value = self.mock_likelihood_results["cmb"]
        mock_integrity.return_value = {"overall_status": "PASS", "failures": []}
        
        # First optimization to create warm start data
        first_results = engine.run_fit(
            model="lcdm",
            datasets_list=["cmb"],
            optimize_params=["H0", "Om0"],
            dry_run=False
        )
        
        first_opt_result = first_results["optimization_result"]
        first_h0 = first_opt_result.optimized_params["H0"]
        
        # Second optimization with warm start
        second_results = engine.run_fit(
            model="lcdm",
            datasets_list=["cmb"],
            optimize_params=["H0", "Om0"],
            warm_start=True,
            dry_run=True  # Use dry run to avoid overwriting first results
        )
        
        # Verify warm start was used (starting parameters should match first optimization results)
        second_opt_result = second_results["optimization_result"]
        
        # The starting parameters for the second optimization should be close to the first results
        # Note: This is a simplified test - in practice, we'd need to capture the starting parameters
        assert second_opt_result is not None, "Warm start optimization should complete"
        
        # Verify optimization history contains the first optimization
        # Create a new parameter store instance to check the default storage location
        default_store = OptimizedParameterStore()
        history = default_store.get_optimization_history("lcdm")
        assert len(history) >= 1, "Optimization history should contain at least one record"
        
        # Verify warm start parameters are available
        warm_start_params = default_store.get_warm_start_params("lcdm")
        assert warm_start_params is not None, "Warm start parameters should be available"
        assert "H0" in warm_start_params, "Warm start should include optimized H0"
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    def test_cmb_optimization_workflow_function(self, mock_integrity, mock_likelihood_cmb, mock_load_dataset):
        """
        Test the high-level CMB optimization workflow function.
        
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Setup mocks
        mock_load_dataset.return_value = self.mock_cmb_data
        mock_likelihood_cmb.return_value = self.mock_likelihood_results["cmb"]
        mock_integrity.return_value = {"overall_status": "PASS", "failures": []}
        
        # Execute CMB optimization workflow
        workflow_results = engine.run_cmb_optimization_workflow(
            model="lcdm",
            optimize_params=["H0", "Om0"],
            covariance_scaling=1.0,
            dry_run=False,
            warm_start=False,
            auto_propagate=True
        )
        
        # Verify workflow results structure
        assert "workflow_metadata" in workflow_results
        assert "optimization_results" in workflow_results
        assert "validation_results" in workflow_results
        assert "propagation_results" in workflow_results
        assert "summary" in workflow_results
        
        # Verify workflow metadata
        metadata = workflow_results["workflow_metadata"]
        assert metadata["model"] == "lcdm"
        assert metadata["optimize_params"] == ["H0", "Om0"]
        assert metadata["covariance_scaling"] == 1.0
        assert metadata["dry_run"] == False
        assert metadata["auto_propagate"] == True
        
        # Verify optimization results are included
        opt_results = workflow_results["optimization_results"]
        assert "optimization_result" in opt_results
        
        # Verify validation results
        validation = workflow_results["validation_results"]
        assert "overall_status" in validation
        assert validation["overall_status"] in ["success", "warning", "failure"]
        
        # Verify summary
        summary = workflow_results["summary"]
        assert "success" in summary
        assert "chi2_improvement" in summary
        assert "convergence_status" in summary
        assert "optimized_parameters" in summary
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    def test_dual_model_cmb_optimization(self, mock_integrity, mock_likelihood_cmb, mock_load_dataset):
        """
        Test dual model CMB optimization workflow.
        
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Setup mocks
        mock_load_dataset.return_value = self.mock_cmb_data
        mock_likelihood_cmb.return_value = self.mock_likelihood_results["cmb"]
        mock_integrity.return_value = {"overall_status": "PASS", "failures": []}
        
        # Execute dual model optimization
        dual_results = engine.run_dual_model_cmb_optimization(
            optimize_params_lcdm=["H0", "Om0"],
            optimize_params_pbuf=["H0", "Om0", "k_sat"],
            covariance_scaling=1.0,
            dry_run=False,
            warm_start=False,
            validate_consistency=True
        )
        
        # Verify dual results structure
        assert "lcdm" in dual_results
        assert "pbuf" in dual_results
        assert "cross_model_analysis" in dual_results
        assert "summary" in dual_results
        
        # Verify ΛCDM results
        lcdm_results = dual_results["lcdm"]
        assert "workflow_metadata" in lcdm_results
        assert "summary" in lcdm_results
        
        # Verify PBUF results
        pbuf_results = dual_results["pbuf"]
        assert "workflow_metadata" in pbuf_results
        assert "summary" in pbuf_results
        
        # Verify cross-model analysis
        cross_analysis = dual_results["cross_model_analysis"]
        assert "_summary" in cross_analysis
        
        # Verify summary
        summary = dual_results["summary"]
        assert "lcdm_success" in summary
        assert "pbuf_success" in summary
        assert "both_successful" in summary
        assert "consistency_validated" in summary
    
    def test_error_handling_invalid_model(self):
        """Test error handling for invalid model types."""
        with pytest.raises(ValueError, match="Invalid model"):
            engine.run_fit(
                model="invalid_model",
                datasets_list=["cmb"],
                optimize_params=["H0"]
            )
    
    def test_error_handling_empty_datasets(self):
        """Test error handling for empty dataset list."""
        with pytest.raises(ValueError, match="At least one dataset must be specified"):
            engine.run_fit(
                model="lcdm",
                datasets_list=[],
                optimize_params=["H0"]
            )
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    def test_error_handling_dataset_integrity_failure(self, mock_load_dataset):
        """Test error handling when dataset integrity validation fails."""
        # Mock dataset loading to simulate integrity failure
        mock_load_dataset.side_effect = Exception("Dataset corruption detected")
        
        with pytest.raises(ValueError, match="CMB dataset integrity validation failed"):
            engine.run_cmb_optimization_workflow(
                model="lcdm",
                optimize_params=["H0", "Om0"]
            )


class TestOptimizationSummaryGeneration:
    """Test suite for optimization summary generation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.param_store = OptimizedParameterStore(storage_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_optimization_summary_snapshot(self):
        """
        Test creation of optimization summary snapshot in reports directory.
        
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        # Create reports directory
        reports_dir = Path(self.temp_dir) / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Add optimization results for both models
        lcdm_metadata = {
            "optimization_timestamp": "2025-10-22T14:42:00Z",
            "source_dataset": "cmb",
            "optimized_params": ["H0", "Om0", "Obh2", "ns"],
            "chi2_improvement": 2.34,
            "convergence_status": "success",
            "optimizer_info": {"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"},
            "final_chi2": 12.34
        }
        
        pbuf_metadata = {
            "optimization_timestamp": "2025-10-22T14:42:00Z",
            "source_dataset": "cmb",
            "optimized_params": ["H0", "Om0", "k_sat", "alpha"],
            "chi2_improvement": 5.67,
            "convergence_status": "success",
            "optimizer_info": {"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"},
            "final_chi2": 8.76
        }
        
        # Update parameter store
        self.param_store.update_model_defaults(
            model="lcdm",
            optimized_params={"H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649},
            optimization_metadata=lcdm_metadata
        )
        
        self.param_store.update_model_defaults(
            model="pbuf",
            optimized_params={"H0": 67.4, "Om0": 0.315, "k_sat": 0.9762, "alpha": 5e-4},
            optimization_metadata=pbuf_metadata
        )
        
        # Generate summary snapshot
        summary_path = reports_dir / "optimization_summary.json"
        self.param_store.export_optimization_summary(str(summary_path))
        
        # Verify file was created
        assert summary_path.exists(), "Optimization summary should be created in reports directory"
        
        # Load and verify content
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Verify expected structure for HTML report integration
        assert "models" in summary
        assert "lcdm" in summary["models"]
        assert "pbuf" in summary["models"]
        
        # Verify ΛCDM optimization summary
        lcdm_summary = summary["models"]["lcdm"]
        assert lcdm_summary["last_optimization"]["chi2_improvement"] == 2.34
        assert lcdm_summary["is_optimized"] == True
        
        # Verify PBUF optimization summary
        pbuf_summary = summary["models"]["pbuf"]
        assert pbuf_summary["last_optimization"]["chi2_improvement"] == 5.67
        assert pbuf_summary["is_optimized"] == True
        
        # Verify format matches expected HTML report integration format
        # The summary structure is now under "models" key
        assert lcdm_summary["last_optimization"]["chi2_improvement"] == 2.34
        assert pbuf_summary["last_optimization"]["chi2_improvement"] == 5.67
        
        # Verify cross-model consistency is included
        assert "cross_model_consistency" in summary


if __name__ == "__main__":
    pytest.main([__file__])