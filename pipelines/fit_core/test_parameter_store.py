"""
Unit tests for OptimizedParameterStore class.

Tests cover round-trip persistence, non-destructive parameter merging,
cross-model consistency validation, and error handling scenarios.
"""

import json
import os
import tempfile
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest

from .parameter_store import OptimizedParameterStore, OptimizationRecord, OptimizationResult
from .parameter import get_defaults


class TestOptimizedParameterStore:
    """Test suite for OptimizedParameterStore functionality."""
    
    def setup_method(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = OptimizedParameterStore(storage_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test parameter store initialization creates required files and directories."""
        # Check directory creation
        assert Path(self.temp_dir).exists()
        
        # Check file creation
        assert self.store.lcdm_file.exists()
        assert self.store.pbuf_file.exists()
        assert self.store.history_file.exists()
        
        # Check file structure
        with open(self.store.lcdm_file, 'r') as f:
            lcdm_data = json.load(f)
        assert "defaults" in lcdm_data
        assert "optimization_metadata" in lcdm_data
        
        with open(self.store.pbuf_file, 'r') as f:
            pbuf_data = json.load(f)
        assert "defaults" in pbuf_data
        assert "optimization_metadata" in pbuf_data
        
        with open(self.store.history_file, 'r') as f:
            history_data = json.load(f)
        assert isinstance(history_data, list)
    
    def test_get_model_defaults_lcdm(self):
        """Test retrieving ΛCDM model defaults."""
        defaults = self.store.get_model_defaults("lcdm")
        
        # Check required parameters are present
        required_params = {"H0", "Om0", "Obh2", "ns", "Neff", "Tcmb", "recomb_method"}
        assert required_params.issubset(set(defaults.keys()))
        
        # Check parameter types and ranges
        assert isinstance(defaults["H0"], (int, float))
        assert 20.0 <= defaults["H0"] <= 150.0
        assert isinstance(defaults["Om0"], (int, float))
        assert 0.01 <= defaults["Om0"] <= 0.99
    
    def test_get_model_defaults_pbuf(self):
        """Test retrieving PBUF model defaults."""
        defaults = self.store.get_model_defaults("pbuf")
        
        # Check ΛCDM parameters are present
        lcdm_params = {"H0", "Om0", "Obh2", "ns", "Neff", "Tcmb", "recomb_method"}
        assert lcdm_params.issubset(set(defaults.keys()))
        
        # Check PBUF-specific parameters are present
        pbuf_params = {"alpha", "Rmax", "eps0", "n_eps", "k_sat"}
        assert pbuf_params.issubset(set(defaults.keys()))
        
        # Check parameter types and ranges
        assert isinstance(defaults["k_sat"], (int, float))
        assert 0.1 <= defaults["k_sat"] <= 2.0
        assert isinstance(defaults["alpha"], (int, float))
        assert 1e-6 <= defaults["alpha"] <= 1e-2
    
    def test_get_model_defaults_invalid_model(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            self.store.get_model_defaults("invalid_model")
    
    def test_update_model_defaults_lcdm(self):
        """Test updating ΛCDM model defaults with optimized parameters."""
        # Get initial defaults
        initial_defaults = self.store.get_model_defaults("lcdm")
        initial_h0 = initial_defaults["H0"]
        
        # Update with optimized parameters
        optimized_params = {"H0": 70.0, "Om0": 0.3}
        metadata = {
            "source_dataset": "cmb",
            "chi2_improvement": 2.5,
            "convergence_status": "success",
            "optimizer_info": {"method": "L-BFGS-B", "library": "scipy"}
        }
        
        self.store.update_model_defaults("lcdm", optimized_params, metadata)
        
        # Verify update
        updated_defaults = self.store.get_model_defaults("lcdm")
        assert updated_defaults["H0"] == 70.0
        assert updated_defaults["Om0"] == 0.3
        
        # Verify non-destructive merge (other parameters unchanged)
        assert updated_defaults["Obh2"] == initial_defaults["Obh2"]
        assert updated_defaults["ns"] == initial_defaults["ns"]
        assert updated_defaults["Tcmb"] == initial_defaults["Tcmb"]
    
    def test_update_model_defaults_pbuf(self):
        """Test updating PBUF model defaults with optimized parameters."""
        # Get initial defaults
        initial_defaults = self.store.get_model_defaults("pbuf")
        initial_k_sat = initial_defaults["k_sat"]
        
        # Update with optimized parameters
        optimized_params = {"k_sat": 1.2, "alpha": 6e-4}
        metadata = {
            "source_dataset": "cmb",
            "chi2_improvement": 5.1,
            "convergence_status": "success"
        }
        
        self.store.update_model_defaults("pbuf", optimized_params, metadata)
        
        # Verify update
        updated_defaults = self.store.get_model_defaults("pbuf")
        assert updated_defaults["k_sat"] == 1.2
        assert updated_defaults["alpha"] == 6e-4
        
        # Verify non-destructive merge
        assert updated_defaults["H0"] == initial_defaults["H0"]
        assert updated_defaults["Rmax"] == initial_defaults["Rmax"]
    
    def test_update_model_defaults_invalid_parameter(self):
        """Test error handling for invalid optimization parameters."""
        optimized_params = {"invalid_param": 1.0}
        metadata = {"source_dataset": "cmb"}
        
        with pytest.raises(ValueError, match="not optimizable"):
            self.store.update_model_defaults("lcdm", optimized_params, metadata)
    
    def test_update_model_defaults_dry_run(self):
        """Test dry run mode doesn't persist changes."""
        initial_defaults = self.store.get_model_defaults("lcdm")
        
        # Dry run update
        optimized_params = {"H0": 75.0}
        metadata = {"source_dataset": "cmb"}
        
        self.store.update_model_defaults("lcdm", optimized_params, metadata, dry_run=True)
        
        # Verify no changes persisted
        current_defaults = self.store.get_model_defaults("lcdm")
        assert current_defaults["H0"] == initial_defaults["H0"]
    
    def test_round_trip_persistence(self):
        """Test critical round-trip persistence of optimized parameters."""
        # Original parameters
        original_params = self.store.get_model_defaults("pbuf")
        
        # Optimize parameters
        optimized_params = {
            "H0": 68.5,
            "Om0": 0.31,
            "k_sat": 1.1,
            "alpha": 5.5e-4
        }
        metadata = {
            "source_dataset": "cmb",
            "chi2_improvement": 3.2,
            "convergence_status": "success"
        }
        
        # Store optimized parameters
        self.store.update_model_defaults("pbuf", optimized_params, metadata)
        
        # Reload from storage
        reloaded_params = self.store.get_model_defaults("pbuf")
        
        # Verify exact numerical precision
        for param, expected_value in optimized_params.items():
            assert abs(reloaded_params[param] - expected_value) < 1e-12, \
                f"Parameter {param}: expected {expected_value}, got {reloaded_params[param]}"
        
        # Verify non-optimized parameters unchanged
        for param, original_value in original_params.items():
            if param not in optimized_params:
                assert reloaded_params[param] == original_value
    
    def test_optimization_history(self):
        """Test optimization history tracking."""
        # Initially empty
        history = self.store.get_optimization_history("lcdm")
        assert len(history) == 0
        
        # Add optimization result
        optimized_params = {"H0": 69.0, "Om0": 0.32}
        metadata = {
            "source_dataset": "cmb",
            "chi2_improvement": 1.8,
            "convergence_status": "success",
            "optimizer_info": {"method": "L-BFGS-B"}
        }
        
        self.store.update_model_defaults("lcdm", optimized_params, metadata)
        
        # Check history
        history = self.store.get_optimization_history("lcdm")
        assert len(history) == 1
        
        record = history[0]
        assert record.model == "lcdm"
        assert record.dataset == "cmb"
        assert record.convergence_status == "success"
        assert record.chi2_improvement == 1.8
        assert set(record.optimized_params) == {"H0", "Om0"}
        assert record.final_values["H0"] == 69.0
        assert record.final_values["Om0"] == 0.32
    
    def test_warm_start_params(self):
        """Test warm-start parameter retrieval."""
        # No warm start initially
        warm_params = self.store.get_warm_start_params("pbuf")
        assert warm_params is None
        
        # Add recent optimization
        optimized_params = {"k_sat": 0.95, "alpha": 4.8e-4}
        metadata = {
            "source_dataset": "cmb",
            "chi2_improvement": 2.1,
            "convergence_status": "success"
        }
        
        self.store.update_model_defaults("pbuf", optimized_params, metadata)
        
        # Should return warm start params
        warm_params = self.store.get_warm_start_params("pbuf")
        assert warm_params is not None
        assert warm_params["k_sat"] == 0.95
        assert warm_params["alpha"] == 4.8e-4
    
    def test_warm_start_params_old_optimization(self):
        """Test warm-start with old optimization results."""
        # Mock old timestamp
        old_timestamp = "2024-01-01T00:00:00Z"
        
        # Manually add old record to history
        old_record = OptimizationRecord(
            timestamp=old_timestamp,
            model="lcdm",
            dataset="cmb",
            optimized_params=["H0"],
            final_values={"H0": 70.0},
            chi2_improvement=1.0,
            convergence_status="success",
            optimizer_info={}
        )
        
        # Add to history file
        with open(self.store.history_file, 'w') as f:
            json.dump([old_record.__dict__], f)
        
        # Should not return warm start for old optimization
        warm_params = self.store.get_warm_start_params("lcdm", max_age_hours=1.0)
        assert warm_params is None
    
    def test_is_optimized(self):
        """Test optimization status checking."""
        # Initially not optimized
        assert not self.store.is_optimized("lcdm", "cmb")
        
        # Add optimization
        optimized_params = {"H0": 68.0}
        metadata = {
            "source_dataset": "cmb",
            "convergence_status": "success"
        }
        
        self.store.update_model_defaults("lcdm", optimized_params, metadata)
        
        # Should be optimized now
        assert self.store.is_optimized("lcdm", "cmb")
        assert not self.store.is_optimized("lcdm", "bao")  # Different dataset
    
    def test_cross_model_consistency_validation(self):
        """Test cross-model consistency validation."""
        # Set up consistent parameters
        lcdm_params = {"H0": 67.4, "Om0": 0.315}
        pbuf_params = {"H0": 67.4, "Om0": 0.315, "k_sat": 1.0}
        
        metadata = {"source_dataset": "cmb", "convergence_status": "success"}
        
        self.store.update_model_defaults("lcdm", lcdm_params, metadata)
        self.store.update_model_defaults("pbuf", pbuf_params, metadata)
        
        # Should be consistent
        comparison = self.store.validate_cross_model_consistency(log_warnings=False)
        assert comparison["_summary"]["is_fully_consistent"]
        assert len(comparison["_summary"]["divergent_params"]) == 0
        
        # Introduce divergence
        divergent_params = {"H0": 70.0}  # Different from ΛCDM
        self.store.update_model_defaults("pbuf", divergent_params, metadata)
        
        # Should detect divergence
        comparison = self.store.validate_cross_model_consistency(tolerance=1e-3, log_warnings=False)
        assert not comparison["_summary"]["is_fully_consistent"]
        assert "H0" in comparison["_summary"]["divergent_params"]
        assert comparison["H0"]["status"] == "divergent"
    
    def test_parameter_drift_detection(self):
        """Test parameter drift detection."""
        # Add initial optimization
        initial_params = {"H0": 67.0, "Om0": 0.30}
        metadata = {
            "source_dataset": "cmb",
            "convergence_status": "success"
        }
        
        self.store.update_model_defaults("lcdm", initial_params, metadata)
        
        # Manually modify the stored defaults to simulate drift without creating new optimization record
        data = self.store._load_model_data("lcdm")
        data["defaults"]["H0"] = 70.0  # Drift from 67.0 to 70.0
        self.store._save_model_data("lcdm", data)
        
        # Detect drift - should compare current (70.0) with last optimization (67.0)
        drift_analysis = self.store.detect_parameter_drift(max_drift_threshold=0.01)
        
        # The drift should be detected because |70.0 - 67.0| / 67.0 = 0.045 > 0.01
        assert drift_analysis["summary"]["drift_detected"]
        assert "lcdm.H0" in drift_analysis["summary"]["drifted_parameters"]
        
        lcdm_drift = drift_analysis["lcdm"]["parameter_drift"]["H0"]
        assert lcdm_drift["exceeds_threshold"]
        assert lcdm_drift["current_value"] == 70.0
        assert lcdm_drift["optimized_value"] == 67.0  # From the optimization history
    
    def test_storage_integrity_verification(self):
        """Test storage integrity verification and recovery."""
        # Normal case - should be healthy
        integrity = self.store.verify_storage_integrity()
        assert integrity["overall_status"] == "healthy"
        
        # Corrupt a file
        with open(self.store.lcdm_file, 'w') as f:
            f.write("invalid json content")
        
        # Should detect and recover
        integrity = self.store.verify_storage_integrity()
        
        # Check that recovery was attempted
        lcdm_status = integrity["models"]["lcdm"]
        assert not lcdm_status["is_valid_json"] or lcdm_status.get("recovery_attempted", False) or lcdm_status.get("rebuilt_from_defaults", False)
        
        # Verify file is now valid after recovery
        defaults = self.store.get_model_defaults("lcdm")
        assert "H0" in defaults
    
    def test_storage_backup_creation(self):
        """Test storage backup creation."""
        # Add some data first
        optimized_params = {"H0": 68.0}
        metadata = {"source_dataset": "cmb"}
        self.store.update_model_defaults("lcdm", optimized_params, metadata)
        
        # Create backup
        backup_path = self.store.create_storage_backup()
        backup_dir = Path(backup_path)
        
        assert backup_dir.exists()
        assert (backup_dir / "lcdm_optimized.json").exists()
        assert (backup_dir / "pbuf_optimized.json").exists()
        assert (backup_dir / "optimization_history.json").exists()
        assert (backup_dir / "backup_manifest.json").exists()
        
        # Check manifest
        with open(backup_dir / "backup_manifest.json", 'r') as f:
            manifest = json.load(f)
        
        assert "backup_timestamp" in manifest
        assert "lcdm_optimized.json" in manifest["backed_up_files"]
    
    def test_concurrent_access_protection(self):
        """Test file locking for concurrent access protection."""
        # This test simulates concurrent access by trying to acquire locks
        
        # Acquire lock for ΛCDM
        fd1 = self.store._acquire_lock("lcdm")
        
        # Try to acquire same lock again (should fail)
        with pytest.raises(RuntimeError, match="Could not acquire lock"):
            self.store._acquire_lock("lcdm")
        
        # Release lock
        self.store._release_lock(fd1)
        
        # Should be able to acquire again
        fd2 = self.store._acquire_lock("lcdm")
        self.store._release_lock(fd2)
    
    def test_export_optimization_summary(self):
        """Test optimization summary export."""
        # Add optimization data
        lcdm_params = {"H0": 67.5}
        pbuf_params = {"k_sat": 1.1, "alpha": 5.2e-4}
        metadata = {
            "source_dataset": "cmb",
            "chi2_improvement": 2.3,
            "convergence_status": "success"
        }
        
        self.store.update_model_defaults("lcdm", lcdm_params, metadata)
        self.store.update_model_defaults("pbuf", pbuf_params, metadata)
        
        # Export summary
        summary_path = Path(self.temp_dir) / "test_summary.json"
        self.store.export_optimization_summary(str(summary_path))
        
        assert summary_path.exists()
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        assert "export_timestamp" in summary
        assert "models" in summary
        assert "lcdm" in summary["models"]
        assert "pbuf" in summary["models"]
        assert "cross_model_consistency" in summary
        
        # Check ΛCDM data
        lcdm_data = summary["models"]["lcdm"]
        assert lcdm_data["current_defaults"]["H0"] == 67.5
        assert lcdm_data["is_recently_optimized"]
        
        # Check PBUF data
        pbuf_data = summary["models"]["pbuf"]
        assert pbuf_data["current_defaults"]["k_sat"] == 1.1
        assert pbuf_data["current_defaults"]["alpha"] == 5.2e-4
    
    def test_file_corruption_recovery(self):
        """Test recovery from various file corruption scenarios."""
        # Test 1: Corrupted JSON
        with open(self.store.lcdm_file, 'w') as f:
            f.write("{invalid json")
        
        # Should recover and return defaults
        defaults = self.store.get_model_defaults("lcdm")
        assert "H0" in defaults
        
        # Test 2: Missing required structure
        with open(self.store.pbuf_file, 'w') as f:
            json.dump({"wrong": "structure"}, f)
        
        defaults = self.store.get_model_defaults("pbuf")
        assert "k_sat" in defaults
        
        # Test 3: Empty file
        with open(self.store.lcdm_file, 'w') as f:
            f.write("")
        
        defaults = self.store.get_model_defaults("lcdm")
        assert "H0" in defaults
    
    def test_optimization_metadata_preservation(self):
        """Test that optimization metadata is properly preserved."""
        optimized_params = {"H0": 68.2, "Om0": 0.31}
        metadata = {
            "source_dataset": "cmb",
            "chi2_improvement": 4.5,
            "convergence_status": "success",
            "optimizer_info": {
                "method": "L-BFGS-B",
                "library": "scipy",
                "version": "1.13.0"
            },
            "covariance_scaling": 1.2,
            "n_function_evaluations": 45,
            "optimization_time": 12.3
        }
        
        self.store.update_model_defaults("lcdm", optimized_params, metadata)
        
        # Check that metadata is preserved in storage
        data = self.store._load_model_data("lcdm")
        stored_metadata = data["optimization_metadata"]
        
        assert stored_metadata["chi2_improvement"] == 4.5
        assert stored_metadata["convergence_status"] == "success"
        assert stored_metadata["optimizer_info"]["method"] == "L-BFGS-B"
        assert stored_metadata["covariance_scaling"] == 1.2
        assert "last_updated" in stored_metadata
        assert stored_metadata["optimized_params"] == ["H0", "Om0"]


if __name__ == "__main__":
    pytest.main([__file__])