"""
Optimized parameter storage system for PBUF cosmology fitting.

This module provides persistent storage and retrieval of optimized cosmological parameters,
with support for cross-model consistency validation, warm-start capabilities, and
comprehensive optimization metadata tracking.
"""

import json
import os
import time
import fcntl
import shutil
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from . import ParameterDict
from .parameter import get_defaults, validate_params, OPTIMIZABLE_PARAMETERS


@dataclass
class OptimizationRecord:
    """Record of a single optimization run."""
    timestamp: str
    model: str
    dataset: str
    optimized_params: List[str]
    final_values: Dict[str, float]
    chi2_improvement: float
    convergence_status: str
    optimizer_info: Dict[str, str]
    covariance_scaling: float = 1.0


@dataclass
class OptimizationResult:
    """Complete optimization result with metadata."""
    model: str
    optimized_params: Dict[str, float]
    starting_params: Dict[str, float]
    final_chi2: float
    chi2_improvement: float
    convergence_status: str
    n_function_evaluations: int
    optimization_time: float
    bounds_reached: List[str]
    optimizer_info: Dict[str, str]
    covariance_scaling: float
    metadata: Dict[str, Any]


class OptimizedParameterStore:
    """
    Centralized storage and retrieval system for optimized cosmological parameters.
    
    Provides persistent storage of optimization results with metadata preservation,
    cross-model consistency validation, and warm-start support for iterative optimization.
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    
    def __init__(self, storage_dir: str = "optimization_results"):
        """
        Initialize parameter store with specified storage directory.
        
        Args:
            storage_dir: Directory path for storing optimization results
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Model-specific storage files
        self.lcdm_file = self.storage_dir / "lcdm_optimized.json"
        self.pbuf_file = self.storage_dir / "pbuf_optimized.json"
        
        # Lock files for concurrent access protection
        self.lcdm_lock = self.storage_dir / "lcdm.lock"
        self.pbuf_lock = self.storage_dir / "pbuf.lock"
        
        # History and summary files
        self.history_file = self.storage_dir / "optimization_history.json"
        self.summary_file = self.storage_dir / "optimization_summary.json"
        
        # Initialize storage files if they don't exist
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize storage files with default structure if they don't exist."""
        # Initialize ΛCDM file
        if not self.lcdm_file.exists():
            lcdm_structure = {
                "defaults": get_defaults("lcdm"),
                "optimization_metadata": {
                    "initialized": datetime.now(timezone.utc).isoformat(),
                    "source": "hardcoded_defaults"
                }
            }
            with open(self.lcdm_file, 'w') as f:
                json.dump(lcdm_structure, f, indent=2)
        
        # Initialize PBUF file
        if not self.pbuf_file.exists():
            pbuf_structure = {
                "defaults": get_defaults("pbuf"),
                "optimization_metadata": {
                    "initialized": datetime.now(timezone.utc).isoformat(),
                    "source": "hardcoded_defaults"
                }
            }
            with open(self.pbuf_file, 'w') as f:
                json.dump(pbuf_structure, f, indent=2)
        
        if not self.history_file.exists():
            with open(self.history_file, 'w') as f:
                json.dump([], f, indent=2)
    
    def _get_lock_file(self, model: str) -> Path:
        """Get lock file path for specified model."""
        if model == "lcdm":
            return self.lcdm_lock
        elif model == "pbuf":
            return self.pbuf_lock
        else:
            raise ValueError(f"Unknown model type: {model}")
    
    def _get_model_file(self, model: str) -> Path:
        """Get storage file path for specified model."""
        if model == "lcdm":
            return self.lcdm_file
        elif model == "pbuf":
            return self.pbuf_file
        else:
            raise ValueError(f"Unknown model type: {model}")
    
    def _acquire_lock(self, model: str) -> int:
        """
        Acquire file lock for model-specific operations.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            
        Returns:
            File descriptor for the lock file
        """
        lock_file = self._get_lock_file(model)
        
        # Create lock file if it doesn't exist
        fd = os.open(str(lock_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
        
        try:
            # Acquire exclusive lock with timeout
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except (OSError, IOError):
            os.close(fd)
            raise RuntimeError(f"Could not acquire lock for {model} model. Another optimization may be running.")
    
    def _release_lock(self, fd: int) -> None:
        """Release file lock."""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except (OSError, IOError):
            pass  # Lock may have been released already
    
    def _load_model_data(self, model: str) -> Dict[str, Any]:
        """
        Load model data from storage with corruption detection.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            
        Returns:
            Model data dictionary
        """
        model_file = self._get_model_file(model)
        
        try:
            with open(model_file, 'r') as f:
                data = json.load(f)
            
            # Validate structure
            if not isinstance(data, dict) or "defaults" not in data:
                raise ValueError("Invalid file structure")
            
            return data
            
        except (json.JSONDecodeError, ValueError, FileNotFoundError) as e:
            # Handle corrupted or missing files
            print(f"Warning: Corrupted or missing {model} parameter file. Rebuilding from defaults.")
            return self._rebuild_from_defaults(model)
    
    def _rebuild_from_defaults(self, model: str) -> Dict[str, Any]:
        """
        Rebuild model data from default parameters.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            
        Returns:
            Rebuilt model data dictionary
        """
        defaults = get_defaults(model)
        
        data = {
            "defaults": defaults,
            "optimization_metadata": {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "rebuilt_from_defaults",
                "rebuild_reason": "corrupted_file"
            }
        }
        
        # Save rebuilt data
        model_file = self._get_model_file(model)
        with open(model_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data
    
    def _save_model_data(self, model: str, data: Dict[str, Any]) -> None:
        """
        Save model data to storage with atomic write and backup recovery.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            data: Model data to save
        """
        model_file = self._get_model_file(model)
        backup_file = model_file.with_suffix('.json.backup')
        temp_file = model_file.with_suffix('.json.tmp')
        
        # Create backup of existing file
        if model_file.exists():
            shutil.copy2(model_file, backup_file)
        
        try:
            # Atomic write: write to temp file first, then rename
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Verify the written data is valid JSON
            with open(temp_file, 'r') as f:
                json.load(f)  # This will raise an exception if JSON is invalid
            
            # Atomic rename (on most filesystems)
            temp_file.replace(model_file)
            
        except Exception as e:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
            
            # Restore backup if write failed and backup exists
            if backup_file.exists() and not model_file.exists():
                shutil.copy2(backup_file, model_file)
            
            raise RuntimeError(f"Failed to save {model} parameter data: {str(e)}") from e
    
    def verify_storage_integrity(self) -> Dict[str, Any]:
        """
        Verify integrity of all storage files and attempt recovery if needed.
        
        Returns:
            Dictionary with integrity check results and recovery actions taken
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        integrity_report = {
            "check_timestamp": datetime.now(timezone.utc).isoformat(),
            "models": {},
            "history": {},
            "recovery_actions": []
        }
        
        # Check model files
        for model in ["lcdm", "pbuf"]:
            model_file = self._get_model_file(model)
            backup_file = model_file.with_suffix('.json.backup')
            
            model_status = {
                "file_exists": model_file.exists(),
                "backup_exists": backup_file.exists(),
                "is_valid_json": False,
                "has_required_structure": False,
                "recovery_attempted": False
            }
            
            # Check main file
            if model_file.exists():
                try:
                    with open(model_file, 'r') as f:
                        data = json.load(f)
                    model_status["is_valid_json"] = True
                    
                    # Check required structure
                    if isinstance(data, dict) and "defaults" in data:
                        model_status["has_required_structure"] = True
                    else:
                        raise ValueError("Missing required structure")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    model_status["error"] = str(e)
                    
                    # Attempt recovery from backup
                    if backup_file.exists():
                        try:
                            with open(backup_file, 'r') as f:
                                backup_data = json.load(f)
                            
                            if isinstance(backup_data, dict) and "defaults" in backup_data:
                                shutil.copy2(backup_file, model_file)
                                model_status["recovery_attempted"] = True
                                model_status["recovery_successful"] = True
                                model_status["is_valid_json"] = True
                                model_status["has_required_structure"] = True
                                integrity_report["recovery_actions"].append(
                                    f"Restored {model} from backup due to corruption"
                                )
                            else:
                                raise ValueError("Backup also corrupted")
                                
                        except Exception as backup_error:
                            model_status["recovery_attempted"] = True
                            model_status["recovery_successful"] = False
                            model_status["backup_error"] = str(backup_error)
                            
                            # Rebuild from defaults as last resort
                            try:
                                self._rebuild_from_defaults(model)
                                model_status["rebuilt_from_defaults"] = True
                                integrity_report["recovery_actions"].append(
                                    f"Rebuilt {model} from defaults due to corruption"
                                )
                            except Exception as rebuild_error:
                                model_status["rebuild_error"] = str(rebuild_error)
            else:
                # File doesn't exist, create from defaults
                try:
                    self._rebuild_from_defaults(model)
                    model_status["created_from_defaults"] = True
                    integrity_report["recovery_actions"].append(
                        f"Created missing {model} file from defaults"
                    )
                except Exception as create_error:
                    model_status["create_error"] = str(create_error)
            
            integrity_report["models"][model] = model_status
        
        # Check history file
        history_status = {
            "file_exists": self.history_file.exists(),
            "is_valid_json": False,
            "is_valid_structure": False
        }
        
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)
                history_status["is_valid_json"] = True
                
                if isinstance(history_data, list):
                    history_status["is_valid_structure"] = True
                    history_status["record_count"] = len(history_data)
                else:
                    raise ValueError("History should be a list")
                    
            except (json.JSONDecodeError, ValueError) as e:
                history_status["error"] = str(e)
                
                # Recreate empty history file
                try:
                    with open(self.history_file, 'w') as f:
                        json.dump([], f, indent=2)
                    history_status["recreated"] = True
                    integrity_report["recovery_actions"].append("Recreated corrupted history file")
                except Exception as recreate_error:
                    history_status["recreate_error"] = str(recreate_error)
        else:
            # Create empty history file
            try:
                with open(self.history_file, 'w') as f:
                    json.dump([], f, indent=2)
                history_status["created"] = True
                integrity_report["recovery_actions"].append("Created missing history file")
            except Exception as create_error:
                history_status["create_error"] = str(create_error)
        
        integrity_report["history"] = history_status
        
        # Overall status
        all_models_ok = all(
            status.get("is_valid_json", False) and status.get("has_required_structure", False)
            for status in integrity_report["models"].values()
        )
        history_ok = integrity_report["history"].get("is_valid_json", False)
        
        integrity_report["overall_status"] = "healthy" if (all_models_ok and history_ok) else "recovered"
        
        return integrity_report
    
    def create_storage_backup(self, backup_dir: Optional[str] = None) -> str:
        """
        Create a complete backup of all storage files.
        
        Args:
            backup_dir: Directory for backup (default: storage_dir/backups/timestamp)
            
        Returns:
            Path to created backup directory
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        if backup_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.storage_dir / "backups" / timestamp
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup all storage files
        files_to_backup = [
            self.lcdm_file,
            self.pbuf_file,
            self.history_file,
            self.summary_file
        ]
        
        backed_up_files = []
        for file_path in files_to_backup:
            if file_path.exists():
                backup_file = backup_path / file_path.name
                shutil.copy2(file_path, backup_file)
                backed_up_files.append(file_path.name)
        
        # Create backup manifest
        manifest = {
            "backup_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(self.storage_dir),
            "backed_up_files": backed_up_files,
            "backup_tool": "OptimizedParameterStore.create_storage_backup"
        }
        
        with open(backup_path / "backup_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return str(backup_path)
    
    def get_model_defaults(self, model: str) -> ParameterDict:
        """
        Get current default parameter values for specified model with optimization metadata support.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            
        Returns:
            Dictionary of current default parameter values
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        if model not in ["lcdm", "pbuf"]:
            raise ValueError(f"Unknown model type: {model}. Must be 'lcdm' or 'pbuf'")
        
        fd = self._acquire_lock(model)
        try:
            data = self._load_model_data(model)
            
            # Get defaults, falling back to hardcoded defaults if empty
            defaults = data.get("defaults", {})
            if not defaults:
                defaults = get_defaults(model)
            
            # Validate parameters before returning
            validate_params(defaults, model)
            
            return defaults.copy()
            
        finally:
            self._release_lock(fd)
    
    def update_model_defaults(
        self,
        model: str,
        optimized_params: Dict[str, float],
        optimization_metadata: Dict[str, Any],
        dry_run: bool = False
    ) -> None:
        """
        Update model defaults with optimized parameters using non-destructive merge.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            optimized_params: Dictionary of optimized parameter values
            optimization_metadata: Metadata about the optimization run
            dry_run: If True, validate but don't persist changes
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        if model not in ["lcdm", "pbuf"]:
            raise ValueError(f"Unknown model type: {model}. Must be 'lcdm' or 'pbuf'")
        
        # Validate optimized parameters
        for param_name in optimized_params:
            if param_name not in OPTIMIZABLE_PARAMETERS.get(model, []):
                raise ValueError(
                    f"Parameter '{param_name}' is not optimizable for {model} model"
                )
        
        if dry_run:
            # For dry run, just validate the merge would work
            current_defaults = self.get_model_defaults(model)
            merged_params = current_defaults.copy()
            merged_params.update(optimized_params)
            validate_params(merged_params, model)
            return
        
        fd = self._acquire_lock(model)
        try:
            # Load current data
            data = self._load_model_data(model)
            
            # Perform non-destructive merge
            current_defaults = data.get("defaults", {})
            if not current_defaults:
                current_defaults = get_defaults(model)
            
            merged_params = current_defaults.copy()
            merged_params.update(optimized_params)
            
            # Validate merged parameters
            validate_params(merged_params, model)
            
            # Update data structure
            data["defaults"] = merged_params
            data["optimization_metadata"] = {
                **data.get("optimization_metadata", {}),
                **optimization_metadata,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "optimized_params": list(optimized_params.keys())
            }
            
            # Save updated data
            self._save_model_data(model, data)
            
            # Add to optimization history
            self._add_to_history(model, optimized_params, optimization_metadata)
            
        finally:
            self._release_lock(fd)
    
    def get_optimization_history(self, model: str) -> List[OptimizationRecord]:
        """
        Get optimization history for specified model with warm-start support.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            
        Returns:
            List of optimization records, most recent first
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        if model not in ["lcdm", "pbuf"]:
            raise ValueError(f"Unknown model type: {model}. Must be 'lcdm' or 'pbuf'")
        
        try:
            with open(self.history_file, 'r') as f:
                history_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        
        # Filter and convert to OptimizationRecord objects
        model_history = []
        for record_data in history_data:
            if record_data.get("model") == model:
                try:
                    record = OptimizationRecord(**record_data)
                    model_history.append(record)
                except (TypeError, ValueError):
                    # Skip malformed records
                    continue
        
        # Sort by timestamp, most recent first
        model_history.sort(key=lambda x: x.timestamp, reverse=True)
        
        return model_history
    
    def get_warm_start_params(self, model: str, max_age_hours: float = 24.0) -> Optional[ParameterDict]:
        """
        Get recent optimization results for warm-start support.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            max_age_hours: Maximum age of optimization results to consider (hours)
            
        Returns:
            Parameter dictionary for warm start, or None if no recent results
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        history = self.get_optimization_history(model)
        
        if not history:
            return None
        
        # Check if most recent optimization is within age limit
        most_recent = history[0]
        
        try:
            record_time = datetime.fromisoformat(most_recent.timestamp.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            age_hours = (current_time - record_time).total_seconds() / 3600
            
            if age_hours <= max_age_hours and most_recent.convergence_status == "success":
                # Return the optimized parameter values
                return most_recent.final_values.copy()
                
        except (ValueError, AttributeError):
            # Skip if timestamp parsing fails
            pass
        
        return None
    
    def is_optimized(self, model: str, dataset: str) -> bool:
        """
        Check if model has been optimized for specified dataset.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            dataset: Dataset name (e.g., "cmb", "bao", "sn")
            
        Returns:
            True if model has optimization results for the dataset
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        history = self.get_optimization_history(model)
        
        for record in history:
            if record.dataset == dataset and record.convergence_status == "success":
                return True
        
        return False
    
    def validate_cross_model_consistency(
        self, 
        tolerance: float = 1e-3, 
        log_warnings: bool = True
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Validate consistency of shared parameters between ΛCDM and PBUF models with detailed reporting.
        
        Args:
            tolerance: Maximum allowed relative difference between shared parameters
            log_warnings: Whether to print warnings for divergent parameters
            
        Returns:
            Dictionary with detailed parameter comparison information
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        # Get current defaults for both models
        lcdm_params = self.get_model_defaults("lcdm")
        pbuf_params = self.get_model_defaults("pbuf")
        
        # Shared parameters between ΛCDM and PBUF
        shared_params = {"H0", "Om0", "Obh2", "ns", "Neff", "Tcmb"}
        
        comparison_results = {}
        divergent_params = []
        
        for param in shared_params:
            if param in lcdm_params and param in pbuf_params:
                lcdm_val = lcdm_params[param]
                pbuf_val = pbuf_params[param]
                
                # Skip non-numeric parameters
                if not isinstance(lcdm_val, (int, float)) or not isinstance(pbuf_val, (int, float)):
                    comparison_results[param] = {
                        "status": "skipped",
                        "reason": "non_numeric",
                        "lcdm_value": str(lcdm_val),
                        "pbuf_value": str(pbuf_val)
                    }
                    continue
                
                # Calculate relative and absolute differences
                if lcdm_val != 0:
                    rel_diff = abs(pbuf_val - lcdm_val) / abs(lcdm_val)
                else:
                    rel_diff = abs(pbuf_val - lcdm_val)
                
                abs_diff = abs(pbuf_val - lcdm_val)
                
                # Determine consistency status
                is_consistent = rel_diff <= tolerance
                
                comparison_results[param] = {
                    "status": "consistent" if is_consistent else "divergent",
                    "lcdm_value": lcdm_val,
                    "pbuf_value": pbuf_val,
                    "absolute_difference": abs_diff,
                    "relative_difference": rel_diff,
                    "tolerance": tolerance,
                    "within_tolerance": is_consistent
                }
                
                if not is_consistent:
                    divergent_params.append(param)
                    
                    if log_warnings:
                        print(f"Warning: Parameter {param} divergence detected:")
                        print(f"  ΛCDM value: {lcdm_val}")
                        print(f"  PBUF value: {pbuf_val}")
                        print(f"  Relative difference: {rel_diff:.6f} (tolerance: {tolerance})")
            else:
                # Parameter missing in one or both models
                missing_in = []
                if param not in lcdm_params:
                    missing_in.append("lcdm")
                if param not in pbuf_params:
                    missing_in.append("pbuf")
                
                comparison_results[param] = {
                    "status": "missing",
                    "missing_in": missing_in,
                    "lcdm_value": lcdm_params.get(param, "missing"),
                    "pbuf_value": pbuf_params.get(param, "missing")
                }
        
        # Add summary information
        comparison_results["_summary"] = {
            "total_shared_params": len(shared_params),
            "compared_params": len([r for r in comparison_results.values() 
                                  if isinstance(r, dict) and r.get("status") not in ["missing", "skipped"]]),
            "consistent_params": len([r for r in comparison_results.values() 
                                    if isinstance(r, dict) and r.get("status") == "consistent"]),
            "divergent_params": divergent_params,
            "is_fully_consistent": len(divergent_params) == 0,
            "tolerance_used": tolerance,
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if log_warnings and divergent_params:
            print(f"\nCross-model consistency validation completed:")
            print(f"  {len(divergent_params)} parameter(s) exceed tolerance: {divergent_params}")
            print(f"  Consider re-running optimization to reduce parameter drift")
        elif log_warnings:
            print("Cross-model consistency validation passed: all shared parameters within tolerance")
        
        return comparison_results
    
    def get_shared_parameter_comparison(self) -> Dict[str, Any]:
        """
        Get detailed comparison of shared parameters between models.
        
        Returns:
            Dictionary with shared parameter values and differences
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        return self.validate_cross_model_consistency(log_warnings=False)
    
    def detect_parameter_drift(self, max_drift_threshold: float = 0.01) -> Dict[str, Any]:
        """
        Detect parameter drift by comparing current values with optimization history.
        
        Args:
            max_drift_threshold: Maximum allowed drift from last optimization
            
        Returns:
            Dictionary with drift analysis results
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        drift_analysis = {
            "lcdm": {},
            "pbuf": {},
            "summary": {
                "drift_detected": False,
                "drifted_parameters": [],
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        for model in ["lcdm", "pbuf"]:
            current_params = self.get_model_defaults(model)
            history = self.get_optimization_history(model)
            
            if not history:
                drift_analysis[model] = {
                    "status": "no_optimization_history",
                    "message": f"No optimization history available for {model} model"
                }
                continue
            
            # Get most recent successful optimization
            last_optimization = None
            for record in history:
                if record.convergence_status == "success":
                    last_optimization = record
                    break
            
            if not last_optimization:
                drift_analysis[model] = {
                    "status": "no_successful_optimization",
                    "message": f"No successful optimization found in {model} history"
                }
                continue
            
            # Compare current values with last optimization
            model_drift = {}
            for param, optimized_value in last_optimization.final_values.items():
                current_value = current_params.get(param)
                
                if current_value is None or not isinstance(current_value, (int, float)):
                    continue
                
                # Calculate drift
                if optimized_value != 0:
                    drift = abs(current_value - optimized_value) / abs(optimized_value)
                else:
                    drift = abs(current_value - optimized_value)
                
                model_drift[param] = {
                    "current_value": current_value,
                    "optimized_value": optimized_value,
                    "drift": drift,
                    "exceeds_threshold": drift > max_drift_threshold,
                    "last_optimization": last_optimization.timestamp
                }
                
                if drift > max_drift_threshold:
                    drift_analysis["summary"]["drift_detected"] = True
                    drift_analysis["summary"]["drifted_parameters"].append(f"{model}.{param}")
            
            drift_analysis[model] = {
                "status": "analyzed",
                "parameter_drift": model_drift,
                "last_optimization_timestamp": last_optimization.timestamp,
                "drift_threshold": max_drift_threshold
            }
        
        return drift_analysis
    
    def export_optimization_summary(self, output_path: str = "reports/optimization_summary.json") -> None:
        """
        Export comprehensive optimization summary for HTML report integration.
        
        Args:
            output_path: Path to save optimization summary JSON file
            
        Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
        """
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Build comprehensive summary
        summary = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "export_version": "1.0",
            "models": {},
            "cross_model_consistency": {},
            "optimization_history": {},
            "summary_statistics": {}
        }
        
        # Process each model
        for model in ["lcdm", "pbuf"]:
            try:
                # Get current model data
                model_data = self._load_model_data(model)
                defaults = model_data.get("defaults", {})
                metadata = model_data.get("optimization_metadata", {})
                
                # Get optimization history
                history = self.get_optimization_history(model)
                
                # Check if optimization is recent (within 24 hours)
                is_recently_optimized = False
                if history:
                    try:
                        last_timestamp = datetime.fromisoformat(history[0].timestamp.replace('Z', '+00:00'))
                        now = datetime.now(timezone.utc)
                        hours_since = (now - last_timestamp).total_seconds() / 3600
                        is_recently_optimized = hours_since <= 24.0
                    except (ValueError, AttributeError):
                        is_recently_optimized = False
                
                # Build model summary
                model_summary = {
                    "current_defaults": defaults,
                    "optimization_metadata": metadata,
                    "is_optimized": len(history) > 0,
                    "is_recently_optimized": is_recently_optimized,
                    "last_optimization": None,
                    "optimization_count": len(history),
                    "optimization_datasets": []
                }
                
                # Add last optimization details if available
                if history:
                    last_opt = history[0]  # Most recent
                    model_summary["last_optimization"] = {
                        "timestamp": last_opt.timestamp,
                        "dataset": last_opt.dataset,
                        "optimized_params": last_opt.optimized_params,
                        "final_values": last_opt.final_values,
                        "chi2_improvement": last_opt.chi2_improvement,
                        "convergence_status": last_opt.convergence_status,
                        "optimizer_info": last_opt.optimizer_info,
                        "covariance_scaling": getattr(last_opt, 'covariance_scaling', 1.0)
                    }
                    
                    # Collect unique datasets
                    model_summary["optimization_datasets"] = list(set(
                        record.dataset for record in history
                    ))
                
                # Add parameter change analysis
                if history and len(history) > 1:
                    # Compare current with initial optimization
                    initial_opt = history[-1]  # Oldest
                    current_opt = history[0]   # Most recent
                    
                    param_changes = {}
                    for param in current_opt.optimized_params:
                        if param in initial_opt.final_values and param in current_opt.final_values:
                            initial_val = initial_opt.final_values[param]
                            current_val = current_opt.final_values[param]
                            change = current_val - initial_val
                            rel_change = abs(change / initial_val) if initial_val != 0 else float('inf')
                            
                            param_changes[param] = {
                                "initial_value": initial_val,
                                "current_value": current_val,
                                "absolute_change": change,
                                "relative_change": rel_change,
                                "drift_magnitude": abs(rel_change)
                            }
                    
                    model_summary["parameter_evolution"] = param_changes
                
                summary["models"][model] = model_summary
                
            except Exception as e:
                summary["models"][model] = {
                    "error": f"Failed to process {model} model: {str(e)}",
                    "is_optimized": False
                }
        
        # Add cross-model consistency analysis
        try:
            consistency_results = self.validate_cross_model_consistency(log_warnings=False)
            summary["cross_model_consistency"] = consistency_results
        except Exception as e:
            summary["cross_model_consistency"] = {
                "error": f"Failed to validate cross-model consistency: {str(e)}"
            }
        
        # Add optimization history summary
        try:
            all_history = []
            for model in ["lcdm", "pbuf"]:
                model_history = self.get_optimization_history(model)
                for record in model_history:
                    all_history.append({
                        "model": record.model,
                        "timestamp": record.timestamp,
                        "dataset": record.dataset,
                        "chi2_improvement": record.chi2_improvement,
                        "convergence_status": record.convergence_status,
                        "param_count": len(record.optimized_params)
                    })
            
            # Sort by timestamp, most recent first
            all_history.sort(key=lambda x: x["timestamp"], reverse=True)
            
            summary["optimization_history"] = {
                "total_optimizations": len(all_history),
                "recent_optimizations": all_history[:10],  # Last 10 optimizations
                "successful_optimizations": len([h for h in all_history if h["convergence_status"] == "success"]),
                "datasets_optimized": list(set(h["dataset"] for h in all_history))
            }
            
        except Exception as e:
            summary["optimization_history"] = {
                "error": f"Failed to process optimization history: {str(e)}"
            }
        
        # Add summary statistics
        try:
            lcdm_optimized = summary["models"].get("lcdm", {}).get("is_optimized", False)
            pbuf_optimized = summary["models"].get("pbuf", {}).get("is_optimized", False)
            
            # Calculate total χ² improvements
            total_chi2_improvement = 0.0
            for model_data in summary["models"].values():
                if isinstance(model_data, dict) and "last_optimization" in model_data:
                    last_opt = model_data["last_optimization"]
                    if last_opt and "chi2_improvement" in last_opt:
                        total_chi2_improvement += last_opt["chi2_improvement"]
            
            summary["summary_statistics"] = {
                "models_optimized": sum([lcdm_optimized, pbuf_optimized]),
                "total_models": 2,
                "optimization_coverage": sum([lcdm_optimized, pbuf_optimized]) / 2.0,
                "total_chi2_improvement": total_chi2_improvement,
                "cross_model_consistent": summary["cross_model_consistency"].get("_summary", {}).get("is_fully_consistent", False),
                "has_recent_optimizations": len(summary["optimization_history"].get("recent_optimizations", [])) > 0
            }
            
        except Exception as e:
            summary["summary_statistics"] = {
                "error": f"Failed to calculate summary statistics: {str(e)}"
            }
        
        # Write summary to file
        try:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"[EXPORT] Optimization summary exported to {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to write optimization summary to {output_path}: {str(e)}")
    
    def get_html_optimization_section(self) -> Dict[str, Any]:
        """
        Get optimization data formatted for HTML report integration.
        
        Returns:
            Dictionary with HTML-ready optimization data
            
        Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
        """
        try:
            # Export to temporary location and read back
            temp_path = "/tmp/optimization_summary_temp.json"
            self.export_optimization_summary(temp_path)
            
            with open(temp_path, 'r') as f:
                summary_data = json.load(f)
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            # Format for HTML template
            html_data = {
                "optimization_enabled": True,
                "models": {},
                "cross_model_status": "unknown",
                "summary_stats": {},
                "recent_activity": []
            }
            
            # Process model data for HTML
            for model, model_data in summary_data.get("models", {}).items():
                if "error" in model_data:
                    html_data["models"][model] = {
                        "status": "error",
                        "error_message": model_data["error"]
                    }
                    continue
                
                last_opt = model_data.get("last_optimization")
                html_model = {
                    "status": "optimized" if model_data.get("is_optimized", False) else "default",
                    "optimization_count": model_data.get("optimization_count", 0),
                    "datasets": model_data.get("optimization_datasets", [])
                }
                
                if last_opt:
                    html_model.update({
                        "last_optimization_date": last_opt.get("timestamp", "unknown"),
                        "last_dataset": last_opt.get("dataset", "unknown"),
                        "chi2_improvement": last_opt.get("chi2_improvement", 0.0),
                        "convergence_status": last_opt.get("convergence_status", "unknown"),
                        "optimized_params": last_opt.get("optimized_params", []),
                        "optimizer_method": last_opt.get("optimizer_info", {}).get("method", "unknown")
                    })
                
                html_data["models"][model] = html_model
            
            # Cross-model consistency status
            consistency = summary_data.get("cross_model_consistency", {})
            consistency_summary = consistency.get("_summary", {})
            html_data["cross_model_status"] = "consistent" if consistency_summary.get("is_fully_consistent", False) else "divergent"
            html_data["divergent_params"] = consistency_summary.get("divergent_params", [])
            
            # Summary statistics
            stats = summary_data.get("summary_statistics", {})
            html_data["summary_stats"] = {
                "models_optimized": stats.get("models_optimized", 0),
                "total_chi2_improvement": stats.get("total_chi2_improvement", 0.0),
                "optimization_coverage": stats.get("optimization_coverage", 0.0) * 100,  # Convert to percentage
                "has_recent_activity": stats.get("has_recent_optimizations", False)
            }
            
            # Recent activity
            history = summary_data.get("optimization_history", {})
            recent_opts = history.get("recent_optimizations", [])[:5]  # Top 5 for HTML
            html_data["recent_activity"] = [
                {
                    "model": opt["model"].upper(),
                    "dataset": opt["dataset"].upper(),
                    "date": opt["timestamp"][:10],  # Just date part
                    "chi2_improvement": opt["chi2_improvement"],
                    "status": opt["convergence_status"]
                }
                for opt in recent_opts
            ]
            
            return html_data
            
        except Exception as e:
            # Return error state for HTML
            return {
                "optimization_enabled": False,
                "error": f"Failed to generate HTML optimization data: {str(e)}",
                "models": {},
                "cross_model_status": "error",
                "summary_stats": {},
                "recent_activity": []
            }
    
    def validate_html_optimization_consistency(
        self, 
        html_data: Dict[str, Any], 
        summary_json_path: str = "reports/optimization_summary.json"
    ) -> Dict[str, Any]:
        """
        Validate that HTML optimization data matches the optimization_summary.json file.
        
        Args:
            html_data: HTML optimization data dictionary
            summary_json_path: Path to optimization summary JSON file
            
        Returns:
            Dictionary with validation results and any discrepancies found
            
        Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
        """
        validation_result = {
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "is_consistent": True,
            "discrepancies": [],
            "summary_file_exists": False,
            "html_data_valid": False,
            "comparison_results": {}
        }
        
        try:
            # Check if summary JSON file exists
            summary_path = Path(summary_json_path)
            if not summary_path.exists():
                validation_result["discrepancies"].append(
                    f"Summary JSON file not found: {summary_json_path}"
                )
                validation_result["is_consistent"] = False
                return validation_result
            
            validation_result["summary_file_exists"] = True
            
            # Load summary JSON
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
            
            # Validate HTML data structure
            required_html_keys = ["optimization_enabled", "models", "cross_model_status", "summary_stats"]
            missing_keys = [key for key in required_html_keys if key not in html_data]
            if missing_keys:
                validation_result["discrepancies"].append(
                    f"HTML data missing required keys: {missing_keys}"
                )
                validation_result["is_consistent"] = False
                return validation_result
            
            validation_result["html_data_valid"] = True
            
            # Compare model optimization status
            for model in ["lcdm", "pbuf"]:
                html_model = html_data.get("models", {}).get(model, {})
                json_model = summary_data.get("models", {}).get(model, {})
                
                comparison = {
                    "model": model,
                    "status_match": True,
                    "chi2_match": True,
                    "param_count_match": True,
                    "discrepancies": []
                }
                
                # Compare optimization status
                html_optimized = html_model.get("status") == "optimized"
                json_optimized = json_model.get("is_optimized", False)
                
                if html_optimized != json_optimized:
                    comparison["status_match"] = False
                    comparison["discrepancies"].append(
                        f"Optimization status mismatch: HTML={html_optimized}, JSON={json_optimized}"
                    )
                
                # Compare χ² improvement (if both are optimized)
                if html_optimized and json_optimized:
                    html_chi2 = html_model.get("chi2_improvement", 0.0)
                    json_chi2 = json_model.get("last_optimization", {}).get("chi2_improvement", 0.0)
                    
                    if abs(html_chi2 - json_chi2) > 1e-6:
                        comparison["chi2_match"] = False
                        comparison["discrepancies"].append(
                            f"χ² improvement mismatch: HTML={html_chi2:.6f}, JSON={json_chi2:.6f}"
                        )
                    
                    # Compare optimized parameter count
                    html_params = html_model.get("optimized_params", [])
                    json_params = json_model.get("last_optimization", {}).get("optimized_params", [])
                    
                    if len(html_params) != len(json_params):
                        comparison["param_count_match"] = False
                        comparison["discrepancies"].append(
                            f"Parameter count mismatch: HTML={len(html_params)}, JSON={len(json_params)}"
                        )
                
                validation_result["comparison_results"][model] = comparison
                
                if comparison["discrepancies"]:
                    validation_result["is_consistent"] = False
                    validation_result["discrepancies"].extend(
                        [f"{model.upper()}: {disc}" for disc in comparison["discrepancies"]]
                    )
            
            # Compare cross-model consistency status
            html_consistency = html_data.get("cross_model_status", "unknown")
            json_consistency_data = summary_data.get("cross_model_consistency", {})
            json_consistency = "consistent" if json_consistency_data.get("_summary", {}).get("is_fully_consistent", False) else "divergent"
            
            if html_consistency != json_consistency and html_consistency != "unknown":
                validation_result["discrepancies"].append(
                    f"Cross-model consistency mismatch: HTML={html_consistency}, JSON={json_consistency}"
                )
                validation_result["is_consistent"] = False
            
            # Compare summary statistics
            html_stats = html_data.get("summary_stats", {})
            json_stats = summary_data.get("summary_statistics", {})
            
            # Check models optimized count
            html_count = html_stats.get("models_optimized", 0)
            json_count = json_stats.get("models_optimized", 0)
            if html_count != json_count:
                validation_result["discrepancies"].append(
                    f"Models optimized count mismatch: HTML={html_count}, JSON={json_count}"
                )
                validation_result["is_consistent"] = False
            
            # Check total χ² improvement
            html_total_chi2 = html_stats.get("total_chi2_improvement", 0.0)
            json_total_chi2 = json_stats.get("total_chi2_improvement", 0.0)
            if abs(html_total_chi2 - json_total_chi2) > 1e-6:
                validation_result["discrepancies"].append(
                    f"Total χ² improvement mismatch: HTML={html_total_chi2:.6f}, JSON={json_total_chi2:.6f}"
                )
                validation_result["is_consistent"] = False
            
        except Exception as e:
            validation_result["discrepancies"].append(f"Validation error: {str(e)}")
            validation_result["is_consistent"] = False
        
        return validation_result
    
    def _add_to_history(
        self,
        model: str,
        optimized_params: Dict[str, float],
        optimization_metadata: Dict[str, Any]
    ) -> None:
        """
        Add optimization result to history.
        
        Args:
            model: Model type
            optimized_params: Optimized parameter values
            optimization_metadata: Optimization metadata
        """
        # Create optimization record
        record = OptimizationRecord(
            timestamp=optimization_metadata.get("timestamp", datetime.now(timezone.utc).isoformat()),
            model=model,
            dataset=optimization_metadata.get("dataset", optimization_metadata.get("source_dataset", "unknown")),
            optimized_params=list(optimized_params.keys()),
            final_values=optimized_params.copy(),
            chi2_improvement=optimization_metadata.get("chi2_improvement", 0.0),
            convergence_status=optimization_metadata.get("convergence_status", "unknown"),
            optimizer_info=optimization_metadata.get("optimizer_info", {}),
            covariance_scaling=optimization_metadata.get("covariance_scaling", 1.0)
        )
        
        # Load existing history
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        
        # Add new record
        history.append(asdict(record))
        
        # Keep only last 100 records to prevent unbounded growth
        if len(history) > 100:
            history = history[-100:]
        
        # Save updated history
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
