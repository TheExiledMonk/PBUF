#!/usr/bin/env python3
"""
Deployment validation for CMB raw parameter processing.

This script performs comprehensive validation of the deployment environment,
including system checks for required dependencies, registry integration testing,
and fitting pipeline compatibility verification.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    from core.cmb_config_integration import (
        DataPreparationConfigManager,
        validate_deployment_config
    )
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from core.cmb_config_integration import (
        DataPreparationConfigManager,
        validate_deployment_config
    )


class DeploymentValidator:
    """Comprehensive deployment validation for CMB processing."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize deployment validator.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_manager = DataPreparationConfigManager(config_file)
        self.config = self.config_manager.get_config()
        self.validation_results = {
            "overall_status": "unknown",
            "checks_passed": 0,
            "checks_failed": 0,
            "checks_warned": 0,
            "system_checks": {},
            "dependency_checks": {},
            "registry_checks": {},
            "pipeline_checks": {},
            "performance_checks": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all deployment validation checks.
        
        Returns:
            Dictionary with comprehensive validation results
        """
        print("Running deployment validation checks...")
        print("=" * 60)
        
        # System environment checks
        self._check_system_environment()
        
        # Dependency checks
        self._check_dependencies()
        
        # Configuration validation
        self._check_configuration()
        
        # Registry integration checks
        self._check_registry_integration()
        
        # Pipeline compatibility checks
        self._check_pipeline_compatibility()
        
        # Performance checks
        self._check_performance_requirements()
        
        # Determine overall status
        self._determine_overall_status()
        
        return self.validation_results
    
    def _check_system_environment(self):
        """Check system environment and requirements."""
        print("Checking system environment...")
        
        checks = {
            "python_version": self._check_python_version(),
            "memory_available": self._check_memory_availability(),
            "disk_space": self._check_disk_space(),
            "file_permissions": self._check_file_permissions(),
            "environment_variables": self._check_environment_variables()
        }
        
        self.validation_results["system_checks"] = checks
        self._update_counters(checks)
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        import sys
        
        version = sys.version_info
        required_major, required_minor = 3, 8
        
        if version.major >= required_major and version.minor >= required_minor:
            return {
                "status": "pass",
                "message": f"Python {version.major}.{version.minor}.{version.micro} is compatible",
                "details": {"version": f"{version.major}.{version.minor}.{version.micro}"}
            }
        else:
            return {
                "status": "fail",
                "message": f"Python {version.major}.{version.minor} is too old (requires >= {required_major}.{required_minor})",
                "details": {"version": f"{version.major}.{version.minor}.{version.micro}"}
            }
    
    def _check_memory_availability(self) -> Dict[str, Any]:
        """Check available system memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            required_gb = self.config.memory_limit_gb
            
            if available_gb >= required_gb:
                return {
                    "status": "pass",
                    "message": f"Sufficient memory available: {available_gb:.1f}GB (requires {required_gb}GB)",
                    "details": {"available_gb": available_gb, "required_gb": required_gb}
                }
            else:
                return {
                    "status": "warn",
                    "message": f"Low memory: {available_gb:.1f}GB available (requires {required_gb}GB)",
                    "details": {"available_gb": available_gb, "required_gb": required_gb}
                }
        except ImportError:
            return {
                "status": "warn",
                "message": "Cannot check memory (psutil not available)",
                "details": {"psutil_available": False}
            }
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Memory check failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space for output and cache directories."""
        try:
            import shutil
            
            paths_to_check = [
                ("output_path", self.config.output_path),
                ("cache_path", self.config.cache_path)
            ]
            
            results = {}
            overall_status = "pass"
            
            for name, path in paths_to_check:
                try:
                    path_obj = Path(path)
                    path_obj.mkdir(parents=True, exist_ok=True)
                    
                    free_bytes = shutil.disk_usage(path_obj).free
                    free_gb = free_bytes / (1024**3)
                    
                    # Require at least 1GB free space
                    if free_gb >= 1.0:
                        results[name] = {"status": "pass", "free_gb": free_gb}
                    else:
                        results[name] = {"status": "warn", "free_gb": free_gb}
                        overall_status = "warn"
                        
                except Exception as e:
                    results[name] = {"status": "fail", "error": str(e)}
                    overall_status = "fail"
            
            return {
                "status": overall_status,
                "message": f"Disk space check completed",
                "details": results
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Disk space check failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _check_file_permissions(self) -> Dict[str, Any]:
        """Check file system permissions for required directories."""
        paths_to_check = [
            ("output_path", self.config.output_path),
            ("cache_path", self.config.cache_path)
        ]
        
        if self.config.log_file:
            log_path = Path(self.config.log_file)
            paths_to_check.append(("log_path", str(log_path.parent)))
        
        results = {}
        overall_status = "pass"
        
        for name, path in paths_to_check:
            try:
                path_obj = Path(path)
                path_obj.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = path_obj / ".deployment_test"
                test_file.write_text("test")
                test_file.unlink()
                
                results[name] = {"status": "pass", "writable": True}
                
            except Exception as e:
                results[name] = {"status": "fail", "writable": False, "error": str(e)}
                overall_status = "fail"
        
        return {
            "status": overall_status,
            "message": "File permissions check completed",
            "details": results
        }
    
    def _check_environment_variables(self) -> Dict[str, Any]:
        """Check relevant environment variables."""
        import os
        
        relevant_vars = [
            "PBUF_OUTPUT_PATH", "PBUF_CACHE_PATH", "PBUF_LOG_LEVEL",
            "CMB_USE_RAW_PARAMETERS", "CMB_FALLBACK_TO_LEGACY"
        ]
        
        found_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                found_vars[var] = os.environ[var]
        
        return {
            "status": "pass",
            "message": f"Found {len(found_vars)} relevant environment variables",
            "details": {"environment_variables": found_vars}
        }
    
    def _check_dependencies(self):
        """Check required and optional dependencies."""
        print("Checking dependencies...")
        
        checks = {
            "numpy": self._check_numpy(),
            "pbuf_background": self._check_pbuf_background(),
            "pandas": self._check_pandas(),
            "psutil": self._check_psutil(),
            "scipy": self._check_scipy()
        }
        
        self.validation_results["dependency_checks"] = checks
        self._update_counters(checks)
    
    def _check_numpy(self) -> Dict[str, Any]:
        """Check NumPy availability and version."""
        try:
            import numpy as np
            version = np.__version__
            
            # Check for minimum version (1.18.0)
            version_parts = [int(x) for x in version.split('.')]
            if version_parts[0] > 1 or (version_parts[0] == 1 and version_parts[1] >= 18):
                return {
                    "status": "pass",
                    "message": f"NumPy {version} is available and compatible",
                    "details": {"version": version}
                }
            else:
                return {
                    "status": "warn",
                    "message": f"NumPy {version} may be too old (recommend >= 1.18.0)",
                    "details": {"version": version}
                }
        except ImportError:
            return {
                "status": "fail",
                "message": "NumPy is not available (required)",
                "details": {"available": False}
            }
    
    def _check_pbuf_background(self) -> Dict[str, Any]:
        """Check PBUF background integrator availability."""
        try:
            from derivation.cmb_background import BackgroundIntegrator, compute_sound_horizon
            return {
                "status": "pass",
                "message": "PBUF background integrators are available",
                "details": {"available": True}
            }
        except ImportError as e:
            return {
                "status": "fail",
                "message": f"PBUF background integrators not available: {e}",
                "details": {"available": False, "error": str(e)}
            }
    
    def _check_pandas(self) -> Dict[str, Any]:
        """Check pandas availability (optional but recommended)."""
        try:
            import pandas as pd
            version = pd.__version__
            return {
                "status": "pass",
                "message": f"Pandas {version} is available",
                "details": {"version": version, "available": True}
            }
        except ImportError:
            return {
                "status": "warn",
                "message": "Pandas not available (optional, but recommended for CSV parsing)",
                "details": {"available": False}
            }
    
    def _check_psutil(self) -> Dict[str, Any]:
        """Check psutil availability (optional for monitoring)."""
        try:
            import psutil
            version = psutil.__version__
            return {
                "status": "pass",
                "message": f"psutil {version} is available",
                "details": {"version": version, "available": True}
            }
        except ImportError:
            return {
                "status": "warn",
                "message": "psutil not available (optional, for performance monitoring)",
                "details": {"available": False}
            }
    
    def _check_scipy(self) -> Dict[str, Any]:
        """Check scipy availability (optional for advanced numerical operations)."""
        try:
            import scipy
            version = scipy.__version__
            return {
                "status": "pass",
                "message": f"SciPy {version} is available",
                "details": {"version": version, "available": True}
            }
        except ImportError:
            return {
                "status": "warn",
                "message": "SciPy not available (optional, for advanced numerical operations)",
                "details": {"available": False}
            }
    
    def _check_configuration(self):
        """Check configuration validity."""
        print("Checking configuration...")
        
        validation_results = validate_deployment_config(self.config_manager.config_file)
        
        if validation_results["valid"]:
            status = "pass"
        elif validation_results["errors"]:
            status = "fail"
        else:
            status = "warn"
        
        check_result = {
            "configuration_validation": {
                "status": status,
                "message": f"Configuration validation: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings",
                "details": validation_results
            }
        }
        
        self.validation_results["system_checks"].update(check_result)
        self._update_counters(check_result)
    
    def _check_registry_integration(self):
        """Check registry integration with mock datasets."""
        print("Checking registry integration...")
        
        checks = {
            "mock_registry_entry": self._test_mock_registry_entry(),
            "parameter_detection": self._test_parameter_detection(),
            "legacy_fallback": self._test_legacy_fallback()
        }
        
        self.validation_results["registry_checks"] = checks
        self._update_counters(checks)
    
    def _test_mock_registry_entry(self) -> Dict[str, Any]:
        """Test processing with a mock Planck-style registry entry."""
        try:
            # Create mock registry entry
            mock_entry = {
                "metadata": {
                    "name": "mock_planck_cmb",
                    "dataset_type": "cmb",
                    "source": "deployment_test",
                    "citation": "Test Dataset"
                },
                "sources": {
                    "parameters": {
                        "url": "mock_planck_params.json",
                        "extraction": {
                            "target_files": ["mock_planck_params.json"]
                        }
                    }
                }
            }
            
            # Test parameter detection
            from derivation.cmb_derivation import detect_raw_parameters
            result = detect_raw_parameters(mock_entry)
            
            if result is not None:
                return {
                    "status": "pass",
                    "message": "Mock registry entry processed successfully",
                    "details": {"detected_format": result.format_type.value}
                }
            else:
                return {
                    "status": "warn",
                    "message": "No parameters detected in mock entry (expected for test)",
                    "details": {"detection_result": None}
                }
                
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Mock registry test failed: {e}",
                "details": {"error": str(e), "traceback": traceback.format_exc()}
            }
    
    def _test_parameter_detection(self) -> Dict[str, Any]:
        """Test parameter detection functionality."""
        try:
            from derivation.cmb_derivation import classify_parameter_format, normalize_parameter_names
            
            # Test format classification
            test_files = [
                "test_params.json",
                "test_params.csv", 
                "test_params.npy",
                "test_params.txt"
            ]
            
            format_results = {}
            for test_file in test_files:
                format_type = classify_parameter_format(test_file)
                format_results[test_file] = format_type.value
            
            # Test parameter name normalization
            test_params = {
                "H0": 70.0,
                "omega_m": 0.3,
                "omegabh2": 0.022,
                "ns": 0.96,
                "tau_reio": 0.06
            }
            
            normalized = normalize_parameter_names(test_params)
            
            return {
                "status": "pass",
                "message": "Parameter detection functions working correctly",
                "details": {
                    "format_classification": format_results,
                    "name_normalization": normalized
                }
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Parameter detection test failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _test_legacy_fallback(self) -> Dict[str, Any]:
        """Test legacy fallback functionality."""
        try:
            # Test that legacy fallback configuration works
            config = self.config.cmb
            
            if config.fallback_to_legacy:
                return {
                    "status": "pass",
                    "message": "Legacy fallback is enabled",
                    "details": {"fallback_enabled": True}
                }
            else:
                return {
                    "status": "warn",
                    "message": "Legacy fallback is disabled (may cause failures with old datasets)",
                    "details": {"fallback_enabled": False}
                }
                
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Legacy fallback test failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _check_pipeline_compatibility(self):
        """Check fitting pipeline compatibility."""
        print("Checking pipeline compatibility...")
        
        checks = {
            "standard_dataset_format": self._test_standard_dataset_format(),
            "metadata_compatibility": self._test_metadata_compatibility(),
            "covariance_format": self._test_covariance_format()
        }
        
        self.validation_results["pipeline_checks"] = checks
        self._update_counters(checks)
    
    def _test_standard_dataset_format(self) -> Dict[str, Any]:
        """Test StandardDataset format compatibility."""
        try:
            from core.schema import StandardDataset
            from derivation.cmb_models import DistancePriors
            
            # Create mock distance priors
            mock_priors = DistancePriors(
                R=1.75,
                l_A=301.8,
                Omega_b_h2=0.02237,
                theta_star=1.04092
            )
            
            # Create mock StandardDataset
            z_array = np.array([1089.8, 1089.8, 1089.8])
            observable_array = np.array([mock_priors.R, mock_priors.l_A, mock_priors.theta_star])
            uncertainty_array = np.array([0.01, 1.0, 0.001])
            
            mock_dataset = StandardDataset(
                z=z_array,
                observable=observable_array,
                uncertainty=uncertainty_array,
                covariance=None,
                metadata={"dataset_type": "cmb", "test": True}
            )
            
            # Validate dataset structure
            if (len(mock_dataset.z) == 3 and 
                len(mock_dataset.observable) == 3 and
                len(mock_dataset.uncertainty) == 3):
                return {
                    "status": "pass",
                    "message": "StandardDataset format is compatible",
                    "details": {"format_valid": True}
                }
            else:
                return {
                    "status": "fail",
                    "message": "StandardDataset format validation failed",
                    "details": {"format_valid": False}
                }
                
        except Exception as e:
            return {
                "status": "fail",
                "message": f"StandardDataset format test failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _test_metadata_compatibility(self) -> Dict[str, Any]:
        """Test metadata format compatibility."""
        try:
            from derivation.cmb_derivation import create_metadata
            from derivation.cmb_models import ParameterSet, CMBConfig
            
            # Create mock parameter set
            mock_params = ParameterSet(
                H0=70.0,
                Omega_m=0.3,
                Omega_b_h2=0.022,
                n_s=0.96,
                tau=0.06
            )
            
            # Create mock registry entry
            mock_registry = {
                "metadata": {
                    "name": "test_dataset",
                    "source": "test",
                    "citation": "Test Citation"
                }
            }
            
            # Test metadata creation
            config = CMBConfig()
            metadata = create_metadata(mock_params, mock_registry, config, "raw_parameters")
            
            # Check required fields
            required_fields = ["dataset_type", "processing_method", "observables"]
            missing_fields = [field for field in required_fields if field not in metadata]
            
            if not missing_fields:
                return {
                    "status": "pass",
                    "message": "Metadata format is compatible",
                    "details": {"required_fields_present": True}
                }
            else:
                return {
                    "status": "fail",
                    "message": f"Missing required metadata fields: {missing_fields}",
                    "details": {"missing_fields": missing_fields}
                }
                
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Metadata compatibility test failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _test_covariance_format(self) -> Dict[str, Any]:
        """Test covariance matrix format compatibility."""
        try:
            # Create mock covariance matrix
            mock_covariance = np.array([
                [1e-4, 1e-5, 1e-6],
                [1e-5, 1e-3, 1e-5],
                [1e-6, 1e-5, 1e-6]
            ])
            
            # Test matrix properties
            is_symmetric = np.allclose(mock_covariance, mock_covariance.T)
            eigenvals = np.linalg.eigvals(mock_covariance)
            is_positive_definite = np.all(eigenvals > 0)
            
            if is_symmetric and is_positive_definite:
                return {
                    "status": "pass",
                    "message": "Covariance matrix format is valid",
                    "details": {
                        "symmetric": is_symmetric,
                        "positive_definite": is_positive_definite
                    }
                }
            else:
                return {
                    "status": "warn",
                    "message": "Covariance matrix validation issues detected",
                    "details": {
                        "symmetric": is_symmetric,
                        "positive_definite": is_positive_definite
                    }
                }
                
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Covariance format test failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _check_performance_requirements(self):
        """Check performance-related requirements."""
        print("Checking performance requirements...")
        
        checks = {
            "numerical_stability": self._test_numerical_stability(),
            "memory_usage": self._test_memory_usage(),
            "computation_speed": self._test_computation_speed()
        }
        
        self.validation_results["performance_checks"] = checks
        self._update_counters(checks)
    
    def _test_numerical_stability(self) -> Dict[str, Any]:
        """Test numerical stability of computations."""
        try:
            # Test basic numerical operations
            test_matrix = np.random.rand(4, 4)
            test_matrix = test_matrix @ test_matrix.T  # Make positive definite
            
            # Test condition number
            condition_number = np.linalg.cond(test_matrix)
            
            # Test eigenvalue computation
            eigenvals = np.linalg.eigvals(test_matrix)
            min_eigenval = np.min(eigenvals)
            
            if condition_number < 1e12 and min_eigenval > 1e-15:
                return {
                    "status": "pass",
                    "message": "Numerical stability tests passed",
                    "details": {
                        "condition_number": condition_number,
                        "min_eigenvalue": min_eigenval
                    }
                }
            else:
                return {
                    "status": "warn",
                    "message": "Potential numerical stability issues",
                    "details": {
                        "condition_number": condition_number,
                        "min_eigenvalue": min_eigenval
                    }
                }
                
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Numerical stability test failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns."""
        try:
            import gc
            
            # Get initial memory usage
            gc.collect()
            
            # Simulate typical operations
            large_array = np.random.rand(1000, 1000)
            covariance = large_array @ large_array.T
            eigenvals = np.linalg.eigvals(covariance)
            
            # Clean up
            del large_array, covariance, eigenvals
            gc.collect()
            
            return {
                "status": "pass",
                "message": "Memory usage test completed successfully",
                "details": {"test_completed": True}
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Memory usage test failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _test_computation_speed(self) -> Dict[str, Any]:
        """Test computation speed benchmarks."""
        try:
            import time
            
            # Test matrix operations speed
            start_time = time.time()
            
            test_matrix = np.random.rand(100, 100)
            for _ in range(10):
                result = np.linalg.inv(test_matrix @ test_matrix.T)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Expect operations to complete within reasonable time
            if duration < 1.0:  # 1 second threshold
                return {
                    "status": "pass",
                    "message": f"Computation speed acceptable: {duration:.3f}s",
                    "details": {"duration_seconds": duration}
                }
            else:
                return {
                    "status": "warn",
                    "message": f"Computation speed may be slow: {duration:.3f}s",
                    "details": {"duration_seconds": duration}
                }
                
        except Exception as e:
            return {
                "status": "fail",
                "message": f"Computation speed test failed: {e}",
                "details": {"error": str(e)}
            }
    
    def _update_counters(self, checks: Dict[str, Dict[str, Any]]):
        """Update validation counters based on check results."""
        for check_name, result in checks.items():
            status = result.get("status", "unknown")
            
            if status == "pass":
                self.validation_results["checks_passed"] += 1
            elif status == "fail":
                self.validation_results["checks_failed"] += 1
                self.validation_results["errors"].append(f"{check_name}: {result.get('message', 'Unknown error')}")
            elif status == "warn":
                self.validation_results["checks_warned"] += 1
                self.validation_results["warnings"].append(f"{check_name}: {result.get('message', 'Unknown warning')}")
    
    def _determine_overall_status(self):
        """Determine overall deployment validation status."""
        if self.validation_results["checks_failed"] > 0:
            self.validation_results["overall_status"] = "fail"
            self.validation_results["recommendations"].extend([
                "Fix all failed checks before deployment",
                "Review error messages and system requirements",
                "Consider running validation in a test environment first"
            ])
        elif self.validation_results["checks_warned"] > 0:
            self.validation_results["overall_status"] = "warn"
            self.validation_results["recommendations"].extend([
                "Review warnings and consider addressing them",
                "Monitor performance in production environment",
                "Consider enabling additional monitoring"
            ])
        else:
            self.validation_results["overall_status"] = "pass"
            self.validation_results["recommendations"].extend([
                "System appears ready for deployment",
                "Consider enabling performance monitoring",
                "Review configuration for production settings"
            ])
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive deployment validation report.
        
        Args:
            output_file: Optional path to save report
            
        Returns:
            Report content as string
        """
        results = self.validation_results
        
        report_lines = [
            "CMB Raw Parameter Processing - Deployment Validation Report",
            "=" * 70,
            "",
            f"Overall Status: {results['overall_status'].upper()}",
            f"Checks Passed: {results['checks_passed']}",
            f"Checks Failed: {results['checks_failed']}",
            f"Checks Warned: {results['checks_warned']}",
            ""
        ]
        
        # System checks
        if results["system_checks"]:
            report_lines.extend([
                "System Environment Checks:",
                "-" * 30
            ])
            for check_name, result in results["system_checks"].items():
                status_symbol = "✓" if result["status"] == "pass" else "⚠" if result["status"] == "warn" else "✗"
                report_lines.append(f"{status_symbol} {check_name}: {result['message']}")
            report_lines.append("")
        
        # Dependency checks
        if results["dependency_checks"]:
            report_lines.extend([
                "Dependency Checks:",
                "-" * 20
            ])
            for check_name, result in results["dependency_checks"].items():
                status_symbol = "✓" if result["status"] == "pass" else "⚠" if result["status"] == "warn" else "✗"
                report_lines.append(f"{status_symbol} {check_name}: {result['message']}")
            report_lines.append("")
        
        # Registry checks
        if results["registry_checks"]:
            report_lines.extend([
                "Registry Integration Checks:",
                "-" * 30
            ])
            for check_name, result in results["registry_checks"].items():
                status_symbol = "✓" if result["status"] == "pass" else "⚠" if result["status"] == "warn" else "✗"
                report_lines.append(f"{status_symbol} {check_name}: {result['message']}")
            report_lines.append("")
        
        # Pipeline checks
        if results["pipeline_checks"]:
            report_lines.extend([
                "Pipeline Compatibility Checks:",
                "-" * 35
            ])
            for check_name, result in results["pipeline_checks"].items():
                status_symbol = "✓" if result["status"] == "pass" else "⚠" if result["status"] == "warn" else "✗"
                report_lines.append(f"{status_symbol} {check_name}: {result['message']}")
            report_lines.append("")
        
        # Performance checks
        if results["performance_checks"]:
            report_lines.extend([
                "Performance Checks:",
                "-" * 20
            ])
            for check_name, result in results["performance_checks"].items():
                status_symbol = "✓" if result["status"] == "pass" else "⚠" if result["status"] == "warn" else "✗"
                report_lines.append(f"{status_symbol} {check_name}: {result['message']}")
            report_lines.append("")
        
        # Errors and warnings
        if results["errors"]:
            report_lines.extend([
                "Errors:",
                "-" * 10
            ])
            for error in results["errors"]:
                report_lines.append(f"✗ {error}")
            report_lines.append("")
        
        if results["warnings"]:
            report_lines.extend([
                "Warnings:",
                "-" * 10
            ])
            for warning in results["warnings"]:
                report_lines.append(f"⚠ {warning}")
            report_lines.append("")
        
        # Recommendations
        if results["recommendations"]:
            report_lines.extend([
                "Recommendations:",
                "-" * 15
            ])
            for i, recommendation in enumerate(results["recommendations"], 1):
                report_lines.append(f"{i}. {recommendation}")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_content)
        
        return report_content


def main():
    """Main deployment validation script."""
    parser = argparse.ArgumentParser(
        description="Validate CMB raw parameter processing deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output report file path"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    try:
        # Run validation
        validator = DeploymentValidator(args.config)
        
        if not args.quiet:
            print("Starting deployment validation...")
        
        results = validator.run_all_checks()
        
        # Output results
        if args.json:
            output_content = json.dumps(results, indent=2, default=str)
        else:
            output_content = validator.generate_report()
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_content)
            
            if not args.quiet:
                print(f"Report saved to: {args.output}")
        else:
            print(output_content)
        
        # Exit with appropriate code
        if results["overall_status"] == "fail":
            sys.exit(1)
        elif results["overall_status"] == "warn":
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"Deployment validation failed: {e}")
        if not args.quiet:
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()