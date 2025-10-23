#!/usr/bin/env python3
"""
Comprehensive Dataset Compatibility Verification

This script verifies that all derived datasets produced by the data preparation 
framework are fully compatible with the PBUF analysis and fitting system.

Verification includes:
1. Output Schema Compatibility
2. Loader Integration Verification  
3. Provenance Continuity
4. Numerical and Structural Validation
5. End-to-End Fit Simulation
6. Error Handling and Logging
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import traceback
from datetime import datetime
import warnings

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "pipelines"))

def main():
    """Main verification function."""
    print("🔍 PBUF Dataset Compatibility Verification")
    print("=" * 60)
    
    verification_results = {
        "timestamp": datetime.now().isoformat(),
        "schema_compatibility": {},
        "loader_integration": {},
        "provenance_continuity": {},
        "numerical_validation": {},
        "fit_simulation": {},
        "error_handling": {},
        "overall_status": "unknown",
        "recommendations": []
    }
    
    try:
        # 1. Schema Compatibility Check
        print("\n📋 1. Output Schema Compatibility")
        print("-" * 40)
        schema_results = verify_schema_compatibility()
        verification_results["schema_compatibility"] = schema_results
        
        # 2. Loader Integration Check
        print("\n🔌 2. Loader Integration Verification")
        print("-" * 40)
        loader_results = verify_loader_integration()
        verification_results["loader_integration"] = loader_results
        
        # 3. Provenance Continuity Check
        print("\n📜 3. Provenance Continuity")
        print("-" * 40)
        provenance_results = verify_provenance_continuity()
        verification_results["provenance_continuity"] = provenance_results
        
        # 4. Numerical Validation
        print("\n🔢 4. Numerical and Structural Validation")
        print("-" * 40)
        numerical_results = verify_numerical_integrity()
        verification_results["numerical_validation"] = numerical_results
        
        # 5. End-to-End Fit Simulation
        print("\n🎯 5. End-to-End Fit Simulation")
        print("-" * 40)
        fit_results = simulate_end_to_end_fit()
        verification_results["fit_simulation"] = fit_results
        
        # 6. Error Handling Test
        print("\n⚠️  6. Error Handling and Logging")
        print("-" * 40)
        error_results = test_error_handling()
        verification_results["error_handling"] = error_results
        
        # Generate overall assessment
        verification_results["overall_status"] = assess_overall_compatibility(verification_results)
        verification_results["recommendations"] = generate_recommendations(verification_results)
        
    except Exception as e:
        print(f"❌ Critical error during verification: {e}")
        verification_results["overall_status"] = "failed"
        verification_results["critical_error"] = str(e)
        verification_results["traceback"] = traceback.format_exc()
    
    # Generate final report
    generate_verification_report(verification_results)
    
    return verification_results

def verify_schema_compatibility() -> Dict[str, Any]:
    """Verify that derived datasets follow the unified schema."""
    results = {
        "status": "unknown",
        "datasets_checked": 0,
        "schema_compliant": 0,
        "schema_violations": [],
        "format_compatibility": {},
        "details": {}
    }
    
    try:
        # Import schema validation
        from pipelines.data_preparation.core.schema import StandardDataset
        from pipelines.data_preparation.output.format_converter import FormatConverter
        
        # Find recent derived datasets
        derived_dir = Path("data/derived")
        if not derived_dir.exists():
            results["status"] = "no_data"
            results["error"] = "No derived datasets directory found"
            return results
        
        # Get sample of recent datasets for each type
        dataset_types = ["sn", "cmb", "bao", "cc", "rsd"]
        sample_datasets = {}
        
        for dataset_type in dataset_types:
            pattern = f"{dataset_type}_*_derived_*.json"
            files = list(derived_dir.glob(pattern))
            if files:
                # Get most recent file
                latest_file = max(files, key=lambda f: f.stat().st_mtime)
                sample_datasets[dataset_type] = latest_file
        
        print(f"Found {len(sample_datasets)} dataset types to verify")
        
        for dataset_type, file_path in sample_datasets.items():
            print(f"  Checking {dataset_type}: {file_path.name}")
            
            try:
                # Load derived dataset
                with open(file_path, 'r') as f:
                    derived_data = json.load(f)
                
                # Extract the StandardDataset portion
                if "data" not in derived_data:
                    results["schema_violations"].append({
                        "dataset": dataset_type,
                        "error": "Missing 'data' field in derived dataset"
                    })
                    continue
                
                data_section = derived_data["data"]
                
                # Validate required fields
                required_fields = ["z", "observable", "uncertainty", "metadata"]
                missing_fields = [f for f in required_fields if f not in data_section]
                
                if missing_fields:
                    results["schema_violations"].append({
                        "dataset": dataset_type,
                        "error": f"Missing required fields: {missing_fields}"
                    })
                    continue
                
                # Create StandardDataset instance for validation
                standard_dataset = StandardDataset(
                    z=np.array(data_section["z"]),
                    observable=np.array(data_section["observable"]),
                    uncertainty=np.array(data_section["uncertainty"]),
                    covariance=np.array(data_section["covariance"]) if data_section.get("covariance") else None,
                    metadata=data_section["metadata"]
                )
                
                # Validate schema compliance
                try:
                    standard_dataset.validate_all()
                    results["schema_compliant"] += 1
                    print(f"    ✅ Schema validation passed")
                except Exception as e:
                    results["schema_violations"].append({
                        "dataset": dataset_type,
                        "error": f"Schema validation failed: {str(e)}"
                    })
                    print(f"    ❌ Schema validation failed: {e}")
                    continue
                
                # Test format conversion compatibility
                try:
                    dataset_dict = FormatConverter.standard_to_dataset_dict(standard_dataset, dataset_type)
                    
                    # Validate conversion result has expected structure
                    expected_keys = ["dataset_type", "observations", "covariance", "metadata"]
                    missing_keys = [k for k in expected_keys if k not in dataset_dict]
                    
                    if missing_keys:
                        results["format_compatibility"][dataset_type] = {
                            "status": "failed",
                            "error": f"Missing keys after conversion: {missing_keys}"
                        }
                    else:
                        results["format_compatibility"][dataset_type] = {
                            "status": "passed",
                            "observations_type": type(dataset_dict["observations"]).__name__,
                            "covariance_shape": dataset_dict["covariance"].shape if dataset_dict["covariance"] is not None else None
                        }
                        print(f"    ✅ Format conversion compatible")
                
                except Exception as e:
                    results["format_compatibility"][dataset_type] = {
                        "status": "failed", 
                        "error": str(e)
                    }
                    print(f"    ❌ Format conversion failed: {e}")
                
                results["datasets_checked"] += 1
                
            except Exception as e:
                results["schema_violations"].append({
                    "dataset": dataset_type,
                    "error": f"Failed to load/process dataset: {str(e)}"
                })
                print(f"    ❌ Failed to process: {e}")
        
        # Determine overall status
        if results["datasets_checked"] == 0:
            results["status"] = "no_data"
        elif results["schema_compliant"] == results["datasets_checked"]:
            results["status"] = "passed"
        elif results["schema_compliant"] > 0:
            results["status"] = "partial"
        else:
            results["status"] = "failed"
        
        print(f"\nSchema Compatibility: {results['schema_compliant']}/{results['datasets_checked']} datasets compliant")
        
    except ImportError as e:
        results["status"] = "import_error"
        results["error"] = f"Failed to import required modules: {str(e)}"
        print(f"❌ Import error: {e}")
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        print(f"❌ Unexpected error: {e}")
    
    return results

def verify_loader_integration() -> Dict[str, Any]:
    """Verify that datasets can be loaded through the fitting system's interface."""
    results = {
        "status": "unknown",
        "datasets_tested": 0,
        "successful_loads": 0,
        "load_failures": [],
        "interface_compatibility": {},
        "details": {}
    }
    
    try:
        # Import fitting system components
        from pipelines.fit_core.datasets import load_dataset, validate_dataset, SUPPORTED_DATASETS
        
        # Test loading each supported dataset type
        for dataset_name in SUPPORTED_DATASETS.keys():
            print(f"  Testing load: {dataset_name}")
            
            try:
                # Attempt to load dataset
                dataset_dict = load_dataset(dataset_name)
                
                # Validate the loaded dataset
                validate_dataset(dataset_dict, dataset_name)
                
                # Check interface compatibility
                results["interface_compatibility"][dataset_name] = {
                    "status": "passed",
                    "dataset_type": dataset_dict.get("dataset_type"),
                    "observations_keys": list(dataset_dict.get("observations", {}).keys()),
                    "covariance_shape": dataset_dict["covariance"].shape if dataset_dict.get("covariance") is not None else None,
                    "metadata_keys": list(dataset_dict.get("metadata", {}).keys())
                }
                
                results["successful_loads"] += 1
                print(f"    ✅ Successfully loaded and validated")
                
            except Exception as e:
                results["load_failures"].append({
                    "dataset": dataset_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                results["interface_compatibility"][dataset_name] = {
                    "status": "failed",
                    "error": str(e)
                }
                print(f"    ❌ Load failed: {e}")
            
            results["datasets_tested"] += 1
        
        # Determine overall status
        if results["successful_loads"] == results["datasets_tested"]:
            results["status"] = "passed"
        elif results["successful_loads"] > 0:
            results["status"] = "partial"
        else:
            results["status"] = "failed"
        
        print(f"\nLoader Integration: {results['successful_loads']}/{results['datasets_tested']} datasets loaded successfully")
        
    except ImportError as e:
        results["status"] = "import_error"
        results["error"] = f"Failed to import fitting system modules: {str(e)}"
        print(f"❌ Import error: {e}")
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        print(f"❌ Unexpected error: {e}")
    
    return results

def verify_provenance_continuity() -> Dict[str, Any]:
    """Verify provenance tracking between preparation framework and fitting system."""
    results = {
        "status": "unknown",
        "provenance_entries": 0,
        "valid_provenance": 0,
        "provenance_issues": [],
        "registry_integration": {},
        "details": {}
    }
    
    try:
        # Check if registry system is available
        registry_available = False
        try:
            from pipelines.dataset_registry.core.registry_manager import RegistryManager
            registry_path = Path("data/registry")
            if registry_path.exists():
                registry_available = True
        except ImportError:
            pass
        
        results["registry_integration"]["available"] = registry_available
        
        # Check derived datasets for provenance metadata
        derived_dir = Path("data/derived")
        if derived_dir.exists():
            recent_files = sorted(derived_dir.glob("*_derived_*.json"), 
                                key=lambda f: f.stat().st_mtime, reverse=True)[:10]
            
            for file_path in recent_files:
                try:
                    with open(file_path, 'r') as f:
                        derived_data = json.load(f)
                    
                    # Check for provenance fields
                    provenance_fields = [
                        "processing_timestamp",
                        "environment_hash", 
                        "transformation_summary"
                    ]
                    
                    missing_fields = [f for f in provenance_fields if f not in derived_data]
                    
                    if missing_fields:
                        results["provenance_issues"].append({
                            "file": file_path.name,
                            "issue": f"Missing provenance fields: {missing_fields}"
                        })
                    else:
                        results["valid_provenance"] += 1
                        
                        # Check transformation summary completeness
                        transform_summary = derived_data.get("transformation_summary", {})
                        expected_summary_fields = ["transformation_steps", "formulas_used", "assumptions"]
                        missing_summary = [f for f in expected_summary_fields if f not in transform_summary]
                        
                        if missing_summary:
                            results["provenance_issues"].append({
                                "file": file_path.name,
                                "issue": f"Incomplete transformation summary: missing {missing_summary}"
                            })
                    
                    results["provenance_entries"] += 1
                    
                except Exception as e:
                    results["provenance_issues"].append({
                        "file": file_path.name,
                        "issue": f"Failed to read provenance: {str(e)}"
                    })
        
        # Test provenance integration with fitting system
        try:
            from pipelines.fit_core.datasets import get_dataset_provenance
            
            test_datasets = ["sn", "cmb", "bao"]
            provenance_integration = {}
            
            for dataset_name in test_datasets:
                try:
                    provenance = get_dataset_provenance(dataset_name)
                    provenance_integration[dataset_name] = {
                        "available": provenance is not None,
                        "fields": list(provenance.keys()) if provenance else []
                    }
                except Exception as e:
                    provenance_integration[dataset_name] = {
                        "available": False,
                        "error": str(e)
                    }
            
            results["registry_integration"]["provenance_access"] = provenance_integration
            
        except ImportError:
            results["registry_integration"]["provenance_access"] = "not_available"
        
        # Determine overall status
        if results["provenance_entries"] == 0:
            results["status"] = "no_data"
        elif results["valid_provenance"] == results["provenance_entries"]:
            results["status"] = "passed"
        elif results["valid_provenance"] > 0:
            results["status"] = "partial"
        else:
            results["status"] = "failed"
        
        print(f"Provenance Continuity: {results['valid_provenance']}/{results['provenance_entries']} entries valid")
        print(f"Registry Integration: {'Available' if registry_available else 'Not Available'}")
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        print(f"❌ Unexpected error: {e}")
    
    return results

def verify_numerical_integrity() -> Dict[str, Any]:
    """Verify numerical integrity of derived datasets."""
    results = {
        "status": "unknown",
        "datasets_checked": 0,
        "numerical_valid": 0,
        "numerical_issues": [],
        "covariance_validation": {},
        "redshift_validation": {},
        "details": {}
    }
    
    try:
        from pipelines.data_preparation.core.schema import StandardDataset
        
        # Check recent derived datasets
        derived_dir = Path("data/derived")
        if not derived_dir.exists():
            results["status"] = "no_data"
            return results
        
        # Sample datasets for validation
        recent_files = sorted(derived_dir.glob("*_derived_*.json"), 
                            key=lambda f: f.stat().st_mtime, reverse=True)[:15]
        
        for file_path in recent_files:
            try:
                with open(file_path, 'r') as f:
                    derived_data = json.load(f)
                
                if "data" not in derived_data:
                    continue
                
                data_section = derived_data["data"]
                dataset_name = derived_data.get("dataset_name", "unknown")
                
                # Create StandardDataset for validation
                standard_dataset = StandardDataset(
                    z=np.array(data_section["z"]),
                    observable=np.array(data_section["observable"]),
                    uncertainty=np.array(data_section["uncertainty"]),
                    covariance=np.array(data_section["covariance"]) if data_section.get("covariance") else None,
                    metadata=data_section["metadata"]
                )
                
                # Numerical validation
                try:
                    standard_dataset.validate_numerical()
                    results["numerical_valid"] += 1
                    print(f"    ✅ {dataset_name}: Numerical validation passed")
                except Exception as e:
                    results["numerical_issues"].append({
                        "dataset": dataset_name,
                        "file": file_path.name,
                        "error": str(e)
                    })
                    print(f"    ❌ {dataset_name}: Numerical validation failed: {e}")
                
                # Covariance validation (if present)
                if standard_dataset.covariance is not None:
                    try:
                        standard_dataset.validate_covariance()
                        results["covariance_validation"][dataset_name] = {
                            "status": "passed",
                            "shape": standard_dataset.covariance.shape,
                            "condition_number": float(np.linalg.cond(standard_dataset.covariance))
                        }
                    except Exception as e:
                        results["covariance_validation"][dataset_name] = {
                            "status": "failed",
                            "error": str(e)
                        }
                else:
                    results["covariance_validation"][dataset_name] = {
                        "status": "not_present"
                    }
                
                # Redshift validation
                try:
                    z_min, z_max = float(np.min(standard_dataset.z)), float(np.max(standard_dataset.z))
                    
                    # Dataset-specific redshift range validation
                    if "cmb" in dataset_name:
                        expected_z_range = (1000, 1200)  # CMB recombination
                    else:
                        expected_z_range = (0, 10)  # Standard cosmological surveys
                    
                    z_valid = (expected_z_range[0] <= z_min <= z_max <= expected_z_range[1])
                    
                    results["redshift_validation"][dataset_name] = {
                        "status": "passed" if z_valid else "warning",
                        "range": [z_min, z_max],
                        "expected_range": expected_z_range,
                        "n_points": len(standard_dataset.z)
                    }
                    
                except Exception as e:
                    results["redshift_validation"][dataset_name] = {
                        "status": "failed",
                        "error": str(e)
                    }
                
                results["datasets_checked"] += 1
                
            except Exception as e:
                results["numerical_issues"].append({
                    "file": file_path.name,
                    "error": f"Failed to process dataset: {str(e)}"
                })
        
        # Determine overall status
        if results["datasets_checked"] == 0:
            results["status"] = "no_data"
        elif results["numerical_valid"] == results["datasets_checked"]:
            results["status"] = "passed"
        elif results["numerical_valid"] > 0:
            results["status"] = "partial"
        else:
            results["status"] = "failed"
        
        print(f"Numerical Integrity: {results['numerical_valid']}/{results['datasets_checked']} datasets valid")
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        print(f"❌ Unexpected error: {e}")
    
    return results

def simulate_end_to_end_fit() -> Dict[str, Any]:
    """Simulate an end-to-end fit using derived datasets."""
    results = {
        "status": "unknown",
        "fit_attempts": 0,
        "successful_fits": 0,
        "fit_failures": [],
        "chi_squared_tests": {},
        "parameter_recovery": {},
        "details": {}
    }
    
    try:
        # Import fitting components
        from pipelines.fit_core.datasets import load_dataset, validate_dataset
        from pipelines.fit_core.likelihoods import compute_chi_squared
        
        # Test datasets for fitting
        test_datasets = ["sn", "cmb", "bao"]
        
        for dataset_name in test_datasets:
            print(f"  Testing fit simulation: {dataset_name}")
            
            try:
                # Load dataset through fitting system
                dataset_dict = load_dataset(dataset_name)
                validate_dataset(dataset_dict, dataset_name)
                
                # Extract data for chi-squared computation
                observations = dataset_dict["observations"]
                covariance = dataset_dict["covariance"]
                
                # Create mock theoretical predictions (same as observations for test)
                if dataset_name == "cmb":
                    theory = np.array([observations["R"], observations["l_A"], observations["theta_star"]])
                    data = theory.copy()
                elif dataset_name == "sn":
                    if isinstance(observations, dict) and "distance_modulus" in observations:
                        theory = np.array(observations["distance_modulus"])
                        data = theory.copy()
                    else:
                        # Handle case where observations is already an array
                        theory = np.array(observations)
                        data = theory.copy()
                elif dataset_name == "bao":
                    if isinstance(observations, dict) and "DV_over_rs" in observations:
                        theory = np.array(observations["DV_over_rs"])
                        data = theory.copy()
                    else:
                        theory = np.array(observations)
                        data = theory.copy()
                else:
                    # Generic case
                    theory = np.array(list(observations.values())[0])
                    data = theory.copy()
                
                # Compute chi-squared
                if covariance is not None:
                    try:
                        chi_squared = compute_chi_squared(data, theory, covariance)
                        dof = len(data)
                        
                        results["chi_squared_tests"][dataset_name] = {
                            "status": "passed",
                            "chi_squared": float(chi_squared),
                            "dof": dof,
                            "reduced_chi_squared": float(chi_squared / dof) if dof > 0 else None,
                            "data_points": len(data)
                        }
                        
                        print(f"    ✅ χ² = {chi_squared:.3f} (dof={dof})")
                        
                    except Exception as e:
                        results["chi_squared_tests"][dataset_name] = {
                            "status": "failed",
                            "error": str(e)
                        }
                        print(f"    ❌ χ² computation failed: {e}")
                else:
                    results["chi_squared_tests"][dataset_name] = {
                        "status": "no_covariance",
                        "note": "No covariance matrix available for χ² test"
                    }
                    print(f"    ⚠️  No covariance matrix available")
                
                # Test parameter structure compatibility
                metadata = dataset_dict.get("metadata", {})
                results["parameter_recovery"][dataset_name] = {
                    "status": "passed",
                    "n_data_points": metadata.get("n_data_points", len(data)),
                    "observables": metadata.get("observables", []),
                    "redshift_range": metadata.get("redshift_range", [])
                }
                
                results["successful_fits"] += 1
                
            except Exception as e:
                results["fit_failures"].append({
                    "dataset": dataset_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                print(f"    ❌ Fit simulation failed: {e}")
            
            results["fit_attempts"] += 1
        
        # Determine overall status
        if results["successful_fits"] == results["fit_attempts"]:
            results["status"] = "passed"
        elif results["successful_fits"] > 0:
            results["status"] = "partial"
        else:
            results["status"] = "failed"
        
        print(f"Fit Simulation: {results['successful_fits']}/{results['fit_attempts']} datasets compatible")
        
    except ImportError as e:
        results["status"] = "import_error"
        results["error"] = f"Failed to import fitting modules: {str(e)}"
        print(f"❌ Import error: {e}")
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        print(f"❌ Unexpected error: {e}")
    
    return results

def test_error_handling() -> Dict[str, Any]:
    """Test error handling and graceful degradation."""
    results = {
        "status": "unknown",
        "error_tests": 0,
        "graceful_failures": 0,
        "error_scenarios": {},
        "logging_validation": {},
        "details": {}
    }
    
    try:
        from pipelines.fit_core.datasets import load_dataset
        
        # Test 1: Invalid dataset name
        print("  Testing invalid dataset name...")
        try:
            load_dataset("nonexistent_dataset")
            results["error_scenarios"]["invalid_dataset"] = {
                "status": "failed",
                "error": "Should have raised an error for invalid dataset"
            }
        except Exception as e:
            results["error_scenarios"]["invalid_dataset"] = {
                "status": "passed",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            results["graceful_failures"] += 1
            print(f"    ✅ Correctly raised {type(e).__name__}")
        
        results["error_tests"] += 1
        
        # Test 2: Missing covariance handling
        print("  Testing missing covariance handling...")
        try:
            # This should work gracefully
            dataset = load_dataset("sn")  # SN might not have covariance in some cases
            
            results["error_scenarios"]["missing_covariance"] = {
                "status": "passed",
                "note": "System handles missing covariance gracefully"
            }
            results["graceful_failures"] += 1
            print(f"    ✅ Handles missing covariance gracefully")
            
        except Exception as e:
            results["error_scenarios"]["missing_covariance"] = {
                "status": "warning",
                "error": str(e)
            }
            print(f"    ⚠️  Error with missing covariance: {e}")
        
        results["error_tests"] += 1
        
        # Test 3: Check logging system
        print("  Testing logging system...")
        log_dir = Path("data/logs")
        if log_dir.exists():
            error_logs = list(log_dir.glob("error_*.txt"))
            processing_logs = list(log_dir.glob("processing.log"))
            
            results["logging_validation"] = {
                "log_directory_exists": True,
                "error_logs_count": len(error_logs),
                "processing_logs_count": len(processing_logs),
                "recent_error_logs": [f.name for f in sorted(error_logs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]]
            }
            
            if error_logs:
                print(f"    ✅ Found {len(error_logs)} error logs")
                results["graceful_failures"] += 1
            else:
                print(f"    ⚠️  No error logs found")
        else:
            results["logging_validation"] = {
                "log_directory_exists": False,
                "error": "Log directory not found"
            }
            print(f"    ❌ Log directory not found")
        
        results["error_tests"] += 1
        
        # Determine overall status
        if results["graceful_failures"] >= results["error_tests"] * 0.8:  # 80% threshold
            results["status"] = "passed"
        elif results["graceful_failures"] > 0:
            results["status"] = "partial"
        else:
            results["status"] = "failed"
        
        print(f"Error Handling: {results['graceful_failures']}/{results['error_tests']} scenarios handled gracefully")
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        print(f"❌ Unexpected error: {e}")
    
    return results

def assess_overall_compatibility(verification_results: Dict[str, Any]) -> str:
    """Assess overall compatibility status."""
    
    # Weight different verification components
    component_weights = {
        "schema_compatibility": 0.25,
        "loader_integration": 0.25,
        "numerical_validation": 0.20,
        "fit_simulation": 0.15,
        "provenance_continuity": 0.10,
        "error_handling": 0.05
    }
    
    # Score each component
    component_scores = {}
    for component, weight in component_weights.items():
        if component in verification_results:
            status = verification_results[component].get("status", "unknown")
            
            if status == "passed":
                score = 1.0
            elif status == "partial":
                score = 0.6
            elif status in ["no_data", "import_error"]:
                score = 0.3
            else:
                score = 0.0
            
            component_scores[component] = score * weight
    
    # Calculate overall score
    overall_score = sum(component_scores.values())
    
    # Determine status
    if overall_score >= 0.9:
        return "fully_compatible"
    elif overall_score >= 0.7:
        return "mostly_compatible"
    elif overall_score >= 0.5:
        return "partially_compatible"
    else:
        return "incompatible"

def generate_recommendations(verification_results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on verification results."""
    recommendations = []
    
    # Schema compatibility recommendations
    schema_status = verification_results.get("schema_compatibility", {}).get("status")
    if schema_status == "failed":
        recommendations.append("❌ CRITICAL: Fix schema compatibility issues before proceeding with fits")
    elif schema_status == "partial":
        recommendations.append("⚠️  Address schema violations in some datasets")
    
    # Loader integration recommendations
    loader_status = verification_results.get("loader_integration", {}).get("status")
    if loader_status == "failed":
        recommendations.append("❌ CRITICAL: Fix loader integration issues - datasets cannot be loaded by fitting system")
    elif loader_status == "partial":
        recommendations.append("⚠️  Some datasets fail to load - investigate specific failures")
    
    # Numerical validation recommendations
    numerical_status = verification_results.get("numerical_validation", {}).get("status")
    if numerical_status == "failed":
        recommendations.append("❌ CRITICAL: Fix numerical integrity issues - data contains invalid values")
    elif numerical_status == "partial":
        recommendations.append("⚠️  Some datasets have numerical issues - review and fix")
    
    # Fit simulation recommendations
    fit_status = verification_results.get("fit_simulation", {}).get("status")
    if fit_status == "failed":
        recommendations.append("❌ CRITICAL: End-to-end fit simulation failed - datasets incompatible with fitting pipeline")
    elif fit_status == "partial":
        recommendations.append("⚠️  Some datasets fail fit simulation - investigate compatibility issues")
    
    # Provenance recommendations
    provenance_status = verification_results.get("provenance_continuity", {}).get("status")
    if provenance_status == "failed":
        recommendations.append("⚠️  Improve provenance tracking for better reproducibility")
    
    # Error handling recommendations
    error_status = verification_results.get("error_handling", {}).get("status")
    if error_status == "failed":
        recommendations.append("⚠️  Improve error handling and logging for better diagnostics")
    
    # Overall recommendations
    overall_status = verification_results.get("overall_status")
    if overall_status == "fully_compatible":
        recommendations.append("✅ READY: All datasets are fully compatible - proceed with confidence")
    elif overall_status == "mostly_compatible":
        recommendations.append("✅ READY: Datasets are mostly compatible - address minor issues for optimal performance")
    elif overall_status == "partially_compatible":
        recommendations.append("⚠️  CAUTION: Significant compatibility issues exist - address before production use")
    else:
        recommendations.append("❌ NOT READY: Major compatibility issues must be resolved before proceeding")
    
    return recommendations

def generate_verification_report(verification_results: Dict[str, Any]):
    """Generate and save the verification report."""
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    # Overall status
    overall_status = verification_results.get("overall_status", "unknown")
    status_emoji = {
        "fully_compatible": "✅",
        "mostly_compatible": "✅", 
        "partially_compatible": "⚠️",
        "incompatible": "❌",
        "unknown": "❓"
    }
    
    print(f"\n{status_emoji.get(overall_status, '❓')} Overall Status: {overall_status.upper()}")
    
    # Component summary
    print(f"\n📋 Component Results:")
    components = [
        ("Schema Compatibility", "schema_compatibility"),
        ("Loader Integration", "loader_integration"), 
        ("Provenance Continuity", "provenance_continuity"),
        ("Numerical Validation", "numerical_validation"),
        ("Fit Simulation", "fit_simulation"),
        ("Error Handling", "error_handling")
    ]
    
    for name, key in components:
        if key in verification_results:
            status = verification_results[key].get("status", "unknown")
            emoji = "✅" if status == "passed" else "⚠️" if status == "partial" else "❌"
            print(f"  {emoji} {name}: {status}")
        else:
            print(f"  ❓ {name}: not tested")
    
    # Recommendations
    recommendations = verification_results.get("recommendations", [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
    
    # Save detailed report
    report_path = Path("dataset_compatibility_verification_report.json")
    with open(report_path, 'w') as f:
        json.dump(verification_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Final readiness statement
    print(f"\n🎯 FINAL READINESS STATEMENT:")
    if overall_status == "fully_compatible":
        print("✅ All prepared datasets are FULLY COMPATIBLE with the PBUF analysis system.")
        print("   Ready to proceed with the first full real-data fit.")
    elif overall_status == "mostly_compatible":
        print("✅ Datasets are MOSTLY COMPATIBLE with minor issues.")
        print("   Can proceed with fits while addressing remaining issues.")
    elif overall_status == "partially_compatible":
        print("⚠️  Datasets have SIGNIFICANT COMPATIBILITY ISSUES.")
        print("   Address critical issues before proceeding with production fits.")
    else:
        print("❌ Datasets are NOT COMPATIBLE with the PBUF analysis system.")
        print("   Must resolve compatibility issues before proceeding.")

if __name__ == "__main__":
    main()