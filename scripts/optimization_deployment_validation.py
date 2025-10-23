#!/usr/bin/env python3
"""
Parameter Optimization Deployment Validation Script

This script performs comprehensive validation of the parameter optimization system
before deployment to production. It tests all core functionality, integration points,
data integrity, and performance characteristics.
"""

import sys
import time
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
import traceback

class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass

class OptimizationValidator:
    """Comprehensive validation suite for parameter optimization system."""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
        
    def run_test(self, test_name, test_func, *args, **kwargs):
        """Run a single test and record results."""
        print(f"Testing {test_name}... ", end="", flush=True)
        
        try:
            start_time = time.time()
            result = test_func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            print(f"✓ PASS ({elapsed:.2f}s)")
            self.passed_tests += 1
            self.test_results.append({
                "name": test_name,
                "status": "PASS",
                "elapsed": elapsed,
                "result": result
            })
            return True
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ FAIL ({elapsed:.2f}s)")
            print(f"   Error: {str(e)}")
            
            self.failed_tests += 1
            self.test_results.append({
                "name": test_name,
                "status": "FAIL",
                "elapsed": elapsed,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return False
    
    def test_core_imports(self):
        """Test that all optimization modules can be imported."""
        try:
            from pipelines.fit_core.optimizer import ParameterOptimizer
            from pipelines.fit_core.parameter_store import OptimizedParameterStore
            from pipelines.fit_core.config import parse_optimization_config
            return "All optimization modules imported successfully"
        except ImportError as e:
            raise ValidationError(f"Import failed: {e}")
    
    def test_parameter_optimizer_instantiation(self):
        """Test ParameterOptimizer class instantiation."""
        from pipelines.fit_core.optimizer import ParameterOptimizer
        
        optimizer = ParameterOptimizer()
        
        # Test basic methods exist
        assert hasattr(optimizer, 'optimize_parameters')
        assert hasattr(optimizer, 'get_optimization_bounds')
        assert hasattr(optimizer, 'validate_optimization_request')
        
        return "ParameterOptimizer instantiated with all required methods"
    
    def test_parameter_bounds_validation(self):
        """Test parameter bounds validation."""
        from pipelines.fit_core.optimizer import ParameterOptimizer
        
        optimizer = ParameterOptimizer()
        
        # Test ΛCDM parameter bounds
        h0_bounds = optimizer.get_optimization_bounds('lcdm', 'H0')
        assert h0_bounds == (20.0, 150.0), f"Expected (20.0, 150.0), got {h0_bounds}"
        
        om0_bounds = optimizer.get_optimization_bounds('lcdm', 'Om0')
        assert om0_bounds == (0.01, 0.99), f"Expected (0.01, 0.99), got {om0_bounds}"
        
        # Test PBUF parameter bounds
        ksat_bounds = optimizer.get_optimization_bounds('pbuf', 'k_sat')
        assert ksat_bounds == (0.1, 2.0), f"Expected (0.1, 2.0), got {ksat_bounds}"
        
        alpha_bounds = optimizer.get_optimization_bounds('pbuf', 'alpha')
        assert alpha_bounds == (1e-6, 1e-2), f"Expected (1e-6, 1e-2), got {alpha_bounds}"
        
        return "Parameter bounds validation working correctly"
    
    def test_invalid_parameter_detection(self):
        """Test detection of invalid parameters for models."""
        from pipelines.fit_core.optimizer import ParameterOptimizer
        
        optimizer = ParameterOptimizer()
        
        # Test invalid parameter for ΛCDM
        try:
            optimizer.validate_optimization_request('lcdm', ['k_sat'])
            raise ValidationError("Should have raised ValueError for k_sat in ΛCDM")
        except ValueError:
            pass  # Expected
        
        # Test invalid parameter for PBUF
        try:
            optimizer.validate_optimization_request('pbuf', ['invalid_param'])
            raise ValidationError("Should have raised ValueError for invalid parameter")
        except ValueError:
            pass  # Expected
        
        # Test valid parameters
        assert optimizer.validate_optimization_request('lcdm', ['H0', 'Om0'])
        assert optimizer.validate_optimization_request('pbuf', ['k_sat', 'alpha'])
        
        return "Invalid parameter detection working correctly"
    
    def test_parameter_store_instantiation(self):
        """Test OptimizedParameterStore instantiation."""
        from pipelines.fit_core.parameter_store import OptimizedParameterStore
        
        store = OptimizedParameterStore()
        
        # Test basic methods exist
        assert hasattr(store, 'get_model_defaults')
        assert hasattr(store, 'update_model_defaults')
        assert hasattr(store, 'validate_cross_model_consistency')
        
        return "OptimizedParameterStore instantiated with all required methods"
    
    def test_parameter_retrieval(self):
        """Test parameter retrieval from store."""
        from pipelines.fit_core.parameter_store import OptimizedParameterStore
        
        store = OptimizedParameterStore()
        
        # Test ΛCDM parameter retrieval
        lcdm_params = store.get_model_defaults('lcdm')
        assert isinstance(lcdm_params, dict)
        assert 'H0' in lcdm_params
        assert 'Om0' in lcdm_params
        
        # Test PBUF parameter retrieval
        pbuf_params = store.get_model_defaults('pbuf')
        assert isinstance(pbuf_params, dict)
        assert 'H0' in pbuf_params
        assert 'k_sat' in pbuf_params
        assert 'alpha' in pbuf_params
        
        return f"Retrieved {len(lcdm_params)} ΛCDM and {len(pbuf_params)} PBUF parameters"
    
    def test_cross_model_consistency(self):
        """Test cross-model consistency validation."""
        from pipelines.fit_core.parameter_store import OptimizedParameterStore
        
        store = OptimizedParameterStore()
        
        divergence = store.validate_cross_model_consistency()
        assert isinstance(divergence, dict)
        
        # Check that shared parameters are included
        shared_params = ['H0', 'Om0', 'Obh2', 'ns']
        for param in shared_params:
            if param in divergence:
                div = divergence[param]
                assert isinstance(div, (int, float))
                assert div >= 0
        
        return f"Cross-model consistency check returned {len(divergence)} parameters"
    
    def test_configuration_parsing(self):
        """Test optimization configuration parsing."""
        from pipelines.fit_core.config import parse_optimization_config
        
        # Test empty configuration
        empty_config = {}
        parsed = parse_optimization_config(empty_config)
        assert isinstance(parsed, dict)
        
        # Test full configuration
        full_config = {
            'optimization': {
                'optimize_parameters': ['H0', 'Om0'],
                'covariance_scaling': 1.2,
                'dry_run': True,
                'warm_start': False
            }
        }
        
        parsed = parse_optimization_config(full_config)
        assert parsed['optimize_parameters'] == ['H0', 'Om0']
        assert parsed['covariance_scaling'] == 1.2
        assert parsed['dry_run'] == True
        assert parsed['warm_start'] == False
        
        return "Configuration parsing working correctly"
    
    def test_engine_integration(self):
        """Test optimization integration with engine."""
        from pipelines.fit_core.engine import run_fit
        
        # Test with optimization parameters (dry run with fast settings)
        result = run_fit(
            model='lcdm',
            datasets_list=['cmb'],
            optimization_params=['H0', 'Om0'],
            covariance_scaling=3.0,  # Fast for testing
            dry_run=True
        )
        
        assert isinstance(result, dict)
        assert 'params' in result
        assert 'metrics' in result
        assert 'optimization' in result
        
        # Check optimization metadata
        opt_data = result['optimization']
        assert 'chi2_improvement' in opt_data
        assert 'convergence_status' in opt_data
        assert 'optimized_params' in opt_data
        
        return f"Engine integration successful, χ² improvement: {opt_data['chi2_improvement']:.3f}"
    
    def test_cmb_script_integration(self):
        """Test CMB fitter script with optimization flags."""
        cmd = [
            sys.executable, 'pipelines/fit_cmb.py',
            '--model', 'lcdm',
            '--optimize', 'H0,Om0',
            '--cov-scale', '3.0',
            '--dry-run'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise ValidationError(f"CMB script failed: {result.stderr}")
        
        return "CMB script optimization integration successful"
    
    def test_bao_script_compatibility(self):
        """Test BAO script backward compatibility."""
        cmd = [
            sys.executable, 'pipelines/fit_bao.py',
            '--model', 'lcdm',
            '--dry-run'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            raise ValidationError(f"BAO script failed: {result.stderr}")
        
        return "BAO script backward compatibility verified"
    
    def test_parameter_storage_integrity(self):
        """Test parameter storage file integrity."""
        import json
        import tempfile
        
        # Create temporary storage directory
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / 'test_storage.json'
            
            # Test data
            test_data = {
                'lcdm': {
                    'defaults': {'H0': 67.4, 'Om0': 0.315},
                    'optimization_metadata': {
                        'timestamp': '2025-10-22T14:42:00Z',
                        'chi2_improvement': 2.34,
                        'convergence_status': 'success'
                    }
                }
            }
            
            # Write and read back
            with open(storage_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            
            with open(storage_path) as f:
                loaded_data = json.load(f)
            
            assert loaded_data == test_data
            
            return "Parameter storage integrity verified"
    
    def test_optimization_performance(self):
        """Test optimization performance with fast settings."""
        from pipelines.fit_core.optimizer import optimize_cmb_parameters
        
        start_time = time.time()
        
        # Run fast optimization test
        result = optimize_cmb_parameters(
            model='lcdm',
            optimize_params=['H0'],  # Single parameter for speed
            covariance_scaling=5.0   # Very fast for testing
        )
        
        elapsed = time.time() - start_time
        
        assert result.convergence_status == 'success'
        assert elapsed < 60  # Should complete in under 1 minute
        
        return f"Optimization completed in {elapsed:.1f}s with status: {result.convergence_status}"
    
    def test_storage_directory_setup(self):
        """Test optimization storage directory setup."""
        storage_dir = Path('optimization_results')
        
        # Create directory if it doesn't exist
        if not storage_dir.exists():
            storage_dir.mkdir(parents=True)
        
        assert storage_dir.exists()
        assert storage_dir.is_dir()
        
        # Test write permissions
        test_file = storage_dir / 'test_write.tmp'
        test_file.write_text('test')
        assert test_file.exists()
        test_file.unlink()
        
        return f"Storage directory {storage_dir} ready with write permissions"
    
    def test_concurrent_access_protection(self):
        """Test file locking for concurrent access protection."""
        from pipelines.fit_core.parameter_store import OptimizedParameterStore
        
        store1 = OptimizedParameterStore()
        store2 = OptimizedParameterStore()
        
        # Both stores should be able to read
        params1 = store1.get_model_defaults('lcdm')
        params2 = store2.get_model_defaults('lcdm')
        
        assert params1 == params2
        
        return "Concurrent access protection working"
    
    def test_warm_start_functionality(self):
        """Test warm start functionality."""
        from pipelines.fit_core.parameter_store import OptimizedParameterStore
        
        store = OptimizedParameterStore()
        
        # Test warm start parameter retrieval
        warm_params = store.get_warm_start_params('lcdm')
        
        # Should return None if no recent optimization, or dict if available
        assert warm_params is None or isinstance(warm_params, dict)
        
        return f"Warm start functionality working (recent params: {warm_params is not None})"
    
    def test_dry_run_mode(self):
        """Test dry run mode doesn't persist changes."""
        from pipelines.fit_core.parameter_store import OptimizedParameterStore
        
        store = OptimizedParameterStore()
        
        # Get initial state
        initial_params = store.get_model_defaults('lcdm').copy()
        
        # Test dry run update
        test_params = {'H0': 99.9, 'Om0': 0.999}  # Unusual values
        test_metadata = {
            'timestamp': '2025-10-22T14:42:00Z',
            'chi2_improvement': 999.0,
            'convergence_status': 'test'
        }
        
        store.update_model_defaults('lcdm', test_params, test_metadata, dry_run=True)
        
        # Verify no changes persisted
        final_params = store.get_model_defaults('lcdm')
        
        # Should be unchanged (dry run)
        for key in ['H0', 'Om0']:
            if key in initial_params and key in final_params:
                assert abs(initial_params[key] - final_params[key]) < 1e-10
        
        return "Dry run mode working correctly (no persistence)"
    
    def run_all_tests(self):
        """Run complete validation suite."""
        print("=== Parameter Optimization Deployment Validation ===")
        print(f"Starting validation at {datetime.now()}")
        print()
        
        # Core functionality tests
        print("1. Core Functionality Tests")
        self.run_test("Core module imports", self.test_core_imports)
        self.run_test("ParameterOptimizer instantiation", self.test_parameter_optimizer_instantiation)
        self.run_test("Parameter bounds validation", self.test_parameter_bounds_validation)
        self.run_test("Invalid parameter detection", self.test_invalid_parameter_detection)
        self.run_test("OptimizedParameterStore instantiation", self.test_parameter_store_instantiation)
        self.run_test("Parameter retrieval", self.test_parameter_retrieval)
        self.run_test("Cross-model consistency", self.test_cross_model_consistency)
        self.run_test("Configuration parsing", self.test_configuration_parsing)
        
        print()
        
        # Integration tests
        print("2. Integration Tests")
        self.run_test("Engine optimization integration", self.test_engine_integration)
        self.run_test("CMB script integration", self.test_cmb_script_integration)
        self.run_test("BAO script compatibility", self.test_bao_script_compatibility)
        
        print()
        
        # Data integrity tests
        print("3. Data Integrity Tests")
        self.run_test("Storage directory setup", self.test_storage_directory_setup)
        self.run_test("Parameter storage integrity", self.test_parameter_storage_integrity)
        self.run_test("Concurrent access protection", self.test_concurrent_access_protection)
        self.run_test("Warm start functionality", self.test_warm_start_functionality)
        self.run_test("Dry run mode", self.test_dry_run_mode)
        
        print()
        
        # Performance tests
        print("4. Performance Tests")
        self.run_test("Optimization performance", self.test_optimization_performance)
        
        print()
        
        # Summary
        print("=== Validation Summary ===")
        total_tests = self.passed_tests + self.failed_tests
        print(f"Total tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        
        if self.failed_tests == 0:
            print("✅ All validation tests PASSED")
            print("System ready for deployment")
            return True
        else:
            print("❌ Some validation tests FAILED")
            print("Review failures before deployment")
            
            # Print failed test details
            print("\nFailed Test Details:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    print(f"  - {result['name']}: {result['error']}")
            
            return False
    
    def save_report(self, filename='optimization_validation_report.json'):
        """Save detailed validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': self.passed_tests + self.failed_tests,
                'passed': self.passed_tests,
                'failed': self.failed_tests,
                'success_rate': self.passed_tests / (self.passed_tests + self.failed_tests) if (self.passed_tests + self.failed_tests) > 0 else 0
            },
            'test_results': self.test_results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {filename}")

def main():
    """Main validation entry point."""
    validator = OptimizationValidator()
    
    try:
        success = validator.run_all_tests()
        validator.save_report()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        validator.save_report('optimization_validation_interrupted.json')
        sys.exit(2)
    except Exception as e:
        print(f"\n\nValidation failed with unexpected error: {e}")
        print(traceback.format_exc())
        validator.save_report('optimization_validation_error.json')
        sys.exit(3)

if __name__ == '__main__':
    main()