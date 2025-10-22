#!/usr/bin/env python3
"""
Integration tests for wrapper scripts to ensure they work seamlessly with the unified architecture.

This module tests all wrapper scripts (fit_cmb.py, fit_bao.py, etc.) to verify they
properly interface with the unified engine and produce consistent results.
"""

import subprocess
import json
import tempfile
import os
import sys
from pathlib import Path

# Add pipelines to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fit_core.engine import run_fit


class TestWrapperScriptsIntegration:
    """Test wrapper scripts integration with unified architecture."""
    
    def test_fit_cmb_wrapper(self):
        """Test fit_cmb.py wrapper script integration."""
        
        print("Testing fit_cmb.py wrapper...")
        
        # Test LCDM CMB fitting
        result = subprocess.run([
            sys.executable, "pipelines/fit_cmb.py",
            "--model", "lcdm",
            "--quiet"
        ], capture_output=True, text=True, cwd=".")
        
        assert result.returncode == 0, f"fit_cmb.py failed: {result.stderr}"
        
        # Verify output contains expected elements
        output = result.stdout
        assert "chi2" in output.lower()
        assert "aic" in output.lower()
        
        print("✓ fit_cmb.py wrapper working correctly")
        
        # Test PBUF CMB fitting
        result = subprocess.run([
            sys.executable, "pipelines/fit_cmb.py", 
            "--model", "pbuf",
            "--quiet"
        ], capture_output=True, text=True, cwd=".")
        
        assert result.returncode == 0, f"fit_cmb.py PBUF failed: {result.stderr}"
        
        print("✓ fit_cmb.py PBUF model working correctly")
    
    def test_fit_bao_wrapper(self):
        """Test fit_bao.py wrapper script integration."""
        
        print("Testing fit_bao.py wrapper...")
        
        # Test LCDM BAO fitting
        result = subprocess.run([
            sys.executable, "pipelines/fit_bao.py",
            "--model", "lcdm", 
            "--quiet"
        ], capture_output=True, text=True, cwd=".")
        
        assert result.returncode == 0, f"fit_bao.py failed: {result.stderr}"
        
        output = result.stdout
        assert "chi2" in output.lower()
        
        print("✓ fit_bao.py wrapper working correctly")
    
    def test_fit_sn_wrapper(self):
        """Test fit_sn.py wrapper script integration."""
        
        print("Testing fit_sn.py wrapper...")
        
        # Test LCDM SN fitting
        result = subprocess.run([
            sys.executable, "pipelines/fit_sn.py",
            "--model", "lcdm",
            "--quiet"
        ], capture_output=True, text=True, cwd=".")
        
        assert result.returncode == 0, f"fit_sn.py failed: {result.stderr}"
        
        output = result.stdout
        assert "chi2" in output.lower()
        
        print("✓ fit_sn.py wrapper working correctly")
    
    def test_fit_joint_wrapper(self):
        """Test fit_joint.py wrapper script integration."""
        
        print("Testing fit_joint.py wrapper...")
        
        # Test joint fitting with multiple datasets
        result = subprocess.run([
            sys.executable, "pipelines/fit_joint.py",
            "--model", "lcdm",
            "--datasets", "cmb", "bao",
            "--quiet"
        ], capture_output=True, text=True, cwd=".")
        
        assert result.returncode == 0, f"fit_joint.py failed: {result.stderr}"
        
        output = result.stdout
        assert "chi2" in output.lower()
        assert "cmb" in output.lower()
        assert "bao" in output.lower()
        
        print("✓ fit_joint.py wrapper working correctly")
    
    def test_wrapper_vs_engine_consistency(self):
        """Test that wrapper scripts produce identical results to direct engine calls."""
        
        print("Testing wrapper vs engine consistency...")
        
        # Compare CMB fitting: wrapper vs direct engine
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            # Run via wrapper with JSON output
            result = subprocess.run([
                sys.executable, "pipelines/fit_cmb.py",
                "--model", "lcdm",
                "--output-format", "json",
                "--output-file", output_file,
                "--quiet"
            ], capture_output=True, text=True, cwd=".")
            
            assert result.returncode == 0, f"Wrapper failed: {result.stderr}"
            
            # Load wrapper results
            with open(output_file, 'r') as f:
                wrapper_results = json.load(f)
            
            # Run via direct engine call
            engine_results = run_fit(model="lcdm", datasets_list=["cmb"], mode="individual")
            
            # Compare key metrics (allowing for small numerical differences)
            wrapper_chi2 = wrapper_results["metrics"]["total_chi2"]
            engine_chi2 = engine_results["metrics"]["total_chi2"]
            
            assert abs(wrapper_chi2 - engine_chi2) < 1e-6, \
                f"Chi2 mismatch: wrapper={wrapper_chi2}, engine={engine_chi2}"
            
            wrapper_aic = wrapper_results["metrics"]["aic"]
            engine_aic = engine_results["metrics"]["aic"]
            
            assert abs(wrapper_aic - engine_aic) < 1e-6, \
                f"AIC mismatch: wrapper={wrapper_aic}, engine={engine_aic}"
            
            print("✓ Wrapper and engine results are consistent")
            
        finally:
            # Clean up
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_command_line_options(self):
        """Test various command-line options work correctly."""
        
        print("Testing command-line options...")
        
        # Test parameter overrides
        result = subprocess.run([
            sys.executable, "pipelines/fit_cmb.py",
            "--model", "lcdm",
            "--override", "H0=70.0",
            "--override", "Om0=0.3",
            "--quiet"
        ], capture_output=True, text=True, cwd=".")
        
        assert result.returncode == 0, f"Parameter override failed: {result.stderr}"
        
        # Verify overrides were applied (check output contains the values)
        output = result.stdout
        assert "70.0" in output or "70" in output  # H0 override
        
        print("✓ Parameter overrides working correctly")
        
        # Test integrity verification flag
        result = subprocess.run([
            sys.executable, "pipelines/fit_cmb.py",
            "--model", "lcdm",
            "--verify-integrity",
            "--quiet"
        ], capture_output=True, text=True, cwd=".")
        
        assert result.returncode == 0, f"Integrity verification failed: {result.stderr}"
        
        output = result.stdout
        assert "integrity" in output.lower() or "validation" in output.lower()
        
        print("✓ Integrity verification flag working correctly")
    
    def test_error_handling_in_wrappers(self):
        """Test error handling in wrapper scripts."""
        
        print("Testing error handling in wrappers...")
        
        # Test invalid model
        result = subprocess.run([
            sys.executable, "pipelines/fit_cmb.py",
            "--model", "invalid_model",
            "--quiet"
        ], capture_output=True, text=True, cwd=".")
        
        assert result.returncode != 0, "Should fail with invalid model"
        
        # Test invalid parameter override
        result = subprocess.run([
            sys.executable, "pipelines/fit_cmb.py",
            "--model", "lcdm",
            "--override", "invalid_format",
            "--quiet"
        ], capture_output=True, text=True, cwd=".")
        
        assert result.returncode != 0, "Should fail with invalid override format"
        
        print("✓ Error handling in wrappers working correctly")


def run_wrapper_integration_tests():
    """Run all wrapper script integration tests."""
    
    print("="*60)
    print("WRAPPER SCRIPTS INTEGRATION TESTING")
    print("="*60)
    
    test_suite = TestWrapperScriptsIntegration()
    
    try:
        print("\n1. Testing individual wrapper scripts...")
        test_suite.test_fit_cmb_wrapper()
        test_suite.test_fit_bao_wrapper()
        test_suite.test_fit_sn_wrapper()
        test_suite.test_fit_joint_wrapper()
        
        print("\n2. Testing wrapper vs engine consistency...")
        test_suite.test_wrapper_vs_engine_consistency()
        
        print("\n3. Testing command-line options...")
        test_suite.test_command_line_options()
        
        print("\n4. Testing error handling...")
        test_suite.test_error_handling_in_wrappers()
        
        print("\n" + "="*60)
        print("ALL WRAPPER INTEGRATION TESTS PASSED")
        print("="*60)
        
        return {"status": "PASSED", "wrappers_tested": 4}
        
    except Exception as e:
        print(f"\n❌ WRAPPER INTEGRATION TEST FAILED: {e}")
        print("="*60)
        raise


if __name__ == "__main__":
    summary = run_wrapper_integration_tests()
    print(f"\nWrapper integration test summary: {summary}")