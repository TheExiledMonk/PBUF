#!/usr/bin/env python3
"""
Test suite for optimization configuration parsing and validation.

Tests the configuration system's ability to handle optimization settings
from both configuration files and command-line arguments.
"""

import unittest
import tempfile
import json
import os
import argparse
from unittest.mock import patch
from pathlib import Path

try:
    from .config import (
        ConfigurationManager, 
        add_optimization_arguments,
        parse_optimization_parameters,
        merge_optimization_config
    )
except ImportError:
    from config import (
        ConfigurationManager, 
        add_optimization_arguments,
        parse_optimization_parameters,
        merge_optimization_config
    )


class TestOptimizationConfigParsing(unittest.TestCase):
    """Test optimization configuration parsing from files."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        for file in Path(self.temp_dir).glob("*"):
            file.unlink()
        os.rmdir(self.temp_dir)
    
    def test_json_optimization_config_parsing(self):
        """Test parsing optimization config from JSON file."""
        config_data = {
            "optimization": {
                "optimize_parameters": ["k_sat", "alpha"],
                "frozen_parameters": ["Tcmb", "Neff"],
                "use_precomputed": True,
                "save_results": True,
                "convergence_tolerance": 1e-6,
                "covariance_scaling": 1.2,
                "warm_start": False,
                "dry_run": False
            }
        }
        
        config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        manager = ConfigurationManager(config_file)
        optimization_config = manager.get_optimization_config()
        
        self.assertEqual(optimization_config['optimize_parameters'], ["k_sat", "alpha"])
        self.assertEqual(optimization_config['frozen_parameters'], ["Tcmb", "Neff"])
        self.assertTrue(optimization_config['use_precomputed'])
        self.assertTrue(optimization_config['save_results'])
        self.assertEqual(optimization_config['convergence_tolerance'], 1e-6)
        self.assertEqual(optimization_config['covariance_scaling'], 1.2)
        self.assertFalse(optimization_config['warm_start'])
        self.assertFalse(optimization_config['dry_run'])
    
    def test_yaml_optimization_config_parsing(self):
        """Test parsing optimization config from YAML file."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not available")
        
        config_data = """
optimization:
  optimize_parameters:
    - H0
    - Om0
  frozen_parameters:
    - ns
  use_precomputed: false
  save_results: true
  convergence_tolerance: 1.0e-8
  covariance_scaling: 0.8
  warm_start: true
  dry_run: false
"""
        
        config_file = os.path.join(self.temp_dir, "test_config.yaml")
        with open(config_file, 'w') as f:
            f.write(config_data)
        
        manager = ConfigurationManager(config_file)
        optimization_config = manager.get_optimization_config()
        
        self.assertEqual(optimization_config['optimize_parameters'], ["H0", "Om0"])
        self.assertEqual(optimization_config['frozen_parameters'], ["ns"])
        self.assertFalse(optimization_config['use_precomputed'])
        self.assertTrue(optimization_config['save_results'])
        self.assertEqual(optimization_config['convergence_tolerance'], 1e-8)
        self.assertEqual(optimization_config['covariance_scaling'], 0.8)
        self.assertTrue(optimization_config['warm_start'])
        self.assertFalse(optimization_config['dry_run'])
    
    def test_ini_optimization_config_parsing(self):
        """Test parsing optimization config from INI file."""
        config_data = """
[optimization]
optimize_parameters = k_sat,alpha,H0
frozen_parameters = Tcmb,Neff
use_precomputed = true
save_results = false
convergence_tolerance = 1e-7
covariance_scaling = 1.5
warm_start = false
dry_run = true
"""
        
        config_file = os.path.join(self.temp_dir, "test_config.ini")
        with open(config_file, 'w') as f:
            f.write(config_data)
        
        manager = ConfigurationManager(config_file)
        optimization_config = manager.get_optimization_config()
        
        self.assertEqual(optimization_config['optimize_parameters'], ["k_sat", "alpha", "H0"])
        self.assertEqual(optimization_config['frozen_parameters'], ["Tcmb", "Neff"])
        self.assertTrue(optimization_config['use_precomputed'])
        self.assertFalse(optimization_config['save_results'])
        self.assertEqual(optimization_config['convergence_tolerance'], 1e-7)
        self.assertEqual(optimization_config['covariance_scaling'], 1.5)
        self.assertFalse(optimization_config['warm_start'])
        self.assertTrue(optimization_config['dry_run'])
    
    def test_default_optimization_config(self):
        """Test default optimization configuration when no config provided."""
        manager = ConfigurationManager()
        optimization_config = manager.get_optimization_config()
        
        self.assertEqual(optimization_config['optimize_parameters'], [])
        self.assertEqual(optimization_config['frozen_parameters'], [])
        self.assertTrue(optimization_config['use_precomputed'])
        self.assertTrue(optimization_config['save_results'])
        self.assertEqual(optimization_config['convergence_tolerance'], 1e-6)
        self.assertEqual(optimization_config['covariance_scaling'], 1.0)
        self.assertFalse(optimization_config['warm_start'])
        self.assertFalse(optimization_config['dry_run'])


class TestOptimizationConfigValidation(unittest.TestCase):
    """Test optimization configuration validation."""
    
    def test_valid_optimization_config(self):
        """Test validation of valid optimization configuration."""
        config_data = {
            "optimization": {
                "optimize_parameters": ["k_sat", "alpha"],
                "frozen_parameters": ["Tcmb"],
                "covariance_scaling": 1.2,
                "convergence_tolerance": 1e-6
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            # Should not raise any exceptions
            optimization_config = manager.get_optimization_config()
            self.assertIsInstance(optimization_config, dict)
        finally:
            os.unlink(config_file)
    
    def test_invalid_optimize_parameters_type(self):
        """Test validation fails for non-list optimize_parameters."""
        config_data = {
            "optimization": {
                "optimize_parameters": "k_sat,alpha",  # Should be list, not string
                "frozen_parameters": []
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            with self.assertRaises(ValueError) as cm:
                manager.get_optimization_config()
            self.assertIn("optimize_parameters must be a list", str(cm.exception))
        finally:
            os.unlink(config_file)
    
    def test_invalid_frozen_parameters_type(self):
        """Test validation fails for non-list frozen_parameters."""
        config_data = {
            "optimization": {
                "optimize_parameters": [],
                "frozen_parameters": "Tcmb,Neff"  # Should be list, not string
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            with self.assertRaises(ValueError) as cm:
                manager.get_optimization_config()
            self.assertIn("frozen_parameters must be a list", str(cm.exception))
        finally:
            os.unlink(config_file)
    
    def test_invalid_covariance_scaling(self):
        """Test validation fails for invalid covariance_scaling."""
        config_data = {
            "optimization": {
                "optimize_parameters": [],
                "frozen_parameters": [],
                "covariance_scaling": -1.0  # Should be positive
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            with self.assertRaises(ValueError) as cm:
                manager.get_optimization_config()
            self.assertIn("covariance_scaling must be a positive number", str(cm.exception))
        finally:
            os.unlink(config_file)
    
    def test_invalid_convergence_tolerance(self):
        """Test validation fails for invalid convergence_tolerance."""
        config_data = {
            "optimization": {
                "optimize_parameters": [],
                "frozen_parameters": [],
                "convergence_tolerance": 0.0  # Should be positive
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            with self.assertRaises(ValueError) as cm:
                manager.get_optimization_config()
            self.assertIn("convergence_tolerance must be a positive number", str(cm.exception))
        finally:
            os.unlink(config_file)
    
    def test_conflicting_optimize_and_frozen_parameters(self):
        """Test validation fails when parameters are both optimized and frozen."""
        config_data = {
            "optimization": {
                "optimize_parameters": ["k_sat", "alpha"],
                "frozen_parameters": ["alpha", "H0"]  # alpha appears in both lists
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            with self.assertRaises(ValueError) as cm:
                manager.get_optimization_config()
            self.assertIn("Parameters cannot be both optimized and frozen", str(cm.exception))
            self.assertIn("alpha", str(cm.exception))
        finally:
            os.unlink(config_file)
    
    def test_invalid_boolean_flags(self):
        """Test validation fails for invalid boolean flags."""
        for flag in ['use_precomputed', 'save_results', 'warm_start', 'dry_run']:
            config_data = {
                "optimization": {
                    "optimize_parameters": [],
                    "frozen_parameters": [],
                    flag: "yes"  # Should be boolean, not string
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_file = f.name
            
            try:
                manager = ConfigurationManager(config_file)
                with self.assertRaises(ValueError) as cm:
                    manager.get_optimization_config()
                self.assertIn(f"{flag} must be a boolean", str(cm.exception))
            finally:
                os.unlink(config_file)
    
    def test_unknown_parameter_warning(self):
        """Test warning for unknown parameters in optimization lists."""
        config_data = {
            "optimization": {
                "optimize_parameters": ["unknown_param", "k_sat"],
                "frozen_parameters": ["another_unknown"]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            # Should issue warnings but not fail
            with patch('builtins.print') as mock_print:
                optimization_config = manager.get_optimization_config()
                # Check that warnings were printed
                mock_print.assert_called()
                warning_calls = [call for call in mock_print.call_args_list 
                               if 'Warning' in str(call)]
                self.assertTrue(len(warning_calls) >= 2)  # At least 2 warnings
        finally:
            os.unlink(config_file)


class TestCommandLineArgumentParsing(unittest.TestCase):
    """Test command-line argument parsing for optimization."""
    
    def test_add_optimization_arguments(self):
        """Test adding optimization arguments to argument parser."""
        parser = argparse.ArgumentParser()
        add_optimization_arguments(parser)
        
        # Test that arguments were added
        args = parser.parse_args([
            '--optimize', 'k_sat,alpha',
            '--cov-scale', '1.2',
            '--dry-run',
            '--warm-start'
        ])
        
        self.assertEqual(args.optimize, 'k_sat,alpha')
        self.assertEqual(args.cov_scale, 1.2)
        self.assertTrue(args.dry_run)
        self.assertTrue(args.warm_start)
    
    def test_parse_optimization_parameters(self):
        """Test parsing optimization parameter strings."""
        # Test normal case
        params = parse_optimization_parameters("k_sat,alpha,H0")
        self.assertEqual(params, ["k_sat", "alpha", "H0"])
        
        # Test with spaces
        params = parse_optimization_parameters("k_sat, alpha , H0")
        self.assertEqual(params, ["k_sat", "alpha", "H0"])
        
        # Test empty string
        params = parse_optimization_parameters("")
        self.assertEqual(params, [])
        
        # Test None
        params = parse_optimization_parameters(None)
        self.assertEqual(params, [])
        
        # Test single parameter
        params = parse_optimization_parameters("k_sat")
        self.assertEqual(params, ["k_sat"])
    
    def test_merge_optimization_config(self):
        """Test merging optimization config from file and CLI."""
        # Config file settings
        config_optimization = {
            'optimize_parameters': ['k_sat'],
            'frozen_parameters': ['Tcmb'],
            'covariance_scaling': 1.0,
            'dry_run': False,
            'warm_start': False
        }
        
        # Mock CLI args
        class MockArgs:
            def __init__(self):
                self.optimize = 'alpha,H0'
                self.cov_scale = 1.5
                self.dry_run = True
                self.warm_start = False
        
        cli_args = MockArgs()
        
        merged = merge_optimization_config(config_optimization, cli_args)
        
        # CLI should override config file
        self.assertEqual(merged['optimize_parameters'], ['alpha', 'H0'])
        self.assertEqual(merged['covariance_scaling'], 1.5)
        self.assertTrue(merged['dry_run'])
        
        # Config file values should be preserved where CLI doesn't override
        self.assertEqual(merged['frozen_parameters'], ['Tcmb'])
        self.assertFalse(merged['warm_start'])


class TestCommandLinePrecedence(unittest.TestCase):
    """Test that command-line arguments take precedence over config files."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        for file in Path(self.temp_dir).glob("*"):
            file.unlink()
        os.rmdir(self.temp_dir)
    
    def test_cli_overrides_config_file(self):
        """Test that CLI arguments override config file settings."""
        # Create config file
        config_data = {
            "optimization": {
                "optimize_parameters": ["k_sat"],
                "covariance_scaling": 1.0,
                "dry_run": False
            }
        }
        
        config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load config
        manager = ConfigurationManager(config_file)
        config_optimization = manager.get_optimization_config()
        
        # Mock CLI args that override config
        class MockArgs:
            def __init__(self):
                self.optimize = 'alpha,H0'
                self.cov_scale = 1.5
                self.dry_run = True
                self.warm_start = False
        
        cli_args = MockArgs()
        
        # Merge configurations
        merged = merge_optimization_config(config_optimization, cli_args)
        
        # Verify CLI takes precedence
        self.assertEqual(merged['optimize_parameters'], ['alpha', 'H0'])  # CLI override
        self.assertEqual(merged['covariance_scaling'], 1.5)  # CLI override
        self.assertTrue(merged['dry_run'])  # CLI override
    
    def test_config_file_used_when_cli_not_provided(self):
        """Test that config file values are used when CLI args not provided."""
        # Create config file
        config_data = {
            "optimization": {
                "optimize_parameters": ["k_sat", "alpha"],
                "covariance_scaling": 1.2,
                "dry_run": True
            }
        }
        
        config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load config
        manager = ConfigurationManager(config_file)
        config_optimization = manager.get_optimization_config()
        
        # Mock CLI args with no optimization settings
        class MockArgs:
            def __init__(self):
                self.optimize = None
                self.cov_scale = 1.0  # Default value
                self.dry_run = False  # Default value
                self.warm_start = False  # Default value
        
        cli_args = MockArgs()
        
        # Merge configurations
        merged = merge_optimization_config(config_optimization, cli_args)
        
        # Verify config file values are preserved
        self.assertEqual(merged['optimize_parameters'], ['k_sat', 'alpha'])
        self.assertEqual(merged['covariance_scaling'], 1.2)
        self.assertTrue(merged['dry_run'])


class TestErrorHandling(unittest.TestCase):
    """Test error handling for invalid optimization settings."""
    
    def test_invalid_config_file_format(self):
        """Test handling of invalid configuration file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            config_file = f.name
        
        try:
            with self.assertRaises(ValueError) as cm:
                ConfigurationManager(config_file)
            self.assertIn("Error loading configuration file", str(cm.exception))
        finally:
            os.unlink(config_file)
    
    def test_nonexistent_config_file(self):
        """Test handling of nonexistent configuration file."""
        # Constructor doesn't raise error for nonexistent files
        manager = ConfigurationManager("/nonexistent/config.json")
        # But load_config should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            manager.load_config("/nonexistent/config.json")
    
    def test_empty_optimization_parameters(self):
        """Test handling of empty optimization parameter lists."""
        config_data = {
            "optimization": {
                "optimize_parameters": [],
                "frozen_parameters": []
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            manager = ConfigurationManager(config_file)
            optimization_config = manager.get_optimization_config()
            # Should not raise any exceptions
            self.assertEqual(optimization_config['optimize_parameters'], [])
            self.assertEqual(optimization_config['frozen_parameters'], [])
        finally:
            os.unlink(config_file)


if __name__ == '__main__':
    unittest.main()