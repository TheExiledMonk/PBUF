"""
Tests for the output manager and format conversion functionality.

This module tests standardized output generation, format conversion,
and compatibility with existing fit pipelines.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from ..core.schema import StandardDataset
from ..core.interfaces import ProvenanceRecord
from ..output.output_manager import OutputManager
from ..output.format_converter import FormatConverter

# Optional pandas import for enhanced tests
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class TestOutputManager(unittest.TestCase):
    """Test cases for OutputManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_manager = OutputManager(self.temp_dir)
        
        # Create test dataset
        self.test_dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([10.0, 15.0, 20.0]),
            uncertainty=np.array([1.0, 1.5, 2.0]),
            covariance=np.array([[1.0, 0.1, 0.0], [0.1, 2.25, 0.2], [0.0, 0.2, 4.0]]),
            metadata={
                'source': 'test_source',
                'dataset_type': 'test',
                'processing_timestamp': '2024-01-01T00:00:00'
            }
        )
        
        # Create test provenance
        self.test_provenance = ProvenanceRecord(
            source_dataset_name='test_raw',
            source_hash='abc123',
            derivation_module='test_module',
            processing_timestamp='2024-01-01T00:00:00',
            environment_hash='env123',
            transformation_steps=['step1', 'step2'],
            validation_results={'passed': True},
            derived_dataset_hash='def456'
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_dataset_csv(self):
        """Test saving dataset in CSV format."""
        output_paths = self.output_manager.save_dataset(
            self.test_dataset,
            'test_dataset',
            formats=['csv'],
            timestamp='20240101_000000'
        )
        
        # Check that files were created
        self.assertIn('csv', output_paths)
        self.assertIn('metadata', output_paths)
        
        csv_path = output_paths['csv']
        self.assertTrue(csv_path.exists())
        
        # Verify CSV content by reading the file manually
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Check header
        self.assertEqual(lines[0].strip(), 'z,observable,uncertainty')
        
        # Check data rows
        for i, line in enumerate(lines[1:]):
            parts = line.strip().split(',')
            self.assertAlmostEqual(float(parts[0]), self.test_dataset.z[i])
            self.assertAlmostEqual(float(parts[1]), self.test_dataset.observable[i])
            self.assertAlmostEqual(float(parts[2]), self.test_dataset.uncertainty[i])
        
        # Check covariance file
        cov_path = csv_path.parent / f"{csv_path.stem}_covariance.csv"
        self.assertTrue(cov_path.exists())
    
    def test_save_dataset_numpy(self):
        """Test saving dataset in NumPy format."""
        output_paths = self.output_manager.save_dataset(
            self.test_dataset,
            'test_dataset',
            formats=['numpy'],
            timestamp='20240101_000000'
        )
        
        # Check that files were created
        self.assertIn('numpy', output_paths)
        
        numpy_path = output_paths['numpy']
        self.assertTrue(numpy_path.exists())
        
        # Verify NumPy content
        data = np.load(numpy_path)
        np.testing.assert_array_equal(data['z'], self.test_dataset.z)
        np.testing.assert_array_equal(data['observable'], self.test_dataset.observable)
        np.testing.assert_array_equal(data['uncertainty'], self.test_dataset.uncertainty)
        np.testing.assert_array_equal(data['covariance'], self.test_dataset.covariance)
        
        # Check metadata
        metadata = json.loads(str(data['metadata_json']))
        self.assertEqual(metadata['source'], 'test_source')
    
    def test_save_dataset_json(self):
        """Test saving dataset in JSON format."""
        output_paths = self.output_manager.save_dataset(
            self.test_dataset,
            'test_dataset',
            formats=['json'],
            timestamp='20240101_000000'
        )
        
        # Check that files were created
        self.assertIn('json', output_paths)
        
        json_path = output_paths['json']
        self.assertTrue(json_path.exists())
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        np.testing.assert_array_equal(data['z'], self.test_dataset.z.tolist())
        np.testing.assert_array_equal(data['observable'], self.test_dataset.observable.tolist())
        np.testing.assert_array_equal(data['uncertainty'], self.test_dataset.uncertainty.tolist())
        np.testing.assert_array_equal(data['covariance'], self.test_dataset.covariance.tolist())
        self.assertEqual(data['metadata']['source'], 'test_source')
    
    def test_save_with_provenance(self):
        """Test saving dataset with provenance record."""
        output_paths = self.output_manager.save_dataset(
            self.test_dataset,
            'test_dataset',
            formats=['csv'],
            provenance=self.test_provenance,
            timestamp='20240101_000000'
        )
        
        # Check provenance file was created
        self.assertIn('provenance', output_paths)
        
        provenance_path = output_paths['provenance']
        self.assertTrue(provenance_path.exists())
        
        # Verify provenance content
        with open(provenance_path, 'r') as f:
            provenance_data = json.load(f)
        
        self.assertEqual(provenance_data['source_dataset_name'], 'test_raw')
        self.assertEqual(provenance_data['source_hash'], 'abc123')
        self.assertEqual(provenance_data['derivation_module'], 'test_module')
    
    def test_load_dataset_csv(self):
        """Test loading dataset from CSV format."""
        # First save a dataset
        output_paths = self.output_manager.save_dataset(
            self.test_dataset,
            'test_dataset',
            formats=['csv'],
            timestamp='20240101_000000'
        )
        
        # Load it back
        loaded_dataset = self.output_manager.load_dataset(output_paths['csv'], 'csv')
        
        # Verify loaded data matches original
        np.testing.assert_array_almost_equal(loaded_dataset.z, self.test_dataset.z)
        np.testing.assert_array_almost_equal(loaded_dataset.observable, self.test_dataset.observable)
        np.testing.assert_array_almost_equal(loaded_dataset.uncertainty, self.test_dataset.uncertainty)
        if loaded_dataset.covariance is not None and self.test_dataset.covariance is not None:
            np.testing.assert_array_almost_equal(loaded_dataset.covariance, self.test_dataset.covariance)
        self.assertEqual(loaded_dataset.metadata['source'], 'test_source')
    
    def test_load_dataset_numpy(self):
        """Test loading dataset from NumPy format."""
        # First save a dataset
        output_paths = self.output_manager.save_dataset(
            self.test_dataset,
            'test_dataset',
            formats=['numpy'],
            timestamp='20240101_000000'
        )
        
        # Load it back
        loaded_dataset = self.output_manager.load_dataset(output_paths['numpy'], 'numpy')
        
        # Verify loaded data matches original
        np.testing.assert_array_equal(loaded_dataset.z, self.test_dataset.z)
        np.testing.assert_array_equal(loaded_dataset.observable, self.test_dataset.observable)
        np.testing.assert_array_equal(loaded_dataset.uncertainty, self.test_dataset.uncertainty)
        np.testing.assert_array_equal(loaded_dataset.covariance, self.test_dataset.covariance)
        self.assertEqual(loaded_dataset.metadata['source'], 'test_source')
    
    def test_auto_format_detection(self):
        """Test automatic format detection when loading."""
        # Save in different formats
        csv_paths = self.output_manager.save_dataset(
            self.test_dataset, 'test_csv', formats=['csv'], timestamp='20240101_000000'
        )
        numpy_paths = self.output_manager.save_dataset(
            self.test_dataset, 'test_numpy', formats=['numpy'], timestamp='20240101_000000'
        )
        
        # Load with auto-detection
        csv_loaded = self.output_manager.load_dataset(csv_paths['csv'], 'auto')
        numpy_loaded = self.output_manager.load_dataset(numpy_paths['numpy'], 'auto')
        
        # Both should match original
        np.testing.assert_array_almost_equal(csv_loaded.z, self.test_dataset.z)
        np.testing.assert_array_almost_equal(numpy_loaded.z, self.test_dataset.z)
    
    def test_invalid_format(self):
        """Test handling of invalid formats."""
        with self.assertRaises(ValueError):
            self.output_manager.save_dataset(
                self.test_dataset,
                'test_dataset',
                formats=['invalid_format']
            )
    
    def test_cleanup_old_files(self):
        """Test cleanup of old dataset files."""
        # Create multiple versions
        for i in range(7):
            timestamp = f"2024010{i}_000000"
            self.output_manager.save_dataset(
                self.test_dataset,
                'test_dataset',
                formats=['csv'],
                timestamp=timestamp
            )
        
        # Cleanup keeping only 3 latest
        deleted_files = self.output_manager.cleanup_old_files('test_dataset', keep_latest=3)
        
        # Should have deleted some files
        self.assertGreater(len(deleted_files), 0)
        
        # Check remaining files
        remaining_files = list(Path(self.temp_dir).glob('test_dataset_derived_*'))
        # Should have at most 3 timestamps worth of files (each timestamp creates multiple files)
        unique_timestamps = set()
        for file_path in remaining_files:
            parts = file_path.stem.split('_')
            if len(parts) >= 3:
                timestamp = '_'.join(parts[-2:])
                unique_timestamps.add(timestamp)
        
        self.assertLessEqual(len(unique_timestamps), 3)


class TestFormatConverter(unittest.TestCase):
    """Test cases for FormatConverter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test StandardDataset for different types
        self.cmb_dataset = StandardDataset(
            z=np.array([1089.80, 1089.80, 1089.80]),  # Repeated redshift for 3 observables
            observable=np.array([1.7502, 301.845, 1.04092]),
            uncertainty=np.array([0.0015, 2.08, 0.00068]),
            covariance=np.array([
                [2.30e-6, 2.99e-6, -8.93e-9],
                [2.99e-6, 4.33e-6, -1.28e-8],
                [-8.93e-9, -1.28e-8, 4.64e-11]
            ]),
            metadata={'source': 'Planck2018', 'dataset_type': 'cmb'}
        )
        
        self.bao_dataset = StandardDataset(
            z=np.array([0.106, 0.15, 0.38]),
            observable=np.array([4.47, 4.47, 10.23]),
            uncertainty=np.array([0.17, 0.17, 0.17]),
            covariance=np.diag([0.17**2, 0.17**2, 0.17**2]),
            metadata={'source': 'BAO_Compilation', 'dataset_type': 'bao'}
        )
        
        self.sn_dataset = StandardDataset(
            z=np.array([0.01, 0.1, 0.5]),
            observable=np.array([33.0, 38.5, 42.8]),
            uncertainty=np.array([0.15, 0.20, 0.25]),
            covariance=np.diag([0.15**2, 0.20**2, 0.25**2]),
            metadata={'source': 'Pantheon+', 'dataset_type': 'sn'}
        )
    
    def test_cmb_standard_to_dataset_dict(self):
        """Test conversion of CMB StandardDataset to DatasetDict."""
        dataset_dict = FormatConverter.standard_to_dataset_dict(self.cmb_dataset, 'cmb')
        
        # Check structure
        self.assertEqual(dataset_dict['dataset_type'], 'cmb')
        self.assertIn('observations', dataset_dict)
        self.assertIn('covariance', dataset_dict)
        self.assertIn('metadata', dataset_dict)
        
        # Check CMB-specific observations
        obs = dataset_dict['observations']
        self.assertAlmostEqual(obs['R'], 1.7502)
        self.assertAlmostEqual(obs['l_A'], 301.845)
        self.assertAlmostEqual(obs['theta_star'], 1.04092)
        
        # Check covariance
        np.testing.assert_array_equal(dataset_dict['covariance'], self.cmb_dataset.covariance)
    
    def test_bao_standard_to_dataset_dict(self):
        """Test conversion of BAO StandardDataset to DatasetDict."""
        dataset_dict = FormatConverter.standard_to_dataset_dict(self.bao_dataset, 'bao')
        
        # Check structure
        self.assertEqual(dataset_dict['dataset_type'], 'bao')
        
        # Check BAO-specific observations
        obs = dataset_dict['observations']
        np.testing.assert_array_equal(obs['redshift'], self.bao_dataset.z)
        np.testing.assert_array_equal(obs['DV_over_rs'], self.bao_dataset.observable)
    
    def test_sn_standard_to_dataset_dict(self):
        """Test conversion of SN StandardDataset to DatasetDict."""
        dataset_dict = FormatConverter.standard_to_dataset_dict(self.sn_dataset, 'sn')
        
        # Check structure
        self.assertEqual(dataset_dict['dataset_type'], 'sn')
        
        # Check SN-specific observations
        obs = dataset_dict['observations']
        np.testing.assert_array_equal(obs['redshift'], self.sn_dataset.z)
        np.testing.assert_array_equal(obs['distance_modulus'], self.sn_dataset.observable)
        np.testing.assert_array_equal(obs['sigma_mu'], self.sn_dataset.uncertainty)
    
    def test_dataset_dict_to_standard_cmb(self):
        """Test conversion of CMB DatasetDict to StandardDataset."""
        # Create CMB DatasetDict
        dataset_dict = {
            'dataset_type': 'cmb',
            'observations': {
                'R': 1.7502,
                'l_A': 301.845,
                'theta_star': 1.04092
            },
            'covariance': self.cmb_dataset.covariance,
            'metadata': {'source': 'Planck2018'}
        }
        
        standard_dataset = FormatConverter.dataset_dict_to_standard(dataset_dict, 'cmb')
        
        # Check conversion
        self.assertEqual(len(standard_dataset.observable), 3)
        self.assertAlmostEqual(standard_dataset.observable[0], 1.7502)
        self.assertAlmostEqual(standard_dataset.observable[1], 301.845)
        self.assertAlmostEqual(standard_dataset.observable[2], 1.04092)
        np.testing.assert_array_equal(standard_dataset.covariance, self.cmb_dataset.covariance)
    
    def test_dataset_dict_to_standard_bao(self):
        """Test conversion of BAO DatasetDict to StandardDataset."""
        # Create BAO DatasetDict
        dataset_dict = {
            'dataset_type': 'bao',
            'observations': {
                'redshift': self.bao_dataset.z,
                'DV_over_rs': self.bao_dataset.observable
            },
            'covariance': self.bao_dataset.covariance,
            'metadata': {'source': 'BAO_Compilation'}
        }
        
        standard_dataset = FormatConverter.dataset_dict_to_standard(dataset_dict, 'bao')
        
        # Check conversion
        np.testing.assert_array_equal(standard_dataset.z, self.bao_dataset.z)
        np.testing.assert_array_equal(standard_dataset.observable, self.bao_dataset.observable)
        np.testing.assert_array_equal(standard_dataset.covariance, self.bao_dataset.covariance)
    
    def test_round_trip_conversion(self):
        """Test round-trip conversion preserves data integrity."""
        # Test with BAO dataset
        dataset_dict = FormatConverter.standard_to_dataset_dict(self.bao_dataset, 'bao')
        reconverted = FormatConverter.dataset_dict_to_standard(dataset_dict, 'bao')
        
        # Check data is preserved
        np.testing.assert_array_equal(reconverted.z, self.bao_dataset.z)
        np.testing.assert_array_equal(reconverted.observable, self.bao_dataset.observable)
        np.testing.assert_array_equal(reconverted.covariance, self.bao_dataset.covariance)
    
    def test_validate_conversion_compatibility(self):
        """Test validation of conversion compatibility."""
        # Create original DatasetDict
        original_dict = {
            'dataset_type': 'bao',
            'observations': {
                'redshift': self.bao_dataset.z,
                'DV_over_rs': self.bao_dataset.observable
            },
            'covariance': self.bao_dataset.covariance,
            'metadata': {'source': 'BAO_Compilation'}
        }
        
        # Convert to StandardDataset
        converted = FormatConverter.dataset_dict_to_standard(original_dict, 'bao')
        
        # Validate compatibility
        is_compatible = FormatConverter.validate_conversion_compatibility(
            original_dict, converted, 'bao'
        )
        self.assertTrue(is_compatible)
    
    def test_bao_anisotropic_conversion(self):
        """Test conversion of anisotropic BAO data."""
        # Create anisotropic BAO dataset
        z_values = np.array([0.38, 0.51])
        dm_values = np.array([10.23, 13.36])
        dh_values = np.array([0.198, 0.179])
        
        # Combined observable array (DM first, then DH)
        combined_obs = np.concatenate([dm_values, dh_values])
        
        bao_ani_dataset = StandardDataset(
            z=z_values,
            observable=combined_obs,
            uncertainty=np.array([0.17, 0.21, 0.008, 0.009]),
            covariance=np.eye(4),
            metadata={'source': 'BOSS', 'dataset_type': 'bao_ani'}
        )
        
        # Convert to DatasetDict
        dataset_dict = FormatConverter.standard_to_dataset_dict(bao_ani_dataset, 'bao_ani')
        
        # Check anisotropic structure
        obs = dataset_dict['observations']
        np.testing.assert_array_equal(obs['redshift'], z_values)
        np.testing.assert_array_equal(obs['DM_over_rd'], dm_values)
        np.testing.assert_array_equal(obs['DH_over_rd'], dh_values)
        
        # Convert back
        reconverted = FormatConverter.dataset_dict_to_standard(dataset_dict, 'bao_ani')
        np.testing.assert_array_equal(reconverted.observable, combined_obs)
    
    def test_unsupported_dataset_type(self):
        """Test handling of unsupported dataset types."""
        with self.assertRaises(ValueError):
            FormatConverter.standard_to_dataset_dict(self.bao_dataset, 'unsupported_type')
    
    def test_batch_convert_datasets(self):
        """Test batch conversion of multiple datasets."""
        # Create mock output manager
        mock_output_manager = Mock()
        mock_output_manager.save_dataset.return_value = {'csv': Path('/test/path.csv')}
        
        # Create test dataset dicts
        dataset_dicts = {
            'bao': {
                'dataset_type': 'bao',
                'observations': {
                    'redshift': self.bao_dataset.z,
                    'DV_over_rs': self.bao_dataset.observable
                },
                'covariance': self.bao_dataset.covariance,
                'metadata': {'source': 'BAO_Compilation'}
            },
            'sn': {
                'dataset_type': 'sn',
                'observations': {
                    'redshift': self.sn_dataset.z,
                    'distance_modulus': self.sn_dataset.observable,
                    'sigma_mu': self.sn_dataset.uncertainty
                },
                'covariance': self.sn_dataset.covariance,
                'metadata': {'source': 'Pantheon+'}
            }
        }
        
        # Batch convert
        results = FormatConverter.batch_convert_datasets(
            dataset_dicts, mock_output_manager, ['csv']
        )
        
        # Check results
        self.assertIn('bao', results)
        self.assertIn('sn', results)
        self.assertEqual(results['bao']['status'], 'success')
        self.assertEqual(results['sn']['status'], 'success')
        
        # Check that save_dataset was called for each dataset
        self.assertEqual(mock_output_manager.save_dataset.call_count, 2)


if __name__ == '__main__':
    unittest.main()