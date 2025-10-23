"""
Output manager for standardized dataset generation and storage.

This module handles the generation of analysis-ready datasets in standard format,
file I/O operations, metadata serialization, and provenance record creation.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import numpy as np

from ..core.schema import StandardDataset
from ..core.interfaces import ProvenanceRecord

# Optional pandas import for enhanced functionality
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class OutputManager:
    """
    Manages standardized output generation for analysis-ready datasets.
    
    Handles file I/O for multiple formats (CSV, Parquet, NumPy), metadata
    serialization, and provenance record creation as specified in requirements 6.2, 3.2.
    """
    
    def __init__(self, output_base_path: Union[str, Path] = "data/derived"):
        """
        Initialize output manager.
        
        Args:
            output_base_path: Base directory for output files
        """
        self.output_base_path = Path(output_base_path)
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
        # Supported output formats
        self.supported_formats = ['csv', 'parquet', 'numpy', 'json']
    
    def save_dataset(
        self,
        dataset: StandardDataset,
        dataset_name: str,
        formats: Optional[List[str]] = None,
        provenance: Optional[ProvenanceRecord] = None,
        timestamp: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Save analysis-ready dataset in specified formats with provenance.
        
        Args:
            dataset: StandardDataset to save
            dataset_name: Name identifier for the dataset
            formats: List of output formats ('csv', 'parquet', 'numpy', 'json')
            provenance: Provenance record for traceability
            timestamp: Optional timestamp string (defaults to current time)
            
        Returns:
            Dictionary mapping format names to output file paths
            
        Requirements: 6.2, 3.2
        """
        if formats is None:
            formats = ['csv', 'numpy']  # Default formats
        
        # Validate formats
        invalid_formats = set(formats) - set(self.supported_formats)
        if invalid_formats:
            raise ValueError(f"Unsupported formats: {invalid_formats}. Supported: {self.supported_formats}")
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validate dataset before saving
        dataset.validate_all()
        
        # Create output filename base
        filename_base = f"{dataset_name}_derived_{timestamp}"
        
        output_paths = {}
        
        # Save in each requested format
        for format_name in formats:
            if format_name == 'csv':
                output_path = self._save_as_csv(dataset, filename_base)
            elif format_name == 'parquet':
                output_path = self._save_as_parquet(dataset, filename_base)
            elif format_name == 'numpy':
                output_path = self._save_as_numpy(dataset, filename_base)
            elif format_name == 'json':
                output_path = self._save_as_json(dataset, filename_base)
            else:
                raise ValueError(f"Unsupported format: {format_name}")
            
            output_paths[format_name] = output_path
        
        # Save metadata and provenance separately
        metadata_path = self._save_metadata(dataset, filename_base, provenance)
        output_paths['metadata'] = metadata_path
        
        if provenance:
            provenance_path = self._save_provenance(provenance, filename_base)
            output_paths['provenance'] = provenance_path
        
        return output_paths
    
    def _save_as_csv(self, dataset: StandardDataset, filename_base: str) -> Path:
        """Save dataset as CSV format."""
        output_path = self.output_base_path / f"{filename_base}.csv"
        
        if HAS_PANDAS:
            # Use pandas for better CSV handling
            data_dict = {
                'z': dataset.z,
                'observable': dataset.observable,
                'uncertainty': dataset.uncertainty
            }
            
            df = pd.DataFrame(data_dict)
            df.to_csv(output_path, index=False, float_format='%.8g')
            
            # Save covariance matrix separately if present
            if dataset.covariance is not None:
                cov_path = self.output_base_path / f"{filename_base}_covariance.csv"
                cov_df = pd.DataFrame(dataset.covariance)
                cov_df.to_csv(cov_path, index=False, float_format='%.8g')
        else:
            # Fallback to manual CSV writing
            with open(output_path, 'w') as f:
                # Write header
                f.write('z,observable,uncertainty\n')
                
                # Write data rows
                for i in range(len(dataset.z)):
                    f.write(f'{dataset.z[i]:.8g},{dataset.observable[i]:.8g},{dataset.uncertainty[i]:.8g}\n')
            
            # Save covariance matrix separately if present
            if dataset.covariance is not None:
                cov_path = self.output_base_path / f"{filename_base}_covariance.csv"
                with open(cov_path, 'w') as f:
                    # Write header
                    n_cols = dataset.covariance.shape[1]
                    header = ','.join([f'col_{i}' for i in range(n_cols)])
                    f.write(header + '\n')
                    
                    # Write covariance rows
                    for row in dataset.covariance:
                        row_str = ','.join([f'{val:.8g}' for val in row])
                        f.write(row_str + '\n')
        
        return output_path
    
    def _save_as_parquet(self, dataset: StandardDataset, filename_base: str) -> Path:
        """Save dataset as Parquet format."""
        if not HAS_PANDAS:
            raise ValueError("Parquet format requires pandas. Please install pandas or use a different format.")
        
        output_path = self.output_base_path / f"{filename_base}.parquet"
        
        # Create DataFrame with main data
        data_dict = {
            'z': dataset.z,
            'observable': dataset.observable,
            'uncertainty': dataset.uncertainty
        }
        
        df = pd.DataFrame(data_dict)
        
        # Save main data
        df.to_parquet(output_path, index=False)
        
        # Save covariance matrix separately if present
        if dataset.covariance is not None:
            cov_path = self.output_base_path / f"{filename_base}_covariance.parquet"
            cov_df = pd.DataFrame(dataset.covariance)
            cov_df.to_parquet(cov_path, index=False)
        
        return output_path
    
    def _save_as_numpy(self, dataset: StandardDataset, filename_base: str) -> Path:
        """Save dataset as NumPy format (.npz)."""
        output_path = self.output_base_path / f"{filename_base}.npz"
        
        # Prepare data dictionary
        save_dict = {
            'z': dataset.z,
            'observable': dataset.observable,
            'uncertainty': dataset.uncertainty
        }
        
        # Add covariance if present
        if dataset.covariance is not None:
            save_dict['covariance'] = dataset.covariance
        
        # Add metadata as JSON string (NumPy doesn't handle dicts directly)
        save_dict['metadata_json'] = json.dumps(dataset.metadata, default=self._json_serializer)
        
        # Save as compressed NumPy archive
        np.savez_compressed(output_path, **save_dict)
        
        return output_path
    
    def _save_as_json(self, dataset: StandardDataset, filename_base: str) -> Path:
        """Save dataset as JSON format."""
        output_path = self.output_base_path / f"{filename_base}.json"
        
        # Convert dataset to serializable dictionary
        data_dict = {
            'z': dataset.z.tolist(),
            'observable': dataset.observable.tolist(),
            'uncertainty': dataset.uncertainty.tolist(),
            'covariance': dataset.covariance.tolist() if dataset.covariance is not None else None,
            'metadata': dataset.metadata
        }
        
        # Save as JSON with proper formatting
        with open(output_path, 'w') as f:
            json.dump(data_dict, f, indent=2, default=self._json_serializer)
        
        return output_path
    
    def _save_metadata(
        self,
        dataset: StandardDataset,
        filename_base: str,
        provenance: Optional[ProvenanceRecord] = None
    ) -> Path:
        """Save dataset metadata with processing information."""
        metadata_path = self.output_base_path / f"{filename_base}_metadata.json"
        
        # Combine dataset metadata with processing info
        full_metadata = {
            'dataset_metadata': dataset.metadata,
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'data_shape': {
                    'n_points': len(dataset.z),
                    'has_covariance': dataset.covariance is not None,
                    'covariance_shape': dataset.covariance.shape if dataset.covariance is not None else None
                },
                'validation_status': 'passed',  # If we got here, validation passed
                'framework_version': '1.0.0'
            }
        }
        
        # Add provenance summary if available
        if provenance:
            full_metadata['provenance_summary'] = {
                'source_dataset': provenance.source_dataset_name,
                'source_hash': provenance.source_hash,
                'derivation_module': provenance.derivation_module,
                'processing_timestamp': provenance.processing_timestamp
            }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2, default=self._json_serializer)
        
        return metadata_path
    
    def _save_provenance(self, provenance: ProvenanceRecord, filename_base: str) -> Path:
        """Save complete provenance record."""
        provenance_path = self.output_base_path / f"{filename_base}_provenance.json"
        
        # Convert provenance to dictionary
        provenance_dict = {
            'source_dataset_name': provenance.source_dataset_name,
            'source_hash': provenance.source_hash,
            'derivation_module': provenance.derivation_module,
            'processing_timestamp': provenance.processing_timestamp,
            'environment_hash': provenance.environment_hash,
            'transformation_steps': provenance.transformation_steps,
            'validation_results': provenance.validation_results,
            'derived_dataset_hash': provenance.derived_dataset_hash
        }
        
        # Save provenance record
        with open(provenance_path, 'w') as f:
            json.dump(provenance_dict, f, indent=2, default=self._json_serializer)
        
        return provenance_path
    
    def load_dataset(self, file_path: Union[str, Path], format_type: str = 'auto') -> StandardDataset:
        """
        Load StandardDataset from file.
        
        Args:
            file_path: Path to dataset file
            format_type: Format type ('csv', 'parquet', 'numpy', 'json', 'auto')
            
        Returns:
            StandardDataset instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Auto-detect format from extension
        if format_type == 'auto':
            suffix = file_path.suffix.lower()
            if suffix == '.csv':
                format_type = 'csv'
            elif suffix == '.parquet':
                format_type = 'parquet'
            elif suffix == '.npz':
                format_type = 'numpy'
            elif suffix == '.json':
                format_type = 'json'
            else:
                raise ValueError(f"Cannot auto-detect format for file: {file_path}")
        
        # Load based on format
        if format_type == 'csv':
            return self._load_from_csv(file_path)
        elif format_type == 'parquet':
            return self._load_from_parquet(file_path)
        elif format_type == 'numpy':
            return self._load_from_numpy(file_path)
        elif format_type == 'json':
            return self._load_from_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _load_from_csv(self, file_path: Path) -> StandardDataset:
        """Load dataset from CSV format."""
        if HAS_PANDAS:
            # Use pandas for better CSV handling
            df = pd.read_csv(file_path)
            z = df['z'].values
            observable = df['observable'].values
            uncertainty = df['uncertainty'].values
            
            # Load covariance if available
            cov_path = file_path.parent / f"{file_path.stem}_covariance.csv"
            covariance = None
            if cov_path.exists():
                cov_df = pd.read_csv(cov_path)
                covariance = cov_df.values
        else:
            # Fallback to manual CSV parsing
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header
            data_lines = lines[1:]
            
            z = []
            observable = []
            uncertainty = []
            
            for line in data_lines:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    z.append(float(parts[0]))
                    observable.append(float(parts[1]))
                    uncertainty.append(float(parts[2]))
            
            z = np.array(z)
            observable = np.array(observable)
            uncertainty = np.array(uncertainty)
            
            # Load covariance if available
            cov_path = file_path.parent / f"{file_path.stem}_covariance.csv"
            covariance = None
            if cov_path.exists():
                with open(cov_path, 'r') as f:
                    cov_lines = f.readlines()
                
                # Skip header and parse covariance matrix
                cov_data = []
                for line in cov_lines[1:]:
                    row = [float(x) for x in line.strip().split(',')]
                    cov_data.append(row)
                
                if cov_data:
                    covariance = np.array(cov_data)
        
        # Load metadata if available
        metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                full_metadata = json.load(f)
                metadata = full_metadata.get('dataset_metadata', {})
        
        return StandardDataset(
            z=z,
            observable=observable,
            uncertainty=uncertainty,
            covariance=covariance,
            metadata=metadata
        )
    
    def _load_from_parquet(self, file_path: Path) -> StandardDataset:
        """Load dataset from Parquet format."""
        if not HAS_PANDAS:
            raise ValueError("Parquet format requires pandas. Please install pandas or use a different format.")
        
        # Load main data
        df = pd.read_parquet(file_path)
        
        # Load covariance if available
        cov_path = file_path.parent / f"{file_path.stem}_covariance.parquet"
        covariance = None
        if cov_path.exists():
            cov_df = pd.read_parquet(cov_path)
            covariance = cov_df.values
        
        # Load metadata if available
        metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                full_metadata = json.load(f)
                metadata = full_metadata.get('dataset_metadata', {})
        
        return StandardDataset(
            z=df['z'].values,
            observable=df['observable'].values,
            uncertainty=df['uncertainty'].values,
            covariance=covariance,
            metadata=metadata
        )
    
    def _load_from_numpy(self, file_path: Path) -> StandardDataset:
        """Load dataset from NumPy format."""
        data = np.load(file_path)
        
        # Extract metadata from JSON string
        metadata = {}
        if 'metadata_json' in data:
            metadata = json.loads(str(data['metadata_json']))
        
        return StandardDataset(
            z=data['z'],
            observable=data['observable'],
            uncertainty=data['uncertainty'],
            covariance=data.get('covariance'),
            metadata=metadata
        )
    
    def _load_from_json(self, file_path: Path) -> StandardDataset:
        """Load dataset from JSON format."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return StandardDataset(
            z=np.array(data['z']),
            observable=np.array(data['observable']),
            uncertainty=np.array(data['uncertainty']),
            covariance=np.array(data['covariance']) if data['covariance'] is not None else None,
            metadata=data.get('metadata', {})
        )
    
    def generate_processing_summary(
        self,
        dataset_name: str,
        input_info: Dict[str, Any],
        output_paths: Dict[str, Path],
        processing_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate processing summary suitable for publication materials.
        
        Args:
            dataset_name: Name of processed dataset
            input_info: Information about input data
            output_paths: Dictionary of output file paths
            processing_stats: Processing statistics and metrics
            
        Returns:
            Dictionary with processing summary
            
        Requirements: 6.2
        """
        summary = {
            'dataset_name': dataset_name,
            'processing_timestamp': datetime.now().isoformat(),
            'input_summary': input_info,
            'output_files': {
                format_name: str(path) for format_name, path in output_paths.items()
            },
            'processing_statistics': processing_stats,
            'validation_status': 'passed',
            'framework_info': {
                'version': '1.0.0',
                'module': 'data_preparation.output.output_manager'
            }
        }
        
        return summary
    
    def cleanup_old_files(self, dataset_name: str, keep_latest: int = 5) -> List[Path]:
        """
        Clean up old derived dataset files, keeping only the latest versions.
        
        Args:
            dataset_name: Name of dataset to clean up
            keep_latest: Number of latest versions to keep
            
        Returns:
            List of deleted file paths
        """
        # Find all files for this dataset
        pattern = f"{dataset_name}_derived_*"
        all_files = list(self.output_base_path.glob(pattern))
        
        if len(all_files) <= keep_latest:
            return []  # Nothing to clean up
        
        # Group files by timestamp (extract from filename)
        file_groups = {}
        for file_path in all_files:
            # Extract timestamp from filename
            parts = file_path.stem.split('_')
            if len(parts) >= 3:
                timestamp = '_'.join(parts[-2:])  # Last two parts should be date_time
                if timestamp not in file_groups:
                    file_groups[timestamp] = []
                file_groups[timestamp].append(file_path)
        
        # Sort timestamps and keep only the latest
        sorted_timestamps = sorted(file_groups.keys(), reverse=True)
        timestamps_to_delete = sorted_timestamps[keep_latest:]
        
        deleted_files = []
        for timestamp in timestamps_to_delete:
            for file_path in file_groups[timestamp]:
                try:
                    file_path.unlink()
                    deleted_files.append(file_path)
                except OSError:
                    pass  # File might already be deleted
        
        return deleted_files
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for NumPy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")