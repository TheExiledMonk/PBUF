#!/usr/bin/env python3
"""
Simplified Phase A Dataset Validation Script

This script validates the data preparation framework with Phase A datasets
without requiring the full registry system.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Add the pipelines directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only the core components we need
from data_preparation.core.schema import StandardDataset
from data_preparation.core.validation import ValidationEngine
from data_preparation.derivation.sn_derivation import SNDerivationModule
from data_preparation.derivation.bao_derivation import BAODerivationModule
from data_preparation.derivation.cmb_derivation import CMBDerivationModule


class SimplePhaseAValidator:
    """
    Simplified Phase A dataset validator that tests the framework
    without requiring the full registry system.
    """
    
    def __init__(self, output_directory: Optional[Path] = None):
        """Initialize the validator."""
        self.output_directory = output_directory or Path("data/derived")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize validation engine
        self.validation_engine = ValidationEngine()
        
        # Initialize derivation modules
        self.derivation_modules = {
            'sn': SNDerivationModule(),
            'bao': BAODerivationModule(),
            'cmb': CMBDerivationModule()
        }
        
        # Phase A datasets to test
        self.phase_a_datasets = {
            'cmb_planck2018': {
                'type': 'cmb',
                'description': 'Planck 2018 Distance Priors',
                'file_extension': '.json'
            },
            'sn_pantheon_plus': {
                'type': 'sn', 
                'description': 'Pantheon+ Supernova Sample',
                'file_extension': '.csv'
            },
            'bao_compilation': {
                'type': 'bao',
                'description': 'BAO Compilation Dataset (Isotropic)',
                'file_extension': '.csv'
            },
            'bao_aniso_boss_dr12': {
                'type': 'bao',
                'description': 'BOSS DR12 Anisotropic BAO',
                'file_extension': '.csv'
            }
        }
        
        self.validation_results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for validation."""
        logger = logging.getLogger('simple_phase_a_validator')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_directory.parent / "logs" / f"simple_phase_a_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def process_dataset(self, dataset_name: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single Phase A dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset_info: Dataset configuration
            
        Returns:
            Processing results
        """
        self.logger.info(f"Processing dataset: {dataset_name}")
        
        processing_start = time.time()
        result = {
            'dataset_name': dataset_name,
            'dataset_type': dataset_info['type'],
            'description': dataset_info['description'],
            'processing_start': datetime.now(timezone.utc).isoformat(),
            'success': False,
            'error': None,
            'validation_results': {},
            'qa_metrics': {}
        }
        
        try:
            # Get mock data path
            mock_data_dir = Path("data/mock_phase_a")
            raw_data_path = mock_data_dir / f"{dataset_name}_mock{dataset_info['file_extension']}"
            
            if not raw_data_path.exists():
                raise FileNotFoundError(f"Mock data file not found: {raw_data_path}")
            
            # Create metadata
            metadata = {
                'dataset_type': dataset_info['type'],
                'description': dataset_info['description'],
                'source': 'mock_data_for_testing',
                'version': '1.0'
            }
            
            # Get appropriate derivation module
            module = self.derivation_modules[dataset_info['type']]
            
            # Validate input
            self.logger.info(f"Validating input for {dataset_name}")
            if not module.validate_input(raw_data_path, metadata):
                raise ValueError("Input validation failed")
            
            # Process dataset
            self.logger.info(f"Processing {dataset_name}")
            derived_dataset = module.derive(raw_data_path, metadata)
            
            # Validate output
            self.logger.info(f"Validating output for {dataset_name}")
            validation_results = self.validation_engine.validate_dataset(
                derived_dataset, dataset_name
            )
            
            # Generate QA metrics
            qa_metrics = self._generate_qa_metrics(derived_dataset)
            
            processing_time = time.time() - processing_start
            
            result.update({
                'success': True,
                'processing_end': datetime.now(timezone.utc).isoformat(),
                'processing_time_seconds': processing_time,
                'validation_results': validation_results,
                'qa_metrics': qa_metrics,
                'derived_dataset_summary': {
                    'z_range': [float(derived_dataset.z.min()), float(derived_dataset.z.max())],
                    'n_data_points': len(derived_dataset.z),
                    'has_covariance': derived_dataset.covariance is not None,
                    'observable_range': [float(derived_dataset.observable.min()), float(derived_dataset.observable.max())]
                }
            })
            
            self.logger.info(f"Successfully processed {dataset_name} in {processing_time:.2f} seconds")
            
        except Exception as e:
            processing_time = time.time() - processing_start
            error_info = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            
            result.update({
                'success': False,
                'error': error_info,
                'processing_end': datetime.now(timezone.utc).isoformat(),
                'processing_time_seconds': processing_time
            })
            
            self.logger.error(f"Failed to process {dataset_name}: {e}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        return result
    
    def _generate_qa_metrics(self, dataset: StandardDataset) -> Dict[str, Any]:
        """Generate quality assurance metrics for a processed dataset."""
        qa_metrics = {
            'data_completeness': {
                'has_redshifts': len(dataset.z) > 0,
                'has_observables': len(dataset.observable) > 0,
                'has_uncertainties': len(dataset.uncertainty) > 0,
                'arrays_same_length': len(dataset.z) == len(dataset.observable) == len(dataset.uncertainty)
            },
            'data_quality': {
                'no_nan_redshifts': not any(x != x for x in dataset.z),  # Check for NaN
                'no_nan_observables': not any(x != x for x in dataset.observable),
                'no_nan_uncertainties': not any(x != x for x in dataset.uncertainty),
                'positive_uncertainties': all(u > 0 for u in dataset.uncertainty),
                'non_negative_redshifts': all(z >= 0 for z in dataset.z)
            },
            'schema_compliance': {
                'has_required_fields': all(hasattr(dataset, field) for field in ['z', 'observable', 'uncertainty', 'metadata']),
                'metadata_is_dict': isinstance(dataset.metadata, dict),
                'covariance_available': dataset.covariance is not None
            },
            'physical_consistency': {
                'redshift_range_reasonable': 0 <= min(dataset.z) and max(dataset.z) <= 10,
                'uncertainty_reasonable': max(dataset.uncertainty) / max(abs(o) for o in dataset.observable) < 1.0 if len(dataset.observable) > 0 else True
            }
        }
        
        return qa_metrics
    
    def run_validation(self) -> Dict[str, Any]:
        """
        Run validation of all Phase A datasets.
        
        Returns:
            Comprehensive validation report
        """
        self.logger.info("Starting Phase A dataset validation")
        
        validation_start = time.time()
        
        # Process each dataset
        all_results = {}
        successful_datasets = []
        failed_datasets = []
        
        for dataset_name, dataset_info in self.phase_a_datasets.items():
            self.logger.info(f"Processing Phase A dataset: {dataset_name}")
            
            result = self.process_dataset(dataset_name, dataset_info)
            all_results[dataset_name] = result
            
            if result['success']:
                successful_datasets.append(dataset_name)
            else:
                failed_datasets.append(dataset_name)
        
        # Generate report
        validation_end = time.time()
        total_validation_time = validation_end - validation_start
        
        report = {
            'validation_summary': {
                'total_datasets': len(self.phase_a_datasets),
                'successful_datasets': len(successful_datasets),
                'failed_datasets': len(failed_datasets),
                'success_rate': len(successful_datasets) / len(self.phase_a_datasets),
                'total_validation_time_seconds': total_validation_time,
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            },
            'dataset_results': all_results,
            'successful_datasets': successful_datasets,
            'failed_datasets': failed_datasets,
            'framework_status': {
                'available_modules': list(self.derivation_modules.keys()),
                'output_directory': str(self.output_directory)
            }
        }
        
        # Save report
        report_file = self.output_directory.parent / "logs" / f"simple_phase_a_validation_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"Validation completed in {total_validation_time:.2f} seconds")
        self.logger.info(f"Success rate: {report['validation_summary']['success_rate']:.1%}")
        self.logger.info(f"Validation report saved to: {report_file}")
        
        return report
    
    def generate_certification(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate certification based on validation results."""
        self.logger.info("Generating deployment certification")
        
        # Certification criteria
        success_rate = validation_report['validation_summary']['success_rate']
        processing_time = validation_report['validation_summary']['total_validation_time_seconds']
        successful_datasets = validation_report['successful_datasets']
        
        # Required datasets for Phase A
        required_datasets = ['cmb_planck2018', 'sn_pantheon_plus', 'bao_compilation']
        required_datasets_success = all(
            dataset in successful_datasets 
            for dataset in required_datasets
        )
        
        # Check QA compliance
        qa_compliance = True
        for dataset_name, result in validation_report['dataset_results'].items():
            if result['success']:
                qa_metrics = result.get('qa_metrics', {})
                # Check critical QA criteria
                if not qa_metrics.get('schema_compliance', {}).get('has_required_fields', False):
                    qa_compliance = False
                    break
        
        # Determine certification
        certification_passed = (
            success_rate >= 0.75 and  # 75% success rate
            required_datasets_success and
            processing_time <= 300 and  # 5 minutes max
            qa_compliance
        )
        
        certification_report = {
            'certification_status': 'PASSED' if certification_passed else 'FAILED',
            'certification_timestamp': datetime.now(timezone.utc).isoformat(),
            'evaluation_results': {
                'success_rate': success_rate,
                'required_datasets_success': required_datasets_success,
                'processing_time_acceptable': processing_time <= 300,
                'qa_compliance': qa_compliance
            },
            'recommendations': []
        }
        
        if not certification_passed:
            if success_rate < 0.75:
                certification_report['recommendations'].append(
                    f"Improve success rate from {success_rate:.1%} to at least 75%"
                )
            if not required_datasets_success:
                missing = [d for d in required_datasets if d not in successful_datasets]
                certification_report['recommendations'].append(
                    f"Fix processing for required datasets: {', '.join(missing)}"
                )
            if processing_time > 300:
                certification_report['recommendations'].append(
                    f"Optimize processing time from {processing_time:.1f}s to under 300s"
                )
            if not qa_compliance:
                certification_report['recommendations'].append(
                    "Address QA compliance issues"
                )
        else:
            certification_report['recommendations'].append(
                "Framework is certified for production deployment"
            )
        
        # Save certification
        cert_file = self.output_directory.parent / "logs" / f"simple_phase_a_certification_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(cert_file, 'w') as f:
            json.dump(certification_report, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"Certification status: {certification_report['certification_status']}")
        self.logger.info(f"Certification report saved to: {cert_file}")
        
        return certification_report


def main():
    """Main execution function."""
    print("Simple Phase A Dataset Validation")
    print("=" * 40)
    
    try:
        # Initialize validator
        validator = SimplePhaseAValidator()
        
        # Run validation
        validation_report = validator.run_validation()
        
        # Generate certification
        certification_report = validator.generate_certification(validation_report)
        
        # Print summary
        print(f"\nValidation Summary:")
        print(f"- Total datasets: {validation_report['validation_summary']['total_datasets']}")
        print(f"- Successful: {validation_report['validation_summary']['successful_datasets']}")
        print(f"- Failed: {validation_report['validation_summary']['failed_datasets']}")
        print(f"- Success rate: {validation_report['validation_summary']['success_rate']:.1%}")
        print(f"- Processing time: {validation_report['validation_summary']['total_validation_time_seconds']:.2f}s")
        
        print(f"\nCertification Status: {certification_report['certification_status']}")
        
        if certification_report['certification_status'] == 'PASSED':
            print("✅ Framework is certified for production deployment")
            return 0
        else:
            print("❌ Framework requires fixes before production deployment")
            print("\nRecommendations:")
            for rec in certification_report['recommendations']:
                print(f"- {rec}")
            return 1
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())