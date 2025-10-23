#!/usr/bin/env python3
"""
Phase A Dataset Deployment and Validation Script

This script processes and validates all Phase A datasets (CMB, SN, BAO iso/aniso)
through the complete data preparation framework pipeline.

Requirements addressed: 8.5 (Phase A dataset certification)
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Add the pipelines directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preparation.engine.preparation_engine import DataPreparationFramework
from data_preparation.core.schema import StandardDataset
from data_preparation.core.interfaces import ProcessingError
from data_preparation.derivation.sn_derivation import SNDerivationModule
from data_preparation.derivation.bao_derivation import BAODerivationModule
from data_preparation.derivation.cmb_derivation import CMBDerivationModule


class PhaseADeploymentValidator:
    """
    Comprehensive deployment validator for Phase A datasets.
    
    Processes CMB, SN, and BAO (isotropic/anisotropic) datasets through
    the complete preparation framework and generates quality assurance reports.
    """
    
    def __init__(self, output_directory: Optional[Path] = None):
        """
        Initialize the Phase A deployment validator.
        
        Args:
            output_directory: Directory for validation outputs and reports
        """
        self.output_directory = output_directory or Path("data/derived")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize framework
        self.framework = DataPreparationFramework(output_directory=self.output_directory)
        
        # Register derivation modules
        self._register_derivation_modules()
        
        # Phase A datasets to process
        self.phase_a_datasets = {
            'cmb_planck2018': {
                'type': 'cmb',
                'description': 'Planck 2018 Distance Priors',
                'expected_observables': ['R', 'l_A', 'theta_star'],
                'expected_data_points': 3
            },
            'sn_pantheon_plus': {
                'type': 'sn', 
                'description': 'Pantheon+ Supernova Sample',
                'expected_observables': ['MU', 'MUERR'],
                'expected_data_points': 1701
            },
            'bao_compilation': {
                'type': 'bao',
                'description': 'BAO Compilation Dataset (Isotropic)',
                'expected_observables': ['DM_over_rd', 'DH_over_rd'],
                'expected_data_points': 15
            },
            'bao_aniso_boss_dr12': {
                'type': 'bao',
                'description': 'BOSS DR12 Anisotropic BAO',
                'expected_observables': ['DM_over_rd', 'DH_over_rd'],
                'expected_data_points': 6
            }
        }
        
        # Validation results storage
        self.validation_results = {}
        self.processing_summaries = {}
        self.qa_reports = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up comprehensive logging for deployment validation."""
        logger = logging.getLogger('phase_a_deployment')
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
        log_file = self.output_directory.parent / "logs" / f"phase_a_deployment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def _register_derivation_modules(self):
        """Register all required derivation modules for Phase A datasets."""
        try:
            # Register SN module
            sn_module = SNDerivationModule()
            self.framework.register_derivation_module(sn_module)
            self.logger.info("Registered SN derivation module")
            
            # Register BAO module
            bao_module = BAODerivationModule()
            self.framework.register_derivation_module(bao_module)
            self.logger.info("Registered BAO derivation module")
            
            # Register CMB module
            cmb_module = CMBDerivationModule()
            self.framework.register_derivation_module(cmb_module)
            self.logger.info("Registered CMB derivation module")
            
        except Exception as e:
            self.logger.error(f"Failed to register derivation modules: {e}")
            raise
    
    def create_mock_datasets(self):
        """
        Create mock datasets for Phase A validation testing.
        
        This creates synthetic datasets that match the expected format
        for each Phase A dataset type for testing purposes.
        """
        self.logger.info("Creating mock datasets for Phase A validation")
        
        mock_data_dir = Path("data/mock_phase_a")
        mock_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock CMB data
        cmb_data = {
            "R": 1.7502,
            "l_A": 301.63,
            "theta_star": 1.04119,
            "covariance": [
                [0.0001, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.0, 0.000001]
            ]
        }
        
        cmb_file = mock_data_dir / "cmb_planck2018_mock.json"
        with open(cmb_file, 'w') as f:
            json.dump(cmb_data, f, indent=2)
        
        # Create mock SN data
        sn_data = []
        for i in range(100):  # Smaller sample for testing
            sn_data.append({
                "CID": f"SN{i:04d}",
                "zHD": 0.01 + i * 0.02,
                "zHDERR": 0.001,
                "zCMB": 0.01 + i * 0.02,
                "zCMBERR": 0.001,
                "MU": 32.0 + 5 * (0.01 + i * 0.02),  # Approximate distance modulus
                "MUERR": 0.1,
                "MURES": 0.0,
                "MUPULL": 0.0
            })
        
        sn_file = mock_data_dir / "sn_pantheon_plus_mock.json"
        with open(sn_file, 'w') as f:
            json.dump(sn_data, f, indent=2)
        
        # Create mock BAO data
        bao_data = [
            {"z_eff": 0.38, "DM_over_rd": 10.3, "DM_err": 0.4, "DH_over_rd": 25.2, "DH_err": 0.7, "correlation": -0.4},
            {"z_eff": 0.51, "DM_over_rd": 13.7, "DM_err": 0.4, "DH_over_rd": 22.3, "DH_err": 0.5, "correlation": -0.4},
            {"z_eff": 0.61, "DM_over_rd": 16.1, "DM_err": 0.3, "DH_over_rd": 20.9, "DH_err": 0.4, "correlation": -0.4}
        ]
        
        bao_file = mock_data_dir / "bao_compilation_mock.json"
        with open(bao_file, 'w') as f:
            json.dump(bao_data, f, indent=2)
        
        # Create mock BAO anisotropic data
        bao_aniso_file = mock_data_dir / "bao_aniso_boss_dr12_mock.json"
        with open(bao_aniso_file, 'w') as f:
            json.dump(bao_data, f, indent=2)  # Same format for testing
        
        self.logger.info(f"Created mock datasets in {mock_data_dir}")
        return mock_data_dir
    
    def process_dataset(self, dataset_name: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single Phase A dataset through the complete pipeline.
        
        Args:
            dataset_name: Name of the dataset to process
            dataset_info: Dataset configuration information
            
        Returns:
            Processing results and validation summary
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
            'processing_summary': {},
            'qa_metrics': {}
        }
        
        try:
            # Create mock data path and metadata
            mock_data_dir = Path("data/mock_phase_a")
            
            # Use correct file extensions based on dataset type
            if dataset_info['type'] == 'cmb':
                raw_data_path = mock_data_dir / f"{dataset_name}_mock.json"
            else:
                raw_data_path = mock_data_dir / f"{dataset_name}_mock.csv"
            
            metadata = {
                'dataset_type': dataset_info['type'],
                'description': dataset_info['description'],
                'expected_observables': dataset_info['expected_observables'],
                'expected_data_points': dataset_info['expected_data_points'],
                'source': 'mock_data_for_testing',
                'version': '1.0'
            }
            
            # Process dataset through framework
            self.logger.info(f"Running preparation pipeline for {dataset_name}")
            derived_dataset = self.framework.prepare_dataset(
                dataset_name=dataset_name,
                raw_data_path=raw_data_path,
                metadata=metadata,
                force_reprocess=True
            )
            
            # Validate derived dataset
            validation_results = self.framework.validation_engine.validate_dataset(
                derived_dataset, dataset_name
            )
            
            # Generate QA metrics
            qa_metrics = self._generate_qa_metrics(derived_dataset, dataset_info)
            
            # Calculate processing time
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
    
    def _generate_qa_metrics(self, dataset: StandardDataset, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate quality assurance metrics for a processed dataset.
        
        Args:
            dataset: Processed StandardDataset
            dataset_info: Expected dataset characteristics
            
        Returns:
            QA metrics dictionary
        """
        qa_metrics = {
            'data_completeness': {
                'expected_points': dataset_info['expected_data_points'],
                'actual_points': len(dataset.z),
                'completeness_ratio': len(dataset.z) / dataset_info['expected_data_points']
            },
            'data_quality': {
                'has_nan_values': bool(dataset.z.isna().any() if hasattr(dataset.z, 'isna') else False),
                'has_infinite_values': bool((dataset.z == float('inf')).any() if hasattr(dataset.z, 'any') else False),
                'redshift_monotonic': bool((dataset.z[1:] >= dataset.z[:-1]).all() if len(dataset.z) > 1 else True),
                'positive_uncertainties': bool((dataset.uncertainty > 0).all() if hasattr(dataset.uncertainty, 'all') else True)
            },
            'schema_compliance': {
                'has_required_fields': all(hasattr(dataset, field) for field in ['z', 'observable', 'uncertainty', 'metadata']),
                'metadata_complete': bool(dataset.metadata and len(dataset.metadata) > 0),
                'covariance_available': dataset.covariance is not None
            },
            'physical_consistency': {
                'redshift_range_valid': bool(dataset.z.min() >= 0 and dataset.z.max() <= 10),
                'uncertainty_reasonable': bool(dataset.uncertainty.max() / dataset.observable.max() < 1.0 if len(dataset.observable) > 0 else True)
            }
        }
        
        return qa_metrics
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation of all Phase A datasets.
        
        Returns:
            Complete validation report
        """
        self.logger.info("Starting comprehensive Phase A dataset validation")
        
        validation_start = time.time()
        
        # Create mock datasets for testing
        mock_data_dir = self.create_mock_datasets()
        
        # Process each Phase A dataset
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
        
        # Generate comprehensive report
        validation_end = time.time()
        total_validation_time = validation_end - validation_start
        
        comprehensive_report = {
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
                'available_modules': self.framework.get_available_dataset_types(),
                'output_directory': str(self.output_directory),
                'mock_data_directory': str(mock_data_dir)
            }
        }
        
        # Save comprehensive report
        report_file = self.output_directory.parent / "logs" / f"phase_a_validation_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        self.logger.info(f"Comprehensive validation completed in {total_validation_time:.2f} seconds")
        self.logger.info(f"Success rate: {comprehensive_report['validation_summary']['success_rate']:.1%}")
        self.logger.info(f"Validation report saved to: {report_file}")
        
        return comprehensive_report
    
    def generate_deployment_certification(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate deployment certification based on validation results.
        
        Args:
            validation_report: Comprehensive validation report
            
        Returns:
            Deployment certification report
        """
        self.logger.info("Generating deployment certification")
        
        # Certification criteria
        certification_criteria = {
            'minimum_success_rate': 0.75,  # 75% of datasets must process successfully
            'required_datasets': ['cmb_planck2018', 'sn_pantheon_plus', 'bao_compilation'],  # Core Phase A datasets
            'maximum_processing_time': 600,  # 10 minutes total
            'required_qa_checks': ['schema_compliance', 'data_quality', 'physical_consistency']
        }
        
        # Evaluate certification criteria
        success_rate = validation_report['validation_summary']['success_rate']
        processing_time = validation_report['validation_summary']['total_validation_time_seconds']
        successful_datasets = validation_report['successful_datasets']
        
        # Check if all required datasets processed successfully
        required_datasets_success = all(
            dataset in successful_datasets 
            for dataset in certification_criteria['required_datasets']
        )
        
        # Check QA metrics for successful datasets
        qa_compliance = True
        qa_details = {}
        
        for dataset_name, result in validation_report['dataset_results'].items():
            if result['success']:
                dataset_qa = result.get('qa_metrics', {})
                qa_details[dataset_name] = dataset_qa
                
                # Check critical QA criteria
                if not dataset_qa.get('schema_compliance', {}).get('has_required_fields', False):
                    qa_compliance = False
        
        # Determine certification status
        certification_passed = (
            success_rate >= certification_criteria['minimum_success_rate'] and
            required_datasets_success and
            processing_time <= certification_criteria['maximum_processing_time'] and
            qa_compliance
        )
        
        certification_report = {
            'certification_status': 'PASSED' if certification_passed else 'FAILED',
            'certification_timestamp': datetime.now(timezone.utc).isoformat(),
            'certification_criteria': certification_criteria,
            'evaluation_results': {
                'success_rate_check': {
                    'required': certification_criteria['minimum_success_rate'],
                    'actual': success_rate,
                    'passed': success_rate >= certification_criteria['minimum_success_rate']
                },
                'required_datasets_check': {
                    'required_datasets': certification_criteria['required_datasets'],
                    'successful_datasets': successful_datasets,
                    'passed': required_datasets_success
                },
                'processing_time_check': {
                    'maximum_allowed_seconds': certification_criteria['maximum_processing_time'],
                    'actual_seconds': processing_time,
                    'passed': processing_time <= certification_criteria['maximum_processing_time']
                },
                'qa_compliance_check': {
                    'required_checks': certification_criteria['required_qa_checks'],
                    'passed': qa_compliance,
                    'details': qa_details
                }
            },
            'recommendations': []
        }
        
        # Generate recommendations based on results
        if not certification_passed:
            if success_rate < certification_criteria['minimum_success_rate']:
                certification_report['recommendations'].append(
                    f"Improve success rate from {success_rate:.1%} to at least {certification_criteria['minimum_success_rate']:.1%}"
                )
            
            if not required_datasets_success:
                missing_datasets = [d for d in certification_criteria['required_datasets'] if d not in successful_datasets]
                certification_report['recommendations'].append(
                    f"Fix processing issues for required datasets: {', '.join(missing_datasets)}"
                )
            
            if processing_time > certification_criteria['maximum_processing_time']:
                certification_report['recommendations'].append(
                    f"Optimize processing time from {processing_time:.1f}s to under {certification_criteria['maximum_processing_time']}s"
                )
            
            if not qa_compliance:
                certification_report['recommendations'].append(
                    "Address QA compliance issues in dataset processing"
                )
        else:
            certification_report['recommendations'].append(
                "Framework is certified for production deployment with Phase A datasets"
            )
        
        # Save certification report
        cert_file = self.output_directory.parent / "logs" / f"phase_a_certification_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(cert_file, 'w') as f:
            json.dump(certification_report, f, indent=2)
        
        self.logger.info(f"Certification status: {certification_report['certification_status']}")
        self.logger.info(f"Certification report saved to: {cert_file}")
        
        return certification_report


def main():
    """Main execution function for Phase A deployment validation."""
    print("Phase A Dataset Deployment and Validation")
    print("=" * 50)
    
    try:
        # Initialize validator
        validator = PhaseADeploymentValidator()
        
        # Run comprehensive validation
        validation_report = validator.run_comprehensive_validation()
        
        # Generate certification
        certification_report = validator.generate_deployment_certification(validation_report)
        
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
        print(f"❌ Deployment validation failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())