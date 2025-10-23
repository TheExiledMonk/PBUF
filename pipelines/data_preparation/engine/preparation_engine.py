"""
Main orchestration engine for the data preparation framework.

This module coordinates the entire data preparation workflow, including
dataset loading, module dispatching, validation, and error handling.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

from ..core.schema import StandardDataset
from ..core.interfaces import DerivationModule, ProcessingError
from ..core.validation import ValidationEngine
from ..core.registry_integration import RegistryIntegration
from ..core.error_handling import (
    EnhancedProcessingError, ErrorRecoveryManager, ProcessingLogger,
    ProcessingStage, ErrorSeverity
)
from ..core.transformation_logging import TransformationLogger, SystemHealthMonitor


class DataPreparationFramework:
    """
    Main orchestration engine that coordinates the data preparation workflow.
    
    This class manages the complete pipeline from raw dataset loading through
    standardized output generation, including error handling and logging.
    """
    
    def __init__(self, registry_manager=None, output_directory: Optional[Path] = None):
        """
        Initialize the data preparation framework.
        
        Args:
            registry_manager: Optional registry manager for dataset retrieval
            output_directory: Optional directory for derived dataset outputs
        """
        self.registry_manager = registry_manager
        self.registry_integration = RegistryIntegration(registry_manager) if registry_manager else None
        self.derivation_modules: Dict[str, DerivationModule] = {}
        self.validation_engine = ValidationEngine()
        self.logger = logging.getLogger(__name__)
        
        # Set up output directory
        self.output_directory = output_directory or Path("data/derived")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Environment hash for deterministic processing
        self._environment_hash = None
        
        # Processing cache for avoiding reprocessing
        self._processing_cache: Dict[str, str] = {}
        
        # Enhanced error handling and logging systems
        self.processing_logger = ProcessingLogger(
            name="data_preparation_framework",
            log_directory=self.output_directory.parent / "logs"
        )
        self.error_recovery_manager = ErrorRecoveryManager(self.processing_logger.logger)
        self.system_health_monitor = SystemHealthMonitor()
        
        # Load available derivation modules
        self._load_derivation_modules()
        
        # Set up detailed logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up detailed logging for the framework."""
        # Configure logger for detailed processing information
        self.logger.setLevel(logging.INFO)
        
        # Create formatter for detailed logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def _load_derivation_modules(self):
        """
        Dynamically detect and load all available derivation modules.
        
        This method implements the plugin-like architecture by discovering
        derivation modules at runtime.
        """
        # For now, this is a placeholder for the module loading system
        # In task 5, we'll implement the actual derivation modules
        self.logger.info("Derivation module loading system initialized")
        self.logger.info("Available modules will be loaded when implemented")
        
        # Initialize module registry for future use
        self._module_registry = {
            'sn': None,      # Will be loaded in task 5.1
            'bao': None,     # Will be loaded in task 5.2
            'cmb': None,     # Will be loaded in task 5.3
            'cc': None,      # Will be loaded in task 5.4
            'rsd': None      # Will be loaded in task 5.5
        }
    
    def register_derivation_module(self, module: DerivationModule):
        """
        Register a derivation module for a specific dataset type.
        
        Args:
            module: DerivationModule instance to register
        """
        dataset_type = module.dataset_type
        self.derivation_modules[dataset_type] = module
        self.logger.info(f"Registered derivation module for dataset type: {dataset_type}")
    
    def get_available_dataset_types(self) -> List[str]:
        """
        Get list of supported dataset types.
        
        Returns:
            List of dataset type identifiers
        """
        return list(self.derivation_modules.keys())
    
    def _get_derivation_module(self, dataset_name: str, dataset_type: Optional[str] = None) -> DerivationModule:
        """
        Get appropriate derivation module for dataset.
        
        Args:
            dataset_name: Name of dataset to process
            dataset_type: Optional explicit dataset type
            
        Returns:
            DerivationModule instance for the dataset type
            
        Raises:
            ProcessingError: If no suitable module found
        """
        # Use provided dataset type or infer from registry/name
        if dataset_type is None:
            if self.registry_integration:
                try:
                    processing_requirements = self.registry_integration.get_dataset_processing_requirements(dataset_name)
                    dataset_type = processing_requirements["dataset_type"]
                except ProcessingError:
                    # Fall back to name-based inference
                    dataset_type = self._infer_dataset_type(dataset_name)
            else:
                dataset_type = self._infer_dataset_type(dataset_name)
        
        self.logger.info(f"Detected dataset type '{dataset_type}' for dataset '{dataset_name}'")
        
        if dataset_type not in self.derivation_modules:
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="module_selection",
                error_type="unsupported_dataset_type",
                error_message=f"No derivation module available for dataset type: {dataset_type}",
                context={
                    'detected_type': dataset_type,
                    'available_types': list(self.derivation_modules.keys()),
                    'registered_modules': len(self.derivation_modules)
                },
                suggested_actions=[
                    f"Implement derivation module for {dataset_type}",
                    "Check dataset name and type classification",
                    "Verify module registration",
                    "Use register_derivation_module() to add missing module"
                ]
            )
        
        module = self.derivation_modules[dataset_type]
        if module is None:
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="module_selection",
                error_type="module_not_implemented",
                error_message=f"Derivation module for {dataset_type} is registered but not implemented",
                suggested_actions=[
                    f"Complete implementation of {dataset_type} derivation module",
                    "Check module initialization"
                ]
            )
        
        return module
    
    def _infer_dataset_type(self, dataset_name: str) -> str:
        """
        Infer dataset type from dataset name.
        
        This is a fallback implementation when registry metadata is not available.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Inferred dataset type
        """
        name_lower = dataset_name.lower()
        
        # More comprehensive pattern matching - order matters, check specific types first
        if any(keyword in name_lower for keyword in ['sn', 'supernova', 'supernovae', 'jla', 'pantheon']):
            return 'sn'
        elif any(keyword in name_lower for keyword in ['bao', 'baryon', 'acoustic', 'oscillation']):
            return 'bao'
        elif any(keyword in name_lower for keyword in ['cmb', 'planck', 'cosmic_microwave', 'wmap']):
            return 'cmb'
        elif any(keyword in name_lower for keyword in ['cc', 'chronometer', 'hubble', 'h_z']):
            return 'cc'
        elif any(keyword in name_lower for keyword in ['rsd', 'redshift_space', 'growth', 'f_sigma8']):
            return 'rsd'
        elif any(keyword in name_lower for keyword in ['test']):
            return 'test'
        else:
            return 'unknown'
    
    def prepare_dataset(
        self,
        dataset_name: str,
        raw_data_path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force_reprocess: bool = False
    ) -> StandardDataset:
        """
        Prepare dataset through complete processing pipeline with enhanced logging.
        
        Args:
            dataset_name: Name of dataset to process
            raw_data_path: Optional path to raw data (if not using registry)
            metadata: Optional metadata (if not using registry)
            force_reprocess: Force reprocessing even if cached result exists
            
        Returns:
            StandardDataset: Processed dataset in standard format
            
        Raises:
            EnhancedProcessingError: If processing fails at any stage
        """
        processing_start_time = datetime.now(timezone.utc)
        
        # Start system health monitoring
        self.system_health_monitor.start_monitoring()
        
        # Initialize transformation logger
        dataset_type = metadata.get("dataset_type") if metadata else self._infer_dataset_type(dataset_name)
        transformation_logger = TransformationLogger(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            output_directory=self.output_directory.parent / "logs" / "transformations"
        )
        
        try:
            # Log processing start
            self.processing_logger.log_processing_start(
                dataset_name, ProcessingStage.INITIALIZATION,
                force_reprocess=force_reprocess,
                has_raw_data_path=raw_data_path is not None,
                has_metadata=metadata is not None
            )
            
            transformation_logger.start_processing(metadata)
            
            self.logger.info(f"Starting preparation of dataset: {dataset_name}")
            self.logger.info(f"Processing started at: {processing_start_time.isoformat()}")
            
            # Check cache first (unless forced reprocessing)
            if not force_reprocess:
                self.processing_logger.log_processing_start(dataset_name, ProcessingStage.CACHE_MANAGEMENT)
                cached_result = self._check_processing_cache(dataset_name, raw_data_path, metadata)
                if cached_result:
                    self.logger.info(f"Using cached result for dataset: {dataset_name}")
                    self.processing_logger.log_processing_end(
                        dataset_name, ProcessingStage.CACHE_MANAGEMENT, True, 0.0, cache_hit=True
                    )
                    return cached_result
                self.processing_logger.log_processing_end(
                    dataset_name, ProcessingStage.CACHE_MANAGEMENT, True, 0.0, cache_hit=False
                )
            
            # 1. Retrieve raw data and metadata with error recovery
            self.processing_logger.log_processing_start(dataset_name, ProcessingStage.DATA_RETRIEVAL)
            retrieval_start = datetime.now(timezone.utc)
            
            if raw_data_path is None or metadata is None:
                raw_data_path, metadata, source_provenance = self._retrieve_from_registry_with_recovery(dataset_name)
            else:
                source_provenance = None
            
            retrieval_duration = (datetime.now(timezone.utc) - retrieval_start).total_seconds()
            self.processing_logger.log_processing_end(
                dataset_name, ProcessingStage.DATA_RETRIEVAL, True, retrieval_duration,
                source_provenance_available=source_provenance is not None
            )
            
            # 2. Get appropriate derivation module with detailed error context
            self.processing_logger.log_processing_start(dataset_name, ProcessingStage.MODULE_SELECTION)
            module_start = datetime.now(timezone.utc)
            
            dataset_type = metadata.get("dataset_type") if metadata else None
            module = self._get_derivation_module(dataset_name, dataset_type)
            self.logger.info(f"Using derivation module: {module.dataset_type} for dataset: {dataset_name}")
            
            module_duration = (datetime.now(timezone.utc) - module_start).total_seconds()
            self.processing_logger.log_processing_end(
                dataset_name, ProcessingStage.MODULE_SELECTION, True, module_duration,
                selected_module=module.dataset_type
            )
            
            # 3. Validate input data with recovery mechanisms
            self.processing_logger.log_processing_start(dataset_name, ProcessingStage.INPUT_VALIDATION)
            validation_start = datetime.now(timezone.utc)
            
            self.logger.info("Validating input data")
            try:
                if not module.validate_input(raw_data_path, metadata):
                    raise EnhancedProcessingError(
                        dataset_name=dataset_name,
                        stage=ProcessingStage.INPUT_VALIDATION,
                        error_type="input_validation_failed",
                        error_message="Input validation failed",
                        severity=ErrorSeverity.ERROR,
                        context={
                            "raw_data_path": str(raw_data_path), 
                            "metadata_keys": list(metadata.keys()) if metadata else [],
                            "file_exists": raw_data_path.exists() if raw_data_path else False,
                            "file_size": raw_data_path.stat().st_size if raw_data_path and raw_data_path.exists() else 0
                        },
                        suggested_actions=[
                            "Check raw data format and integrity",
                            "Verify file permissions and accessibility",
                            "Review dataset metadata completeness"
                        ]
                    )
                
                validation_duration = (datetime.now(timezone.utc) - validation_start).total_seconds()
                self.processing_logger.log_processing_end(
                    dataset_name, ProcessingStage.INPUT_VALIDATION, True, validation_duration
                )
                
            except Exception as e:
                validation_duration = (datetime.now(timezone.utc) - validation_start).total_seconds()
                self.processing_logger.log_processing_end(
                    dataset_name, ProcessingStage.INPUT_VALIDATION, False, validation_duration
                )
                enhanced_error = self._handle_input_validation_error(dataset_name, raw_data_path, metadata, e)
                
                # Attempt error recovery
                if self.error_recovery_manager.attempt_recovery(enhanced_error):
                    self.logger.info("Input validation error recovered, retrying...")
                    # Retry validation after recovery
                    if module.validate_input(raw_data_path, metadata):
                        self.processing_logger.log_processing_end(
                            dataset_name, ProcessingStage.INPUT_VALIDATION, True, validation_duration,
                            recovery_successful=True
                        )
                    else:
                        raise enhanced_error
                else:
                    raise enhanced_error
            
            # 4. Apply transformations with detailed logging
            self.processing_logger.log_processing_start(dataset_name, ProcessingStage.TRANSFORMATION)
            transformation_start = datetime.now(timezone.utc)
            
            self.logger.info("Applying dataset transformations")
            
            try:
                # Create enhanced derivation module wrapper for logging
                enhanced_module = self._create_enhanced_module_wrapper(module, transformation_logger)
                derived_dataset = enhanced_module.derive(raw_data_path, metadata)
                
                transformation_end = datetime.now(timezone.utc)
                transformation_duration = (transformation_end - transformation_start).total_seconds()
                
                self.logger.info(f"Transformation completed in {transformation_duration:.2f} seconds")
                self.processing_logger.log_processing_end(
                    dataset_name, ProcessingStage.TRANSFORMATION, True, transformation_duration,
                    output_shape={
                        "z_length": len(derived_dataset.z),
                        "observable_length": len(derived_dataset.observable),
                        "has_covariance": derived_dataset.covariance is not None
                    }
                )
                
            except Exception as e:
                transformation_duration = (datetime.now(timezone.utc) - transformation_start).total_seconds()
                self.processing_logger.log_processing_end(
                    dataset_name, ProcessingStage.TRANSFORMATION, False, transformation_duration
                )
                enhanced_error = self._handle_transformation_error(dataset_name, module, raw_data_path, metadata, e)
                
                # Attempt error recovery
                if self.error_recovery_manager.attempt_recovery(enhanced_error):
                    self.logger.info("Transformation error recovered, retrying...")
                    # Could implement retry logic here
                    pass
                
                raise enhanced_error
            
            # 5. Validate output with comprehensive error reporting
            self.processing_logger.log_processing_start(dataset_name, ProcessingStage.OUTPUT_VALIDATION)
            output_validation_start = datetime.now(timezone.utc)
            
            self.logger.info("Validating derived dataset")
            try:
                validation_results = self.validation_engine.validate_dataset(
                    derived_dataset, dataset_name
                )
                
                # Log validation results
                transformation_logger.log_validation_results(validation_results)
                self.processing_logger.log_validation_result(
                    dataset_name, "comprehensive_validation", 
                    validation_results['validation_passed'], validation_results
                )
                
                if not validation_results['validation_passed']:
                    raise EnhancedProcessingError(
                        dataset_name=dataset_name,
                        stage=ProcessingStage.OUTPUT_VALIDATION,
                        error_type="output_validation_failed",
                        error_message="Output validation failed",
                        severity=ErrorSeverity.ERROR,
                        context=validation_results,
                        suggested_actions=[
                            "Review transformation logic and validation rules",
                            "Check for data corruption during processing",
                            "Verify input data quality"
                        ]
                    )
                
                output_validation_duration = (datetime.now(timezone.utc) - output_validation_start).total_seconds()
                self.processing_logger.log_processing_end(
                    dataset_name, ProcessingStage.OUTPUT_VALIDATION, True, output_validation_duration
                )
                    
            except EnhancedProcessingError:
                output_validation_duration = (datetime.now(timezone.utc) - output_validation_start).total_seconds()
                self.processing_logger.log_processing_end(
                    dataset_name, ProcessingStage.OUTPUT_VALIDATION, False, output_validation_duration
                )
                raise
            except Exception as e:
                output_validation_duration = (datetime.now(timezone.utc) - output_validation_start).total_seconds()
                self.processing_logger.log_processing_end(
                    dataset_name, ProcessingStage.OUTPUT_VALIDATION, False, output_validation_duration
                )
                enhanced_error = self._handle_output_validation_error(dataset_name, derived_dataset, e)
                raise enhanced_error
            
            # 6. Generate and log transformation summary
            transformation_summary = module.get_transformation_summary()
            self.logger.info(f"Transformation summary: {transformation_summary}")
            
            # 7. Save derived dataset and update cache
            output_file_path = self._save_derived_dataset(dataset_name, derived_dataset, transformation_summary)
            
            # 8. Register with provenance system if available
            if self.registry_integration and source_provenance:
                self.processing_logger.log_processing_start(dataset_name, ProcessingStage.PROVENANCE_RECORDING)
                provenance_start = datetime.now(timezone.utc)
                
                try:
                    derived_hash = self.registry_integration.register_derived_dataset(
                        dataset_name, derived_dataset, source_provenance, 
                        transformation_summary, output_file_path
                    )
                    self.logger.info(f"Registered derived dataset with hash: {derived_hash}")
                    
                    provenance_duration = (datetime.now(timezone.utc) - provenance_start).total_seconds()
                    self.processing_logger.log_processing_end(
                        dataset_name, ProcessingStage.PROVENANCE_RECORDING, True, provenance_duration,
                        derived_hash=derived_hash
                    )
                except Exception as e:
                    provenance_duration = (datetime.now(timezone.utc) - provenance_start).total_seconds()
                    self.processing_logger.log_processing_end(
                        dataset_name, ProcessingStage.PROVENANCE_RECORDING, False, provenance_duration
                    )
                    self.logger.warning(f"Failed to register derived dataset: {e}")
            
            # 9. Update processing cache
            self._update_processing_cache(dataset_name, derived_dataset, raw_data_path, metadata)
            
            # Finalize transformation logging
            transformation_logger.end_processing({
                "output_file_path": str(output_file_path),
                "derived_dataset_hash": self._calculate_dataset_hash(derived_dataset)
            })
            
            # Save processing summaries
            transformation_logger.save_processing_summary()
            transformation_logger.save_publication_summary()
            
            processing_end_time = datetime.now(timezone.utc)
            total_duration = (processing_end_time - processing_start_time).total_seconds()
            
            self.logger.info(f"Successfully prepared dataset: {dataset_name}")
            self.logger.info(f"Total processing time: {total_duration:.2f} seconds")
            
            # Stop system health monitoring and save report
            self.system_health_monitor.stop_monitoring()
            health_summary = self.system_health_monitor.get_health_summary()
            if health_summary.get("threshold_violations", {}).get("total_count", 0) > 0:
                self.logger.warning(f"System health violations detected: {health_summary['threshold_violations']}")
            
            return derived_dataset
            
        except EnhancedProcessingError as e:
            # Log detailed error information
            self.processing_logger.log_error(e)
            self._log_processing_failure(dataset_name, e, processing_start_time)
            
            # Stop monitoring and save health report
            self.system_health_monitor.stop_monitoring()
            self.system_health_monitor.save_health_report(f"health_error_{dataset_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
            
            raise
        except ProcessingError as e:
            # Re-raise ProcessingError without wrapping to preserve error type
            self.processing_logger.log_error(e)
            self._log_processing_failure(dataset_name, e, processing_start_time)
            
            # Stop monitoring and save health report
            self.system_health_monitor.stop_monitoring()
            self.system_health_monitor.save_health_report(f"health_error_{dataset_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
            
            raise
        except Exception as e:
            # Wrap unexpected errors in EnhancedProcessingError with full context
            enhanced_error = EnhancedProcessingError(
                dataset_name=dataset_name,
                stage=ProcessingStage.INITIALIZATION,
                error_type="unexpected_error",
                error_message=str(e),
                severity=ErrorSeverity.CRITICAL,
                context={
                    "exception_type": type(e).__name__,
                    "processing_start_time": processing_start_time.isoformat(),
                    "raw_data_path": str(raw_data_path) if raw_data_path else None,
                    "has_metadata": metadata is not None
                },
                suggested_actions=[
                    "Check logs for detailed error information",
                    "Verify system resources and permissions",
                    "Report issue if error persists"
                ],
                original_exception=e,
                processing_duration=(datetime.now(timezone.utc) - processing_start_time).total_seconds()
            )
            
            self.processing_logger.log_error(enhanced_error)
            self._log_processing_failure(dataset_name, enhanced_error, processing_start_time)
            
            # Stop monitoring and save health report
            self.system_health_monitor.stop_monitoring()
            self.system_health_monitor.save_health_report(f"health_critical_{dataset_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
            
            raise enhanced_error
    
    def _retrieve_from_registry_with_recovery(self, dataset_name: str) -> Tuple[Path, Dict[str, Any], Any]:
        """
        Retrieve verified raw data from registry with error recovery.
        
        Args:
            dataset_name: Name of dataset to retrieve
            
        Returns:
            Tuple of (raw_data_path, metadata, source_provenance)
            
        Raises:
            ProcessingError: If registry retrieval fails after recovery attempts
        """
        if self.registry_integration is None:
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="data_retrieval",
                error_type="no_registry_integration",
                error_message="No registry integration configured",
                context={"registry_manager_available": self.registry_manager is not None},
                suggested_actions=[
                    "Initialize framework with registry manager",
                    "Provide raw_data_path and metadata directly"
                ]
            )
        
        try:
            # Attempt to retrieve dataset from registry
            dataset_info = self.registry_integration.get_verified_dataset(dataset_name)
            
            # Validate dataset integrity
            self.registry_integration.validate_raw_dataset_integrity(
                dataset_name, dataset_info["file_path"]
            )
            
            return (
                dataset_info["file_path"],
                dataset_info["metadata"],
                dataset_info["provenance"]
            )
            
        except ProcessingError as e:
            # Attempt recovery based on error type
            if e.error_type == "dataset_not_found":
                # Try alternative dataset names or suggest available datasets
                available_datasets = self.registry_integration.list_available_datasets()
                similar_datasets = [d["name"] for d in available_datasets 
                                 if dataset_name.lower() in d["name"].lower() or 
                                    d["name"].lower() in dataset_name.lower()]
                
                e.context["available_datasets"] = [d["name"] for d in available_datasets]
                e.context["similar_datasets"] = similar_datasets
                e.suggested_actions.extend([
                    f"Available datasets: {', '.join([d['name'] for d in available_datasets[:5]])}",
                    f"Similar datasets found: {', '.join(similar_datasets[:3])}" if similar_datasets else "No similar datasets found"
                ])
            
            elif e.error_type == "verification_failed":
                # Suggest re-download or manual verification
                e.suggested_actions.extend([
                    "Consider re-downloading the dataset",
                    "Check dataset source for updates",
                    "Manually verify dataset integrity"
                ])
            
            elif e.error_type == "file_not_found":
                # Check if file was moved or deleted
                e.suggested_actions.extend([
                    "Check if dataset file was moved or deleted",
                    "Re-run dataset download process",
                    "Verify file system integrity"
                ])
            
            self.logger.error(f"Registry retrieval failed for {dataset_name}: {e}")
            raise e
        
        except Exception as e:
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="data_retrieval",
                error_type="registry_access_error",
                error_message=f"Unexpected error accessing registry: {str(e)}",
                context={"exception_type": type(e).__name__},
                suggested_actions=[
                    "Check registry system health",
                    "Verify file system permissions",
                    "Check network connectivity if using remote registry"
                ]
            )
    
    def _handle_input_validation_error(self, dataset_name: str, raw_data_path: Path, 
                                     metadata: Dict[str, Any], error: Exception) -> EnhancedProcessingError:
        """Handle input validation errors with enhanced context."""
        if isinstance(error, EnhancedProcessingError):
            return error
        
        # Try to provide more context about the validation failure
        context = {
            "file_exists": raw_data_path.exists() if raw_data_path else False,
            "file_size": raw_data_path.stat().st_size if raw_data_path and raw_data_path.exists() else 0,
            "metadata_available": metadata is not None,
            "error_details": str(error),
            "dataset_info": {
                "raw_data_path": str(raw_data_path),
                "metadata_keys": list(metadata.keys()) if metadata else []
            }
        }
        
        return EnhancedProcessingError(
            dataset_name=dataset_name,
            stage=ProcessingStage.INPUT_VALIDATION,
            error_type="validation_error",
            error_message=f"Input validation failed: {str(error)}",
            severity=ErrorSeverity.ERROR,
            context=context,
            suggested_actions=[
                "Check file format and structure",
                "Verify file is not corrupted",
                "Review dataset documentation for expected format",
                "Check file permissions and accessibility"
            ],
            original_exception=error
        )
    
    def _handle_transformation_error(self, dataset_name: str, module: DerivationModule,
                                   raw_data_path: Path, metadata: Dict[str, Any], error: Exception) -> EnhancedProcessingError:
        """Handle transformation errors with enhanced context."""
        if isinstance(error, EnhancedProcessingError):
            return error
        
        context = {
            "module_type": module.dataset_type,
            "supported_formats": module.supported_formats,
            "file_extension": raw_data_path.suffix if raw_data_path else None,
            "error_details": str(error),
            "transformation_summary": None,
            "dataset_info": {
                "raw_data_path": str(raw_data_path),
                "metadata_keys": list(metadata.keys()) if metadata else []
            }
        }
        
        # Try to get partial transformation summary if available
        try:
            context["transformation_summary"] = module.get_transformation_summary()
        except:
            pass
        
        return EnhancedProcessingError(
            dataset_name=dataset_name,
            stage=ProcessingStage.TRANSFORMATION,
            error_type="transformation_failed",
            error_message=f"Dataset transformation failed: {str(error)}",
            severity=ErrorSeverity.ERROR,
            context=context,
            suggested_actions=[
                "Check input data format compatibility",
                "Review transformation module implementation",
                "Verify all required data fields are present",
                "Check for data corruption or missing values",
                "Review transformation parameters and assumptions"
            ],
            original_exception=error
        )
    
    def _handle_output_validation_error(self, dataset_name: str, derived_dataset: StandardDataset, error: Exception) -> EnhancedProcessingError:
        """Handle output validation errors with enhanced dataset context."""
        if isinstance(error, EnhancedProcessingError):
            return error
        
        context = {
            "dataset_shape": {
                "z_length": len(derived_dataset.z) if hasattr(derived_dataset, 'z') and derived_dataset.z is not None else 0,
                "observable_length": len(derived_dataset.observable) if hasattr(derived_dataset, 'observable') and derived_dataset.observable is not None else 0,
                "uncertainty_length": len(derived_dataset.uncertainty) if hasattr(derived_dataset, 'uncertainty') and derived_dataset.uncertainty is not None else 0,
                "has_covariance": derived_dataset.covariance is not None if hasattr(derived_dataset, 'covariance') else False,
                "covariance_shape": derived_dataset.covariance.shape if hasattr(derived_dataset, 'covariance') and derived_dataset.covariance is not None else None
            },
            "error_details": str(error),
            "dataset_info": {
                "metadata_keys": list(derived_dataset.metadata.keys()) if hasattr(derived_dataset, 'metadata') and derived_dataset.metadata else []
            }
        }
        
        # Preserve original error type if it's a ProcessingError
        if isinstance(error, ProcessingError):
            error_type = error.error_type
            error_message = error.error_message
        else:
            error_type = "output_validation_error"
            error_message = f"Output validation failed: {str(error)}"
        
        return EnhancedProcessingError(
            dataset_name=dataset_name,
            stage=ProcessingStage.OUTPUT_VALIDATION,
            error_type=error_type,
            error_message=error_message,
            severity=ErrorSeverity.ERROR,
            context=context,
            suggested_actions=[
                "Check derived dataset structure and completeness",
                "Verify transformation produced valid output",
                "Review validation rules for appropriateness",
                "Check for data type consistency",
                "Verify array shapes and dimensions"
            ],
            original_exception=error
        )
    
    def _log_processing_failure(self, dataset_name: str, error: ProcessingError, start_time: datetime):
        """Log detailed information about processing failures."""
        failure_time = datetime.now(timezone.utc)
        duration = (failure_time - start_time).total_seconds()
        
        failure_log = {
            "dataset_name": dataset_name,
            "failure_time": failure_time.isoformat(),
            "processing_duration": duration,
            "error_stage": error.stage,
            "error_type": error.error_type,
            "error_message": error.error_message,
            "context": error.context,
            "suggested_actions": error.suggested_actions
        }
        
        self.logger.error(f"Processing failure details: {json.dumps(failure_log, indent=2)}")
        
        # Save failure log to file for debugging
        failure_log_path = self.output_directory / "processing_failures.jsonl"
        try:
            with open(failure_log_path, 'a') as f:
                f.write(json.dumps(failure_log) + '\n')
        except Exception as e:
            self.logger.warning(f"Could not write failure log: {e}")
    
    def _calculate_input_hash(self, dataset_name: str, raw_data_path: Optional[Path], 
                            metadata: Optional[Dict[str, Any]]) -> str:
        """
        Calculate hash of input parameters for caching.
        
        Args:
            dataset_name: Name of dataset
            raw_data_path: Path to raw data file
            metadata: Dataset metadata
            
        Returns:
            SHA256 hash of input parameters
        """
        hash_input = {
            "dataset_name": dataset_name,
            "raw_data_path": str(raw_data_path) if raw_data_path else None,
            "metadata": metadata,
            "environment_hash": self._get_environment_hash()
        }
        
        # Create deterministic hash
        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def _get_environment_hash(self) -> str:
        """
        Get hash of current processing environment for deterministic processing.
        
        Returns:
            SHA256 hash of environment state
        """
        if self._environment_hash is None:
            # Collect environment information
            import sys
            import platform
            
            env_info = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "framework_version": "1.0.0",  # Framework version
                # Add more environment factors as needed
            }
            
            # Try to get git commit if available
            try:
                import subprocess
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], 
                    cwd=Path(__file__).parent.parent.parent.parent,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                env_info["git_commit"] = git_commit
            except:
                env_info["git_commit"] = "unknown"
            
            env_string = json.dumps(env_info, sort_keys=True)
            self._environment_hash = hashlib.sha256(env_string.encode()).hexdigest()
        
        return self._environment_hash
    
    def _check_processing_cache(self, dataset_name: str, raw_data_path: Optional[Path],
                              metadata: Optional[Dict[str, Any]]) -> Optional[StandardDataset]:
        """
        Check if processed result exists in cache.
        
        Args:
            dataset_name: Name of dataset
            raw_data_path: Path to raw data
            metadata: Dataset metadata
            
        Returns:
            Cached StandardDataset if available, None otherwise
        """
        input_hash = self._calculate_input_hash(dataset_name, raw_data_path, metadata)
        cache_file = self.output_directory / f"cache_{input_hash}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Verify cache is still valid
                if cached_data.get("input_hash") == input_hash:
                    self.logger.info(f"Found valid cache entry for dataset: {dataset_name}")
                    
                    # Reconstruct StandardDataset from cached data
                    import numpy as np
                    
                    dataset_data = cached_data["dataset"]
                    cached_dataset = StandardDataset(
                        z=np.array(dataset_data["z"]),
                        observable=np.array(dataset_data["observable"]),
                        uncertainty=np.array(dataset_data["uncertainty"]),
                        covariance=np.array(dataset_data["covariance"]) if dataset_data["covariance"] else None,
                        metadata=dataset_data["metadata"]
                    )
                    
                    return cached_dataset
                    
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {dataset_name}: {e}")
                # Remove invalid cache file
                try:
                    cache_file.unlink()
                except:
                    pass
        
        return None
    
    def _update_processing_cache(self, dataset_name: str, derived_dataset: StandardDataset,
                               raw_data_path: Optional[Path], metadata: Optional[Dict[str, Any]]):
        """
        Update processing cache with new result.
        
        Args:
            dataset_name: Name of dataset
            derived_dataset: Processed dataset
            raw_data_path: Path to raw data
            metadata: Dataset metadata
        """
        try:
            input_hash = self._calculate_input_hash(dataset_name, raw_data_path, metadata)
            cache_file = self.output_directory / f"cache_{input_hash}.json"
            
            # Convert dataset to serializable format
            cache_data = {
                "input_hash": input_hash,
                "dataset_name": dataset_name,
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "environment_hash": self._get_environment_hash(),
                "dataset": {
                    "z": derived_dataset.z.tolist(),
                    "observable": derived_dataset.observable.tolist(),
                    "uncertainty": derived_dataset.uncertainty.tolist(),
                    "covariance": derived_dataset.covariance.tolist() if derived_dataset.covariance is not None else None,
                    "metadata": derived_dataset.metadata
                }
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.info(f"Updated processing cache for dataset: {dataset_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update cache for {dataset_name}: {e}")
    
    def _save_derived_dataset(self, dataset_name: str, derived_dataset: StandardDataset,
                            transformation_summary: Dict[str, Any]) -> Path:
        """
        Save derived dataset to output directory.
        
        Args:
            dataset_name: Name of source dataset
            derived_dataset: Processed dataset
            transformation_summary: Summary of transformations applied
            
        Returns:
            Path to saved dataset file
        """
        # Create output filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_filename = f"{dataset_name}_derived_{timestamp}.json"
        output_path = self.output_directory / output_filename
        
        # Prepare output data
        output_data = {
            "dataset_name": dataset_name,
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "environment_hash": self._get_environment_hash(),
            "transformation_summary": transformation_summary,
            "data": {
                "z": derived_dataset.z.tolist(),
                "observable": derived_dataset.observable.tolist(),
                "uncertainty": derived_dataset.uncertainty.tolist(),
                "covariance": derived_dataset.covariance.tolist() if derived_dataset.covariance is not None else None,
                "metadata": derived_dataset.metadata
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, sort_keys=True)
        
        self.logger.info(f"Saved derived dataset to: {output_path}")
        return output_path
    
    def clear_processing_cache(self):
        """Clear all cached processing results."""
        cache_files = list(self.output_directory.glob("cache_*.json"))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        self.logger.info(f"Cleared {len(cache_files)} cache files")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics and performance metrics.
        
        Returns:
            Dict containing processing statistics
        """
        cache_files = list(self.output_directory.glob("cache_*.json"))
        derived_files = list(self.output_directory.glob("*_derived_*.json"))
        
        stats = {
            "cached_datasets": len(cache_files),
            "derived_datasets": len(derived_files),
            "registered_modules": len(self.derivation_modules),
            "available_dataset_types": list(self.derivation_modules.keys()),
            "output_directory": str(self.output_directory),
            "environment_hash": self._get_environment_hash()
        }
        
        # Add registry statistics if available
        if self.registry_integration:
            try:
                available_datasets = self.registry_integration.list_available_datasets()
                stats["registry_datasets"] = len(available_datasets)
                stats["ready_for_processing"] = len([d for d in available_datasets if d["ready_for_processing"]])
            except Exception as e:
                stats["registry_error"] = str(e)
        
        return stats
    
    def prepare_dataset_from_registry(self, dataset_name: str, force_reprocess: bool = False) -> StandardDataset:
        """
        Prepare dataset by retrieving it from the registry system.
        
        This method provides a simplified interface for processing datasets that are
        already registered in the dataset registry system. It handles the complete
        workflow from registry retrieval through standardized output generation.
        
        Args:
            dataset_name: Name of dataset in the registry
            force_reprocess: Force reprocessing even if cached result exists
            
        Returns:
            StandardDataset: Processed dataset in standard format
            
        Raises:
            ProcessingError: If registry retrieval or processing fails
            
        Requirements: 8.4 - Integration tests for registry integration
        """
        if self.registry_integration is None:
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="registry_retrieval",
                error_type="no_registry_integration",
                error_message="Registry integration not configured",
                context={"registry_manager_available": self.registry_manager is not None},
                suggested_actions=[
                    "Initialize framework with registry manager",
                    "Use prepare_dataset() with raw_data_path and metadata instead"
                ]
            )
        
        self.logger.info(f"Preparing dataset from registry: {dataset_name}")
        
        try:
            # Retrieve dataset information from registry
            dataset_info = self.registry_integration.get_verified_dataset(dataset_name)
            
            # Extract components
            raw_data_path = dataset_info["file_path"]
            metadata = dataset_info["metadata"]
            source_provenance = dataset_info["provenance"]
            
            # Add provenance summary to metadata for processing
            metadata["provenance_summary"] = {
                "source_dataset": dataset_name,
                "source_checksum": source_provenance.verification.sha256_actual,
                "download_timestamp": source_provenance.download_timestamp,
                "source_used": source_provenance.source_used,
                "environment_hash": self._calculate_environment_hash(),
                "processing_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Process through standard pipeline
            result = self.prepare_dataset(
                dataset_name=dataset_name,
                raw_data_path=raw_data_path,
                metadata=metadata,
                force_reprocess=force_reprocess
            )
            
            # Add registry-specific metadata
            result.metadata["registry_integration"] = {
                "used": True,
                "dataset_name": dataset_name,
                "verification_status": "passed",
                "source_provenance_hash": self._calculate_provenance_hash(source_provenance)
            }
            
            return result
            
        except ProcessingError:
            # Re-raise processing errors as-is
            raise
        except Exception as e:
            # Wrap other exceptions in ProcessingError
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="registry_retrieval",
                error_type="registry_integration_error",
                error_message=f"Failed to prepare dataset from registry: {str(e)}",
                context={
                    "exception_type": type(e).__name__,
                    "registry_available": self.registry_integration is not None
                },
                suggested_actions=[
                    "Check dataset exists in registry",
                    "Verify registry system is accessible",
                    "Check dataset verification status"
                ]
            ) from e
    
    def validate_framework_setup(self) -> Dict[str, Any]:
        """
        Validate framework setup and configuration.
        
        Returns:
            Dict containing setup validation results
        """
        results = {
            'framework_ready': True,
            'issues': [],
            'warnings': [],
            'available_modules': list(self.derivation_modules.keys()),
            'validation_rules': [rule.rule_name for rule in self.validation_engine.rules],
            'output_directory': str(self.output_directory),
            'output_directory_writable': self.output_directory.exists() and self.output_directory.is_dir()
        }
        
        # Check derivation modules
        if not self.derivation_modules:
            results['framework_ready'] = False
            results['issues'].append("No derivation modules registered")
        else:
            # Check which modules are actually implemented
            implemented_modules = [name for name, module in self.derivation_modules.items() if module is not None]
            placeholder_modules = [name for name, module in self.derivation_modules.items() if module is None]
            
            results['implemented_modules'] = implemented_modules
            results['placeholder_modules'] = placeholder_modules
            
            if placeholder_modules:
                results['warnings'].append(f"Modules not yet implemented: {', '.join(placeholder_modules)}")
        
        # Check registry integration
        if self.registry_manager is None:
            results['warnings'].append("No registry manager configured - manual data paths required")
        elif self.registry_integration is None:
            results['issues'].append("Registry manager provided but integration not initialized")
        
        # Check output directory
        if not results['output_directory_writable']:
            results['framework_ready'] = False
            results['issues'].append(f"Output directory not writable: {self.output_directory}")
        
        # Check validation engine
        if not self.validation_engine.rules:
            results['warnings'].append("No validation rules configured")
        
        return results
    
    def prepare_multiple_datasets(self, dataset_names: List[str], 
                                parallel: bool = False, 
                                force_reprocess: bool = False) -> Dict[str, StandardDataset]:
        """
        Prepare multiple datasets through the processing pipeline.
        
        Args:
            dataset_names: List of dataset names to process
            parallel: Whether to process datasets in parallel (deterministic)
            force_reprocess: Force reprocessing even if cached results exist
            
        Returns:
            Dict mapping dataset names to processed StandardDatasets
            
        Raises:
            ProcessingError: If any dataset processing fails
        """
        results = {}
        failed_datasets = {}
        
        self.logger.info(f"Starting batch processing of {len(dataset_names)} datasets")
        
        if parallel:
            # Use deterministic parallel processing
            results, failed_datasets = self._process_datasets_parallel(dataset_names, force_reprocess)
        else:
            # Sequential processing
            for dataset_name in dataset_names:
                try:
                    self.logger.info(f"Processing dataset {dataset_name}")
                    results[dataset_name] = self.prepare_dataset(dataset_name, force_reprocess=force_reprocess)
                except ProcessingError as e:
                    self.logger.error(f"Failed to process dataset {dataset_name}: {e}")
                    failed_datasets[dataset_name] = e
                    # Continue with other datasets
        
        # Report results
        success_count = len(results)
        failure_count = len(failed_datasets)
        
        self.logger.info(f"Batch processing completed: {success_count} successful, {failure_count} failed")
        
        if failed_datasets:
            self.logger.warning(f"Failed datasets: {list(failed_datasets.keys())}")
            
            # Create summary error for failed datasets
            if success_count == 0:
                # All datasets failed
                raise ProcessingError(
                    dataset_name="batch_processing",
                    stage="batch_processing",
                    error_type="all_datasets_failed",
                    error_message=f"All {failure_count} datasets failed processing",
                    context={"failed_datasets": {name: str(error) for name, error in failed_datasets.items()}},
                    suggested_actions=[
                        "Check individual dataset error messages",
                        "Verify input data quality",
                        "Check framework configuration"
                    ]
                )
        
        return results
    
    def _process_datasets_parallel(self, dataset_names: List[str], 
                                 force_reprocess: bool) -> Tuple[Dict[str, StandardDataset], Dict[str, ProcessingError]]:
        """
        Process datasets in parallel using deterministic methods.
        
        Args:
            dataset_names: List of dataset names to process
            force_reprocess: Force reprocessing flag
            
        Returns:
            Tuple of (successful_results, failed_results)
        """
        # For now, implement sequential processing to ensure deterministic behavior
        # In the future, this could use multiprocessing with careful synchronization
        self.logger.info("Parallel processing requested - using sequential processing for deterministic results")
        
        results = {}
        failed_datasets = {}
        
        for dataset_name in dataset_names:
            try:
                results[dataset_name] = self.prepare_dataset(dataset_name, force_reprocess=force_reprocess)
            except ProcessingError as e:
                failed_datasets[dataset_name] = e
        
        return results, failed_datasets
    
    def create_processing_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a processing workflow from configuration.
        
        Args:
            workflow_config: Configuration dict containing:
                - datasets: List of dataset names or patterns
                - processing_options: Dict of processing parameters
                - output_format: Desired output format
                - validation_level: Validation strictness level
                
        Returns:
            Dict containing workflow execution results
        """
        workflow_start = datetime.now(timezone.utc)
        
        self.logger.info("Creating processing workflow from configuration")
        
        # Extract configuration
        datasets = workflow_config.get("datasets", [])
        processing_options = workflow_config.get("processing_options", {})
        output_format = workflow_config.get("output_format", "standard")
        validation_level = workflow_config.get("validation_level", "standard")
        
        # Expand dataset patterns if needed
        expanded_datasets = self._expand_dataset_patterns(datasets)
        
        # Configure validation engine based on level
        self._configure_validation_level(validation_level)
        
        # Process datasets
        try:
            results = self.prepare_multiple_datasets(
                expanded_datasets,
                parallel=processing_options.get("parallel", False),
                force_reprocess=processing_options.get("force_reprocess", False)
            )
            
            # Convert to requested output format
            formatted_results = self._format_workflow_output(results, output_format)
            
            workflow_end = datetime.now(timezone.utc)
            workflow_duration = (workflow_end - workflow_start).total_seconds()
            
            workflow_summary = {
                "workflow_config": workflow_config,
                "execution_start": workflow_start.isoformat(),
                "execution_end": workflow_end.isoformat(),
                "execution_duration": workflow_duration,
                "datasets_processed": len(results),
                "datasets_requested": len(expanded_datasets),
                "success_rate": len(results) / len(expanded_datasets) if expanded_datasets else 0,
                "results": formatted_results,
                "environment_hash": self._get_environment_hash()
            }
            
            self.logger.info(f"Workflow completed successfully in {workflow_duration:.2f} seconds")
            return workflow_summary
            
        except ProcessingError as e:
            workflow_end = datetime.now(timezone.utc)
            workflow_duration = (workflow_end - workflow_start).total_seconds()
            
            workflow_summary = {
                "workflow_config": workflow_config,
                "execution_start": workflow_start.isoformat(),
                "execution_end": workflow_end.isoformat(),
                "execution_duration": workflow_duration,
                "error": e.to_dict(),
                "environment_hash": self._get_environment_hash()
            }
            
            self.logger.error(f"Workflow failed after {workflow_duration:.2f} seconds: {e}")
            raise ProcessingError(
                dataset_name="workflow",
                stage="workflow_execution",
                error_type="workflow_failed",
                error_message=f"Processing workflow failed: {e.error_message}",
                context=workflow_summary,
                suggested_actions=e.suggested_actions + [
                    "Review workflow configuration",
                    "Check individual dataset processing logs"
                ]
            )
    
    def _expand_dataset_patterns(self, dataset_patterns: List[str]) -> List[str]:
        """
        Expand dataset patterns to actual dataset names.
        
        Args:
            dataset_patterns: List of dataset names or patterns (e.g., "sn_*", "bao_*")
            
        Returns:
            List of actual dataset names
        """
        expanded = []
        
        if not self.registry_integration:
            # Without registry, return patterns as-is
            return dataset_patterns
        
        try:
            available_datasets = self.registry_integration.list_available_datasets()
            available_names = [d["name"] for d in available_datasets if d["ready_for_processing"]]
            
            for pattern in dataset_patterns:
                if "*" in pattern or "?" in pattern:
                    # Pattern matching
                    import fnmatch
                    matches = [name for name in available_names if fnmatch.fnmatch(name, pattern)]
                    expanded.extend(matches)
                else:
                    # Exact name
                    if pattern in available_names:
                        expanded.append(pattern)
                    else:
                        self.logger.warning(f"Dataset '{pattern}' not found in registry")
            
        except Exception as e:
            self.logger.warning(f"Could not expand dataset patterns: {e}")
            return dataset_patterns
        
        return list(set(expanded))  # Remove duplicates
    
    def _configure_validation_level(self, validation_level: str):
        """
        Configure validation engine based on requested level.
        
        Args:
            validation_level: "strict", "standard", or "lenient"
        """
        # This would configure the validation engine rules
        # For now, just log the configuration
        self.logger.info(f"Configured validation level: {validation_level}")
        
        # In the future, this could modify validation_engine.rules based on level
        if validation_level == "strict":
            # Enable all validation rules with strict thresholds
            pass
        elif validation_level == "lenient":
            # Relax some validation rules
            pass
        # "standard" uses default configuration
    
    def _format_workflow_output(self, results: Dict[str, StandardDataset], 
                              output_format: str) -> Dict[str, Any]:
        """
        Format workflow results according to requested output format.
        
        Args:
            results: Dict of processed datasets
            output_format: Requested output format
            
        Returns:
            Formatted results dict
        """
        if output_format == "standard":
            # Return StandardDataset objects as-is
            return results
        elif output_format == "summary":
            # Return summary information only
            summary = {}
            for name, dataset in results.items():
                summary[name] = {
                    "data_points": len(dataset.z),
                    "redshift_range": [float(dataset.z.min()), float(dataset.z.max())],
                    "has_covariance": dataset.covariance is not None,
                    "metadata": dataset.metadata
                }
            return summary
        elif output_format == "legacy":
            # Convert to legacy DatasetDict format for compatibility
            legacy_results = {}
            for name, dataset in results.items():
                legacy_results[name] = {
                    "observations": dataset.observable,
                    "uncertainties": dataset.uncertainty,
                    "covariance": dataset.covariance,
                    "redshifts": dataset.z,
                    "metadata": dataset.metadata
                }
            return legacy_results
        else:
            self.logger.warning(f"Unknown output format '{output_format}', using standard")
            return results
    
    def validate_processing_determinism(self, dataset_name: str, iterations: int = 3) -> Dict[str, Any]:
        """
        Validate that processing produces deterministic results.
        
        Args:
            dataset_name: Name of dataset to test
            iterations: Number of processing iterations to compare
            
        Returns:
            Dict containing determinism validation results
        """
        self.logger.info(f"Validating processing determinism for {dataset_name} over {iterations} iterations")
        
        results = []
        checksums = []
        
        for i in range(iterations):
            self.logger.info(f"Processing iteration {i+1}/{iterations}")
            
            # Clear cache to force reprocessing
            self.clear_processing_cache()
            
            # Process dataset
            try:
                dataset = self.prepare_dataset(dataset_name, force_reprocess=True)
                
                # Calculate checksum of result
                result_data = {
                    "z": dataset.z.tolist(),
                    "observable": dataset.observable.tolist(),
                    "uncertainty": dataset.uncertainty.tolist(),
                    "covariance": dataset.covariance.tolist() if dataset.covariance is not None else None
                }
                
                result_string = json.dumps(result_data, sort_keys=True)
                checksum = hashlib.sha256(result_string.encode()).hexdigest()
                
                results.append(dataset)
                checksums.append(checksum)
                
            except ProcessingError as e:
                return {
                    "deterministic": False,
                    "error": f"Processing failed on iteration {i+1}: {e}",
                    "iterations_completed": i
                }
        
        # Check if all checksums are identical
        unique_checksums = set(checksums)
        is_deterministic = len(unique_checksums) == 1
        
        validation_result = {
            "deterministic": is_deterministic,
            "iterations": iterations,
            "unique_results": len(unique_checksums),
            "checksums": checksums,
            "environment_hash": self._get_environment_hash()
        }
        
        if is_deterministic:
            self.logger.info(f"Processing is deterministic - all {iterations} iterations produced identical results")
        else:
            self.logger.error(f"Processing is NOT deterministic - found {len(unique_checksums)} different results")
            validation_result["checksum_differences"] = list(unique_checksums)
        
        return validation_result
    
    def generate_processing_report(self, dataset_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive processing report.
        
        Args:
            dataset_names: Optional list of specific datasets to report on
            
        Returns:
            Dict containing comprehensive processing report
        """
        report_time = datetime.now(timezone.utc)
        
        self.logger.info("Generating processing report")
        
        # Get framework statistics
        framework_stats = self.get_processing_statistics()
        framework_setup = self.validate_framework_setup()
        
        # Get dataset information
        dataset_info = {}
        if self.registry_integration:
            try:
                available_datasets = self.registry_integration.list_available_datasets()
                
                if dataset_names:
                    # Filter to requested datasets
                    available_datasets = [d for d in available_datasets if d["name"] in dataset_names]
                
                for dataset in available_datasets:
                    dataset_info[dataset["name"]] = {
                        "registry_status": {
                            "exists": dataset["exists"],
                            "verified": dataset["verified"],
                            "accessible": dataset["accessible"],
                            "ready_for_processing": dataset["ready_for_processing"]
                        },
                        "dataset_type": dataset["dataset_type"],
                        "errors": dataset["errors"],
                        "warnings": dataset["warnings"]
                    }
                    
                    # Add processing requirements if available
                    if dataset["ready_for_processing"]:
                        try:
                            requirements = self.registry_integration.get_dataset_processing_requirements(dataset["name"])
                            dataset_info[dataset["name"]]["processing_requirements"] = requirements
                        except Exception as e:
                            dataset_info[dataset["name"]]["processing_requirements_error"] = str(e)
                            
            except Exception as e:
                dataset_info = {"registry_error": str(e)}
        
        # Compile report
        report = {
            "report_timestamp": report_time.isoformat(),
            "framework_statistics": framework_stats,
            "framework_setup": framework_setup,
            "dataset_information": dataset_info,
            "environment_hash": self._get_environment_hash(),
            "output_directory": str(self.output_directory)
        }
        
        # Save report to file
        report_filename = f"processing_report_{report_time.strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.output_directory / report_filename
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, sort_keys=True)
            
            report["report_file"] = str(report_path)
            self.logger.info(f"Processing report saved to: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not save processing report: {e}")
        
        return report
    
    def _create_enhanced_module_wrapper(self, module: DerivationModule, transformation_logger: TransformationLogger):
        """Create enhanced module wrapper with transformation logging."""
        
        class EnhancedDerivationModule:
            """Wrapper for derivation modules with enhanced logging."""
            
            def __init__(self, wrapped_module: DerivationModule, logger: TransformationLogger):
                self.wrapped_module = wrapped_module
                self.logger = logger
            
            def __getattr__(self, name):
                """Delegate attribute access to wrapped module."""
                return getattr(self.wrapped_module, name)
            
            def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
                """Enhanced derive method with detailed logging."""
                with self.logger.log_transformation_step(
                    step_name=f"{self.wrapped_module.dataset_type}_derivation",
                    description=f"Complete {self.wrapped_module.dataset_type.upper()} dataset derivation",
                    input_description=f"Raw {self.wrapped_module.dataset_type.upper()} data file",
                    output_description="Standardized dataset format"
                ) as step:
                    
                    # Log input file information
                    if raw_data_path and raw_data_path.exists():
                        file_size = raw_data_path.stat().st_size
                        self.logger.log_parameter("input_file_size", file_size, "Size of input data file in bytes")
                        self.logger.log_parameter("input_file_format", raw_data_path.suffix, "Input file format/extension")
                    
                    # Log metadata information
                    if metadata:
                        self.logger.log_parameter("metadata_keys", list(metadata.keys()), "Available metadata fields")
                        if "dataset_type" in metadata:
                            self.logger.log_parameter("dataset_type", metadata["dataset_type"], "Dataset type from metadata")
                    
                    # Call the actual derivation method
                    result = self.wrapped_module.derive(raw_data_path, metadata)
                    
                    # Log output information
                    self.logger.log_data_shape(result, "output_dataset")
                    
                    # Log transformation summary
                    try:
                        transformation_summary = self.wrapped_module.get_transformation_summary()
                        if transformation_summary:
                            for key, value in transformation_summary.items():
                                self.logger.log_parameter(f"summary_{key}", value, f"Transformation summary: {key}")
                    except:
                        pass
                    
                    return result
        
        return EnhancedDerivationModule(module, transformation_logger)
    
    def _calculate_dataset_hash(self, dataset: StandardDataset) -> str:
        """Calculate hash of derived dataset for provenance tracking."""
        import hashlib
        
        # Create deterministic representation of dataset
        hash_data = {
            "z": dataset.z.tolist() if dataset.z is not None else None,
            "observable": dataset.observable.tolist() if dataset.observable is not None else None,
            "uncertainty": dataset.uncertainty.tolist() if dataset.uncertainty is not None else None,
            "covariance": dataset.covariance.tolist() if dataset.covariance is not None else None,
            "metadata": dataset.metadata
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def _log_processing_failure(self, dataset_name: str, error, start_time: datetime):
        """Log detailed information about processing failures with enhanced context."""
        failure_time = datetime.now(timezone.utc)
        duration = (failure_time - start_time).total_seconds()
        
        if isinstance(error, EnhancedProcessingError):
            failure_log = {
                "dataset_name": dataset_name,
                "failure_time": failure_time.isoformat(),
                "processing_duration": duration,
                "error_context": error.error_context.to_dict(),
                "system_health": self.system_health_monitor.record_health_snapshot(),
                "recovery_attempted": error.error_context.recovery_attempted,
                "recovery_successful": error.error_context.recovery_successful
            }
        else:
            # Handle regular ProcessingError
            failure_log = {
                "dataset_name": dataset_name,
                "failure_time": failure_time.isoformat(),
                "processing_duration": duration,
                "error_type": error.error_type,
                "error_message": error.error_message,
                "stage": error.stage,
                "context": error.context,
                "system_health": self.system_health_monitor.record_health_snapshot(),
                "recovery_attempted": False,
                "recovery_successful": False
            }
        
        self.logger.error(f"Enhanced processing failure details: {json.dumps(failure_log, indent=2, default=str)}")
        
        # Save detailed failure log to file
        failure_log_path = self.output_directory / "processing_failures.jsonl"
        try:
            with open(failure_log_path, 'a') as f:
                f.write(json.dumps(failure_log, default=str) + '\n')
        except Exception as e:
            self.logger.warning(f"Could not write enhanced failure log: {e}")
        
        # Save detailed error report
        try:
            error_report_path = self.output_directory.parent / "logs" / f"error_report_{dataset_name}_{failure_time.strftime('%Y%m%d_%H%M%S')}.txt"
            error_report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(error_report_path, 'w') as f:
                f.write(error.generate_detailed_report())
            
            self.logger.info(f"Detailed error report saved to: {error_report_path}")
        except Exception as e:
            self.logger.warning(f"Could not save detailed error report: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics and performance metrics."""
        return {
            "processing_logger_stats": self.processing_logger.get_performance_report(),
            "error_recovery_stats": self.error_recovery_manager.get_recovery_stats(),
            "system_health_summary": self.system_health_monitor.get_health_summary(),
            "cache_statistics": {
                "cache_entries": len(self._processing_cache),
                "cache_directory": str(self.output_directory)
            }
        }
    
    def save_comprehensive_audit_trail(self, filename: Optional[str] = None) -> Path:
        """Save comprehensive audit trail including all logging systems."""
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_audit_trail_{timestamp}.json"
        
        audit_path = self.output_directory.parent / "logs" / filename
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        
        comprehensive_audit = {
            "framework_info": {
                "version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output_directory": str(self.output_directory)
            },
            "processing_statistics": self.get_processing_statistics(),
            "registered_modules": list(self.derivation_modules.keys()),
            "environment_hash": self._get_environment_hash()
        }
        
        with open(audit_path, 'w') as f:
            json.dump(comprehensive_audit, f, indent=2, default=str)
        
        # Also save the processing logger's audit trail
        self.processing_logger.save_audit_trail(f"processing_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
        
        self.logger.info(f"Comprehensive audit trail saved to: {audit_path}")
    
    def _calculate_environment_hash(self) -> str:
        """
        Calculate hash of current environment for deterministic processing.
        
        Returns:
            Environment hash string
        """
        if self._environment_hash is None:
            import platform
            import sys
            
            env_info = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "hostname": platform.node(),
                "pbuf_commit": "unknown",  # Could be enhanced to get actual commit
                "timestamp": datetime.now(timezone.utc).date().isoformat()  # Date only for daily consistency
            }
            
            env_string = json.dumps(env_info, sort_keys=True)
            self._environment_hash = hashlib.sha256(env_string.encode()).hexdigest()[:16]
        
        return self._environment_hash
    
    def _calculate_provenance_hash(self, provenance_record) -> str:
        """
        Calculate hash of provenance record for tracking.
        
        Args:
            provenance_record: ProvenanceRecord instance
            
        Returns:
            Provenance hash string
        """
        provenance_data = {
            "dataset_name": provenance_record.dataset_name,
            "download_timestamp": provenance_record.download_timestamp,
            "source_used": provenance_record.source_used,
            "sha256_hash": provenance_record.verification.sha256_actual
        }
        
        provenance_string = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_string.encode()).hexdigest()[:16]
    
    def _calculate_dataset_hash(self, dataset: StandardDataset) -> str:
        """
        Calculate hash of derived dataset for caching and verification.
        
        Args:
            dataset: StandardDataset instance
            
        Returns:
            Dataset hash string
        """
        # Create deterministic representation of dataset
        dataset_data = {
            "z": dataset.z.tolist(),
            "observable": dataset.observable.tolist(),
            "uncertainty": dataset.uncertainty.tolist(),
            "covariance": dataset.covariance.tolist() if dataset.covariance is not None else None,
            "metadata_keys": sorted(dataset.metadata.keys())
        }
        
        dataset_string = json.dumps(dataset_data, sort_keys=True)
        return hashlib.sha256(dataset_string.encode()).hexdigest()[:16]
        return audit_path