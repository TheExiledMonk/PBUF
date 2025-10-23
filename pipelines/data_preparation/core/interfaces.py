"""
Abstract base classes and core interfaces for derivation modules.

This module defines the plugin-like architecture that allows new dataset types
to be added without modifying core system logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
from .schema import StandardDataset


class DerivationModule(ABC):
    """
    Abstract base class for dataset-specific derivation modules.
    
    Each dataset type (SN, BAO, CMB, CC, RSD) implements this interface
    to provide standardized transformation logic while maintaining
    dataset-specific processing requirements.
    """
    
    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """
        Return the dataset type identifier (e.g., 'sn', 'bao', 'cmb').
        
        Returns:
            str: Dataset type identifier
        """
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """
        Return list of supported input file formats.
        
        Returns:
            List[str]: Supported file extensions (e.g., ['.txt', '.csv', '.fits'])
        """
        pass
    
    @abstractmethod
    def validate_input(self, raw_data_path: Path, metadata: Dict[str, Any]) -> bool:
        """
        Validate raw data before processing.
        
        Args:
            raw_data_path: Path to raw dataset file
            metadata: Dataset metadata from registry
            
        Returns:
            bool: True if input is valid for processing
            
        Raises:
            ValueError: If input validation fails with detailed error message
        """
        pass
    
    @abstractmethod
    def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
        """
        Transform raw data to standardized format.
        
        This is the core transformation method that converts dataset-specific
        raw data into the unified StandardDataset format.
        
        Args:
            raw_data_path: Path to verified raw dataset file
            metadata: Dataset metadata from registry
            
        Returns:
            StandardDataset: Transformed data in standard format
            
        Raises:
            ValueError: If transformation fails
            FileNotFoundError: If raw data file not found
        """
        pass
    
    @abstractmethod
    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Return summary of applied transformations.
        
        This method provides detailed information about the transformations
        applied during the derive() method, suitable for provenance tracking
        and publication materials.
        
        Returns:
            Dict containing:
                - transformation_steps: List of applied transformation descriptions
                - formulas_used: Mathematical formulas applied
                - assumptions: Physical assumptions made
                - references: Literature references for methods
        """
        pass
    
    def get_expected_output_schema(self) -> Dict[str, str]:
        """
        Return expected output schema for this dataset type.
        
        Returns:
            Dict mapping field names to their physical meanings
        """
        return {
            'z': 'Redshift',
            'observable': f'{self.dataset_type.upper()} observable',
            'uncertainty': 'One-sigma uncertainty',
            'covariance': 'Full covariance matrix (optional)',
            'metadata': 'Processing and source metadata'
        }


class ValidationRule(ABC):
    """
    Abstract base class for validation rules.
    
    Allows extensible validation by implementing specific validation
    logic as separate rule classes.
    """
    
    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Return human-readable name for this validation rule."""
        pass
    
    @abstractmethod
    def validate(self, dataset: StandardDataset) -> bool:
        """
        Apply validation rule to dataset.
        
        Args:
            dataset: StandardDataset to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails with detailed error message
        """
        pass


class ProcessingError(Exception):
    """
    Comprehensive error reporting for data preparation failures.
    
    This exception class provides detailed context about processing failures
    to enable effective debugging and user guidance.
    """
    
    def __init__(
        self,
        dataset_name: str,
        stage: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any] = None,
        suggested_actions: List[str] = None
    ):
        """
        Initialize processing error.
        
        Args:
            dataset_name: Name of dataset being processed
            stage: Processing stage where error occurred
            error_type: Type of error (e.g., 'validation_error', 'transformation_error')
            error_message: Detailed error description
            context: Additional context information
            suggested_actions: List of suggested remediation actions
        """
        self.dataset_name = dataset_name
        self.stage = stage  # "input_validation", "transformation", "output_validation"
        self.error_type = error_type
        self.error_message = error_message
        self.context = context or {}
        self.suggested_actions = suggested_actions or []
        
        super().__init__(self.generate_message())
    
    def generate_message(self) -> str:
        """Generate comprehensive error message."""
        message = f"Processing failed for dataset '{self.dataset_name}'\n"
        message += f"Stage: {self.stage}\n"
        message += f"Error Type: {self.error_type}\n"
        message += f"Error: {self.error_message}\n"
        
        if self.context:
            message += f"Context: {self.context}\n"
        
        if self.suggested_actions:
            message += "Suggested Actions:\n"
            for i, action in enumerate(self.suggested_actions, 1):
                message += f"  {i}. {action}\n"
        
        return message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'dataset_name': self.dataset_name,
            'stage': self.stage,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'context': self.context,
            'suggested_actions': self.suggested_actions
        }


@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for derived datasets.
    
    Tracks the complete lineage from raw data to derived dataset,
    enabling full reproducibility and traceability.
    """
    source_dataset_name: str
    source_hash: str
    derivation_module: str
    processing_timestamp: str
    environment_hash: str
    transformation_steps: List[str]
    validation_results: Dict[str, Any]
    derived_dataset_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance record to dictionary for serialization."""
        return {
            'source_dataset_name': self.source_dataset_name,
            'source_hash': self.source_hash,
            'derivation_module': self.derivation_module,
            'processing_timestamp': self.processing_timestamp,
            'environment_hash': self.environment_hash,
            'transformation_steps': self.transformation_steps,
            'validation_results': self.validation_results,
            'derived_dataset_hash': self.derived_dataset_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProvenanceRecord':
        """Create ProvenanceRecord from dictionary."""
        return cls(
            source_dataset_name=data['source_dataset_name'],
            source_hash=data['source_hash'],
            derivation_module=data['derivation_module'],
            processing_timestamp=data['processing_timestamp'],
            environment_hash=data['environment_hash'],
            transformation_steps=data['transformation_steps'],
            validation_results=data['validation_results'],
            derived_dataset_hash=data['derived_dataset_hash']
        )