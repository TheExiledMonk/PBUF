"""
Exception hierarchy for CMB raw parameter processing.

This module defines specialized exceptions for different types of errors that can
occur during CMB parameter detection, validation, derivation, and covariance processing.
"""

from typing import Dict, Any, List, Optional


class CMBProcessingError(Exception):
    """
    Base exception for CMB processing errors.
    
    Provides structured error reporting with context information and suggested
    remediation actions for debugging and user guidance.
    
    Attributes:
        message: Primary error message
        context: Additional context information
        suggested_actions: List of suggested remediation steps
        error_code: Unique error code for programmatic handling
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggested_actions: Optional[List[str]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize CMB processing error.
        
        Args:
            message: Primary error description
            context: Additional context information (file paths, parameter values, etc.)
            suggested_actions: List of suggested remediation actions
            error_code: Unique error code for programmatic handling
        """
        self.message = message
        self.context = context or {}
        self.suggested_actions = suggested_actions or []
        self.error_code = error_code or self.__class__.__name__
        
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format comprehensive error message."""
        formatted = f"CMB Processing Error: {self.message}"
        
        if self.context:
            formatted += f"\nContext: {self.context}"
        
        if self.suggested_actions:
            formatted += "\nSuggested Actions:"
            for i, action in enumerate(self.suggested_actions, 1):
                formatted += f"\n  {i}. {action}"
        
        if self.error_code:
            formatted += f"\nError Code: {self.error_code}"
        
        return formatted
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'context': self.context,
            'suggested_actions': self.suggested_actions,
            'error_code': self.error_code
        }


class ParameterDetectionError(CMBProcessingError):
    """
    Raised when raw cosmological parameters cannot be detected or parsed.
    
    This error occurs during the parameter detection phase when the system
    cannot find or interpret raw parameter files in the dataset registry.
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        available_files: Optional[List[str]] = None,
        expected_parameters: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize parameter detection error.
        
        Args:
            message: Primary error description
            file_path: Path to the problematic file
            available_files: List of available files in registry entry
            expected_parameters: List of expected parameter names
            **kwargs: Additional arguments passed to base class
        """
        context = kwargs.get('context', {})
        
        if file_path:
            context['file_path'] = file_path
        if available_files:
            context['available_files'] = available_files
        if expected_parameters:
            context['expected_parameters'] = expected_parameters
        
        # Default suggested actions for parameter detection errors
        default_actions = [
            "Check that the dataset registry entry contains raw parameter files",
            "Verify file format is supported (CSV, JSON, or NumPy)",
            "Ensure parameter names match expected conventions (H0, Omega_m, etc.)",
            "Check file permissions and accessibility"
        ]
        
        suggested_actions = kwargs.get('suggested_actions', default_actions)
        
        super().__init__(
            message=message,
            context=context,
            suggested_actions=suggested_actions,
            error_code="CMB_PARAM_DETECTION_ERROR",
            **{k: v for k, v in kwargs.items() if k not in ['context', 'suggested_actions']}
        )


class ParameterValidationError(CMBProcessingError):
    """
    Raised when cosmological parameters fail validation checks.
    
    This error occurs when parameter values are outside physical bounds,
    contain invalid values (NaN, inf), or fail consistency checks.
    """
    
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[float] = None,
        valid_range: Optional[tuple] = None,
        validation_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize parameter validation error.
        
        Args:
            message: Primary error description
            parameter_name: Name of the invalid parameter
            parameter_value: Value that failed validation
            valid_range: Expected valid range (min, max)
            validation_type: Type of validation that failed
            **kwargs: Additional arguments passed to base class
        """
        context = kwargs.get('context', {})
        
        if parameter_name:
            context['parameter_name'] = parameter_name
        if parameter_value is not None:
            context['parameter_value'] = parameter_value
        if valid_range:
            context['valid_range'] = valid_range
        if validation_type:
            context['validation_type'] = validation_type
        
        # Default suggested actions for parameter validation errors
        default_actions = [
            "Check parameter values against Planck 2018 constraints",
            "Verify units are correct (H0 in km/s/Mpc, etc.)",
            "Remove or correct NaN and infinite values",
            "Check for data entry errors or corrupted files"
        ]
        
        if parameter_name and valid_range:
            default_actions.insert(0, 
                f"Ensure {parameter_name} is within valid range {valid_range}"
            )
        
        suggested_actions = kwargs.get('suggested_actions', default_actions)
        
        super().__init__(
            message=message,
            context=context,
            suggested_actions=suggested_actions,
            error_code="CMB_PARAM_VALIDATION_ERROR",
            **{k: v for k, v in kwargs.items() if k not in ['context', 'suggested_actions']}
        )


class DerivationError(CMBProcessingError):
    """
    Raised when distance prior computation fails.
    
    This error occurs during the derivation phase when computing CMB distance
    priors (R, l_A, θ*) from raw cosmological parameters fails due to
    numerical issues or integration problems.
    """
    
    def __init__(
        self,
        message: str,
        computation_stage: Optional[str] = None,
        parameter_values: Optional[Dict[str, float]] = None,
        integration_error: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize derivation error.
        
        Args:
            message: Primary error description
            computation_stage: Stage where computation failed (R, l_A, theta_star)
            parameter_values: Parameter values used in computation
            integration_error: Specific integration error message
            **kwargs: Additional arguments passed to base class
        """
        context = kwargs.get('context', {})
        
        if computation_stage:
            context['computation_stage'] = computation_stage
        if parameter_values:
            context['parameter_values'] = parameter_values
        if integration_error:
            context['integration_error'] = integration_error
        
        # Default suggested actions for derivation errors
        default_actions = [
            "Check that PBUF background integrators are properly initialized",
            "Verify cosmological parameters are within integration domain",
            "Check for numerical instabilities in background evolution",
            "Ensure recombination redshift is reasonable (z* ≈ 1090)"
        ]
        
        if computation_stage:
            stage_actions = {
                'R': "Check matter density and Hubble parameter values",
                'l_A': "Verify sound horizon and angular diameter distance calculations",
                'theta_star': "Check sound horizon and angular diameter distance ratio"
            }
            if computation_stage in stage_actions:
                default_actions.insert(0, stage_actions[computation_stage])
        
        suggested_actions = kwargs.get('suggested_actions', default_actions)
        
        super().__init__(
            message=message,
            context=context,
            suggested_actions=suggested_actions,
            error_code="CMB_DERIVATION_ERROR",
            **{k: v for k, v in kwargs.items() if k not in ['context', 'suggested_actions']}
        )


class CovarianceError(CMBProcessingError):
    """
    Raised when covariance matrix processing fails.
    
    This error occurs during covariance matrix validation, Jacobian computation,
    or uncertainty propagation when matrix properties are invalid or numerical
    computations fail.
    """
    
    def __init__(
        self,
        message: str,
        matrix_property: Optional[str] = None,
        matrix_shape: Optional[tuple] = None,
        eigenvalue_info: Optional[Dict[str, float]] = None,
        jacobian_stage: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize covariance error.
        
        Args:
            message: Primary error description
            matrix_property: Property that failed (symmetry, positive_definiteness, etc.)
            matrix_shape: Shape of the problematic matrix
            eigenvalue_info: Information about eigenvalues (min, max, condition_number)
            jacobian_stage: Stage of Jacobian computation that failed
            **kwargs: Additional arguments passed to base class
        """
        context = kwargs.get('context', {})
        
        if matrix_property:
            context['matrix_property'] = matrix_property
        if matrix_shape:
            context['matrix_shape'] = matrix_shape
        if eigenvalue_info:
            context['eigenvalue_info'] = eigenvalue_info
        if jacobian_stage:
            context['jacobian_stage'] = jacobian_stage
        
        # Default suggested actions for covariance errors
        default_actions = [
            "Check covariance matrix file format and dimensions",
            "Verify matrix is symmetric and positive-definite",
            "Check for numerical precision issues in matrix elements",
            "Ensure parameter uncertainties are reasonable"
        ]
        
        if matrix_property:
            property_actions = {
                'symmetry': "Ensure covariance matrix is exactly symmetric",
                'positive_definiteness': "Check for negative eigenvalues or singular matrix",
                'dimensions': "Verify matrix dimensions match number of parameters",
                'numerical_stability': "Check condition number and eigenvalue spread"
            }
            if matrix_property in property_actions:
                default_actions.insert(0, property_actions[matrix_property])
        
        suggested_actions = kwargs.get('suggested_actions', default_actions)
        
        super().__init__(
            message=message,
            context=context,
            suggested_actions=suggested_actions,
            error_code="CMB_COVARIANCE_ERROR",
            **{k: v for k, v in kwargs.items() if k not in ['context', 'suggested_actions']}
        )


class NumericalInstabilityError(CMBProcessingError):
    """
    Raised when numerical computations become unstable.
    
    This error occurs when numerical differentiation, matrix operations,
    or iterative computations fail to converge or produce stable results.
    """
    
    def __init__(
        self,
        message: str,
        computation_type: Optional[str] = None,
        step_size: Optional[float] = None,
        convergence_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize numerical instability error.
        
        Args:
            message: Primary error description
            computation_type: Type of computation that became unstable
            step_size: Step size used in numerical computation
            convergence_info: Information about convergence failure
            **kwargs: Additional arguments passed to base class
        """
        context = kwargs.get('context', {})
        
        if computation_type:
            context['computation_type'] = computation_type
        if step_size is not None:
            context['step_size'] = step_size
        if convergence_info:
            context['convergence_info'] = convergence_info
        
        # Default suggested actions for numerical instability errors
        default_actions = [
            "Reduce numerical differentiation step size",
            "Check parameter values for extreme or boundary cases",
            "Increase numerical precision or use alternative algorithms",
            "Verify input data quality and remove outliers"
        ]
        
        if computation_type:
            type_actions = {
                'jacobian': "Adjust jacobian_step_size in configuration",
                'integration': "Check integration tolerances and limits",
                'matrix_inversion': "Check matrix condition number and regularization",
                'eigenvalue': "Use more stable eigenvalue decomposition method"
            }
            if computation_type in type_actions:
                default_actions.insert(0, type_actions[computation_type])
        
        suggested_actions = kwargs.get('suggested_actions', default_actions)
        
        super().__init__(
            message=message,
            context=context,
            suggested_actions=suggested_actions,
            error_code="CMB_NUMERICAL_INSTABILITY_ERROR",
            **{k: v for k, v in kwargs.items() if k not in ['context', 'suggested_actions']}
        )


# Utility functions for error handling
def create_parameter_detection_error(
    file_path: str,
    missing_parameters: List[str],
    available_keys: List[str]
) -> ParameterDetectionError:
    """
    Create a standardized parameter detection error.
    
    Args:
        file_path: Path to the problematic parameter file
        missing_parameters: List of missing required parameters
        available_keys: List of available parameter names in file
        
    Returns:
        ParameterDetectionError with detailed context
    """
    message = f"Required CMB parameters not found in {file_path}"
    
    return ParameterDetectionError(
        message=message,
        file_path=file_path,
        context={
            'missing_parameters': missing_parameters,
            'available_keys': available_keys,
            'n_missing': len(missing_parameters),
            'n_available': len(available_keys)
        },
        suggested_actions=[
            f"Add missing parameters: {', '.join(missing_parameters)}",
            "Check parameter name conventions (case-sensitive matching)",
            "Verify file format and structure",
            f"Available parameters: {', '.join(available_keys[:10])}"  # Show first 10
        ]
    )


def create_parameter_validation_error(
    parameter_name: str,
    value: float,
    valid_range: tuple,
    validation_type: str = "range_check"
) -> ParameterValidationError:
    """
    Create a standardized parameter validation error.
    
    Args:
        parameter_name: Name of the invalid parameter
        value: Invalid parameter value
        valid_range: Expected valid range (min, max)
        validation_type: Type of validation that failed
        
    Returns:
        ParameterValidationError with detailed context
    """
    min_val, max_val = valid_range
    message = f"Parameter {parameter_name} = {value} outside valid range [{min_val}, {max_val}]"
    
    return ParameterValidationError(
        message=message,
        parameter_name=parameter_name,
        parameter_value=value,
        valid_range=valid_range,
        validation_type=validation_type,
        suggested_actions=[
            f"Ensure {parameter_name} is between {min_val} and {max_val}",
            "Check units and parameter definitions",
            "Verify data source and processing pipeline",
            "Compare with published Planck 2018 constraints"
        ]
    )


def create_covariance_error(
    matrix_shape: tuple,
    property_name: str,
    diagnostic_info: Dict[str, Any]
) -> CovarianceError:
    """
    Create a standardized covariance matrix error.
    
    Args:
        matrix_shape: Shape of the problematic matrix
        property_name: Matrix property that failed validation
        diagnostic_info: Diagnostic information about the failure
        
    Returns:
        CovarianceError with detailed context
    """
    message = f"Covariance matrix {property_name} validation failed"
    
    return CovarianceError(
        message=message,
        matrix_property=property_name,
        matrix_shape=matrix_shape,
        context=diagnostic_info,
        suggested_actions=[
            f"Fix covariance matrix {property_name} issue",
            "Check matrix construction and file format",
            "Verify numerical precision and data quality",
            "Consider matrix regularization if appropriate"
        ]
    )