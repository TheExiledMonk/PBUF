"""
Dataset verification engine with comprehensive validation

This module provides cryptographic verification, file size validation,
and schema structure checking for datasets.
"""

import hashlib
import json
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from enum import Enum

from ..core.logging_integration import log_registry_operation


class VerificationStatus(Enum):
    """Verification status enumeration"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class VerificationError(Exception):
    """Base class for verification errors"""
    pass


class ChecksumError(VerificationError):
    """Raised when checksum verification fails"""
    pass


class SizeError(VerificationError):
    """Raised when file size validation fails"""
    pass


class SchemaError(VerificationError):
    """Raised when schema validation fails"""
    pass


@dataclass
class VerificationResult:
    """
    Comprehensive verification result with detailed reporting
    
    Contains results of all verification checks including checksums,
    file size, and schema validation with error details and suggestions.
    """
    dataset_name: str
    file_path: Path
    verification_time: datetime
    
    # Overall status
    is_valid: bool
    status: VerificationStatus
    
    # Checksum verification
    sha256_match: bool
    sha256_expected: Optional[str]
    sha256_actual: Optional[str]
    
    # Size verification
    size_match: bool
    size_expected: Optional[int]
    size_actual: Optional[int]
    
    # Schema verification
    schema_valid: bool
    schema_errors: List[str]
    schema_warnings: List[str]
    
    # Error reporting
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    # Default values
    size_tolerance_bytes: int = 0
    
    def __post_init__(self):
        """Validate result consistency after initialization"""
        if self.is_valid and self.status == VerificationStatus.FAILED:
            raise ValueError("Inconsistent verification result: is_valid=True but status=FAILED")
        
        if not self.is_valid and self.status == VerificationStatus.SUCCESS:
            raise ValueError("Inconsistent verification result: is_valid=False but status=SUCCESS")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "dataset_name": self.dataset_name,
            "file_path": str(self.file_path),
            "verification_time": self.verification_time.isoformat(),
            "is_valid": self.is_valid,
            "status": self.status.value,
            "sha256_match": self.sha256_match,
            "sha256_expected": self.sha256_expected,
            "sha256_actual": self.sha256_actual,
            "size_match": self.size_match,
            "size_expected": self.size_expected,
            "size_actual": self.size_actual,
            "size_tolerance_bytes": self.size_tolerance_bytes,
            "schema_valid": self.schema_valid,
            "schema_errors": self.schema_errors,
            "schema_warnings": self.schema_warnings,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Create result from dictionary"""
        return cls(
            dataset_name=data["dataset_name"],
            file_path=Path(data["file_path"]),
            verification_time=datetime.fromisoformat(data["verification_time"]),
            is_valid=data["is_valid"],
            status=VerificationStatus(data["status"]),
            sha256_match=data["sha256_match"],
            sha256_expected=data.get("sha256_expected"),
            sha256_actual=data.get("sha256_actual"),
            size_match=data["size_match"],
            size_expected=data.get("size_expected"),
            size_actual=data.get("size_actual"),
            size_tolerance_bytes=data.get("size_tolerance_bytes", 0),
            schema_valid=data["schema_valid"],
            schema_errors=data.get("schema_errors", []),
            schema_warnings=data.get("schema_warnings", []),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            suggestions=data.get("suggestions", [])
        )


class VerificationEngine:
    """
    Comprehensive dataset verification engine
    
    Provides SHA256 checksum verification, file size validation with tolerance,
    and schema structure checking for different dataset formats.
    """
    
    def __init__(self, size_tolerance_percent: float = 5.0):
        """
        Initialize verification engine
        
        Args:
            size_tolerance_percent: Allowed size deviation percentage (default: 5%)
        """
        self.size_tolerance_percent = size_tolerance_percent
    
    def calculate_sha256(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Calculate SHA256 checksum of a file
        
        Args:
            file_path: Path to file
            chunk_size: Size of chunks to read (default: 8192 bytes)
            
        Returns:
            Hexadecimal SHA256 checksum string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files efficiently
                while chunk := f.read(chunk_size):
                    sha256_hash.update(chunk)
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {file_path}")
        
        return sha256_hash.hexdigest()
    
    def verify_checksum(self, file_path: Path, expected_sha256: str) -> Tuple[bool, str]:
        """
        Verify file SHA256 checksum
        
        Args:
            file_path: Path to file to verify
            expected_sha256: Expected SHA256 checksum (hex string)
            
        Returns:
            Tuple of (match_result, actual_checksum)
            
        Raises:
            ChecksumError: If checksum calculation fails
        """
        try:
            actual_sha256 = self.calculate_sha256(file_path)
        except (FileNotFoundError, PermissionError) as e:
            raise ChecksumError(f"Failed to calculate checksum: {e}")
        
        # Normalize checksums for comparison (lowercase, no spaces)
        expected_normalized = expected_sha256.lower().replace(' ', '')
        actual_normalized = actual_sha256.lower().replace(' ', '')
        
        return expected_normalized == actual_normalized, actual_sha256
    
    def verify_file_size(self, file_path: Path, expected_size: Optional[int] = None) -> Tuple[bool, int, int]:
        """
        Verify file size with tolerance checking
        
        Args:
            file_path: Path to file to verify
            expected_size: Expected file size in bytes (None to skip check)
            
        Returns:
            Tuple of (size_match, actual_size, tolerance_bytes)
            
        Raises:
            SizeError: If size check fails due to file access issues
        """
        try:
            actual_size = file_path.stat().st_size
        except (FileNotFoundError, PermissionError) as e:
            raise SizeError(f"Failed to get file size: {e}")
        
        if expected_size is None:
            return True, actual_size, 0
        
        # Calculate tolerance
        tolerance_bytes = int(expected_size * self.size_tolerance_percent / 100)
        size_diff = abs(actual_size - expected_size)
        
        size_match = size_diff <= tolerance_bytes
        
        return size_match, actual_size, tolerance_bytes
    
    def verify_dataset(self, 
                      dataset_name: str,
                      file_path: Path, 
                      verification_config: Dict[str, Any]) -> VerificationResult:
        """
        Perform comprehensive dataset verification
        
        Args:
            dataset_name: Name of the dataset being verified
            file_path: Path to dataset file
            verification_config: Verification configuration from manifest
            
        Returns:
            VerificationResult with detailed verification status
        """
        verification_time = datetime.now()
        start_time = time.time()
        
        # Log verification start
        log_registry_operation(
            "verification",
            dataset_name,
            status="started",
            metadata={
                "file_path": str(file_path),
                "has_checksum": "sha256" in verification_config,
                "has_size_check": "size_bytes" in verification_config,
                "has_schema": "schema" in verification_config
            }
        )
        
        # Initialize result with defaults
        result = VerificationResult(
            dataset_name=dataset_name,
            file_path=file_path,
            verification_time=verification_time,
            is_valid=True,
            status=VerificationStatus.PENDING,
            sha256_match=False,
            sha256_expected=None,
            sha256_actual=None,
            size_match=False,
            size_expected=None,
            size_actual=None,
            size_tolerance_bytes=0,
            schema_valid=False,
            schema_errors=[],
            schema_warnings=[],
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        # Check if file exists
        if not file_path.exists():
            result.is_valid = False
            result.status = VerificationStatus.FAILED
            result.errors.append(f"File not found: {file_path}")
            result.suggestions.append("Download the dataset or check the file path")
            return result
        
        verification_passed = True
        
        # 1. SHA256 Checksum verification
        expected_sha256 = verification_config.get("sha256")
        if expected_sha256:
            try:
                sha256_match, actual_sha256 = self.verify_checksum(file_path, expected_sha256)
                result.sha256_match = sha256_match
                result.sha256_expected = expected_sha256
                result.sha256_actual = actual_sha256
                
                if not sha256_match:
                    verification_passed = False
                    result.errors.append(f"SHA256 checksum mismatch: expected {expected_sha256}, got {actual_sha256}")
                    result.suggestions.append("Re-download the dataset or verify the expected checksum")
                
            except ChecksumError as e:
                verification_passed = False
                result.errors.append(f"Checksum verification failed: {e}")
                result.suggestions.append("Check file permissions and disk integrity")
        else:
            result.warnings.append("No SHA256 checksum provided for verification")
        
        # 2. File size verification
        expected_size = verification_config.get("size_bytes")
        try:
            size_match, actual_size, tolerance_bytes = self.verify_file_size(file_path, expected_size)
            result.size_match = size_match
            result.size_expected = expected_size
            result.size_actual = actual_size
            result.size_tolerance_bytes = tolerance_bytes
            
            if expected_size and not size_match:
                verification_passed = False
                result.errors.append(f"File size mismatch: expected {expected_size} bytes (±{tolerance_bytes}), got {actual_size} bytes")
                result.suggestions.append("Check if file was completely downloaded or corrupted")
            
        except SizeError as e:
            verification_passed = False
            result.errors.append(f"Size verification failed: {e}")
            result.suggestions.append("Check file permissions and accessibility")
        
        # 3. Schema verification (will be implemented in subtask 3.2)
        schema_config = verification_config.get("schema")
        if schema_config:
            try:
                schema_result = self._verify_schema(file_path, schema_config)
                result.schema_valid = schema_result["valid"]
                result.schema_errors = schema_result["errors"]
                result.schema_warnings = schema_result["warnings"]
                
                if not schema_result["valid"]:
                    verification_passed = False
                    result.errors.extend([f"Schema error: {error}" for error in schema_result["errors"]])
                    result.suggestions.extend(schema_result.get("suggestions", []))
                
            except Exception as e:
                verification_passed = False
                result.errors.append(f"Schema verification failed: {e}")
                result.suggestions.append("Check dataset format and structure")
        else:
            result.schema_valid = True  # No schema to validate
            result.warnings.append("No schema validation configured")
        
        # Set final status
        result.is_valid = verification_passed
        if verification_passed:
            result.status = VerificationStatus.SUCCESS
        elif result.errors:
            result.status = VerificationStatus.FAILED
        else:
            result.status = VerificationStatus.PARTIAL
        
        # Log verification completion
        duration_ms = (time.time() - start_time) * 1000
        status = "success" if verification_passed else "failed"
        
        log_registry_operation(
            "verification",
            dataset_name,
            status=status,
            duration_ms=duration_ms,
            error="; ".join(result.errors) if result.errors else None,
            metadata={
                "sha256_match": result.sha256_match,
                "size_match": result.size_match,
                "schema_valid": result.schema_valid,
                "file_size": result.size_actual,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings)
            }
        )
        
        return result
    
    def _verify_schema(self, file_path: Path, schema_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify dataset schema structure
        
        Args:
            file_path: Path to dataset file
            schema_config: Schema configuration from manifest
            
        Returns:
            Dictionary with validation results
        """
        format_type = schema_config.get("format", "").lower()
        
        if format_type == "ascii_table":
            return self._verify_ascii_table_schema(file_path, schema_config)
        elif format_type == "json":
            return self._verify_json_schema(file_path, schema_config)
        elif format_type == "fits":
            return self._verify_fits_schema(file_path, schema_config)
        else:
            return {
                "valid": False,
                "errors": [f"Unsupported schema format: {format_type}"],
                "warnings": [],
                "suggestions": ["Supported formats: ascii_table, json, fits"]
            }
    
    def _verify_ascii_table_schema(self, file_path: Path, schema_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify ASCII table format schema
        
        Args:
            file_path: Path to ASCII table file
            schema_config: Schema configuration
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        suggestions = []
        
        expected_columns = schema_config.get("columns", [])
        expected_rows = schema_config.get("expected_rows")
        delimiter = schema_config.get("delimiter", None)  # Auto-detect if None
        comment_char = schema_config.get("comment_char", "#")
        header_rows = schema_config.get("header_rows", 0)
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Filter out comment lines and empty lines
            data_lines = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith(comment_char):
                    if i >= header_rows:  # Skip header rows
                        data_lines.append(stripped)
            
            if not data_lines:
                errors.append("No data lines found in file")
                suggestions.append("Check if file contains actual data or adjust comment_char/header_rows settings")
                return {"valid": False, "errors": errors, "warnings": warnings, "suggestions": suggestions}
            
            # Auto-detect delimiter if not specified
            if delimiter is None:
                delimiter = self._detect_delimiter(data_lines[0])
                if delimiter:
                    warnings.append(f"Auto-detected delimiter: '{delimiter}'")
                else:
                    errors.append("Could not auto-detect column delimiter")
                    suggestions.append("Specify delimiter explicitly in schema configuration")
                    return {"valid": False, "errors": errors, "warnings": warnings, "suggestions": suggestions}
            
            # Parse first data line to check column structure
            first_line_columns = [col.strip() for col in data_lines[0].split(delimiter)]
            actual_column_count = len(first_line_columns)
            
            # Check column count
            if expected_columns:
                expected_column_count = len(expected_columns)
                if actual_column_count != expected_column_count:
                    errors.append(f"Column count mismatch: expected {expected_column_count} columns {expected_columns}, found {actual_column_count}")
                    suggestions.append("Check delimiter setting or verify dataset format")
            
            # Check row count
            actual_row_count = len(data_lines)
            if expected_rows is not None:
                if actual_row_count != expected_rows:
                    if abs(actual_row_count - expected_rows) <= 1:
                        warnings.append(f"Row count close but not exact: expected {expected_rows}, found {actual_row_count}")
                    else:
                        errors.append(f"Row count mismatch: expected {expected_rows}, found {actual_row_count}")
                        suggestions.append("Verify dataset completeness or update expected row count")
            
            # Validate data types in each column
            if expected_columns and actual_column_count == len(expected_columns):
                type_errors = self._validate_column_types(data_lines, delimiter, expected_columns)
                errors.extend(type_errors)
            
            # Check for consistent column count across all rows
            inconsistent_rows = []
            for i, line in enumerate(data_lines[1:], start=2):  # Start from second line
                columns = [col.strip() for col in line.split(delimiter)]
                if len(columns) != actual_column_count:
                    inconsistent_rows.append(i)
            
            if inconsistent_rows:
                if len(inconsistent_rows) <= 5:
                    errors.append(f"Inconsistent column count in rows: {inconsistent_rows}")
                else:
                    errors.append(f"Inconsistent column count in {len(inconsistent_rows)} rows (first 5: {inconsistent_rows[:5]})")
                suggestions.append("Check for missing values or incorrect delimiters in data")
            
        except Exception as e:
            errors.append(f"Failed to read or parse ASCII table: {e}")
            suggestions.append("Check file encoding and format")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def _detect_delimiter(self, sample_line: str) -> Optional[str]:
        """
        Auto-detect column delimiter in ASCII table
        
        Args:
            sample_line: Sample line from the file
            
        Returns:
            Detected delimiter or None if not found
        """
        # Common delimiters in order of preference
        delimiters = ['\t', ' ', ',', ';', '|']
        
        for delimiter in delimiters:
            if delimiter in sample_line:
                # For space delimiter, check if it's multiple spaces (common in astronomy)
                if delimiter == ' ':
                    # Look for multiple consecutive spaces
                    if re.search(r'\s{2,}', sample_line):
                        return r'\s+'  # Regex for multiple whitespace
                    elif sample_line.count(' ') >= 2:  # At least 2 spaces suggest columns
                        return ' '
                else:
                    return delimiter
        
        return None
    
    def _validate_column_types(self, data_lines: List[str], delimiter: str, expected_columns: List[str]) -> List[str]:
        """
        Validate data types in columns based on column names
        
        Args:
            data_lines: List of data lines
            delimiter: Column delimiter
            expected_columns: List of expected column names
            
        Returns:
            List of type validation errors
        """
        errors = []
        
        # Sample a few rows for type checking (not all for performance)
        sample_size = min(10, len(data_lines))
        sample_lines = data_lines[:sample_size]
        
        for col_idx, col_name in enumerate(expected_columns):
            col_name_lower = col_name.lower()
            
            # Extract values for this column
            column_values = []
            for line in sample_lines:
                if delimiter == r'\s+':
                    columns = re.split(r'\s+', line.strip())
                else:
                    columns = [col.strip() for col in line.split(delimiter)]
                
                if col_idx < len(columns):
                    column_values.append(columns[col_idx])
            
            if not column_values:
                continue
            
            # Infer expected type from column name
            expected_type = self._infer_column_type(col_name_lower)
            
            # Validate types
            type_errors = []
            for i, value in enumerate(column_values):
                if not self._validate_value_type(value, expected_type):
                    type_errors.append(f"Row {i+1}: '{value}' is not a valid {expected_type}")
            
            if type_errors:
                if len(type_errors) <= 3:
                    errors.append(f"Column '{col_name}' type validation failed: {'; '.join(type_errors)}")
                else:
                    errors.append(f"Column '{col_name}' has {len(type_errors)} type validation failures (first 3: {'; '.join(type_errors[:3])})")
        
        return errors
    
    def _infer_column_type(self, column_name: str) -> str:
        """
        Infer expected data type from column name
        
        Args:
            column_name: Column name (lowercase)
            
        Returns:
            Expected type: 'float', 'int', 'string'
        """
        # Common patterns in cosmology datasets
        if any(pattern in column_name for pattern in ['z', 'redshift', 'mu', 'dm', 'dh', 'r', 'l_a', 'theta', 'err', 'error', 'sigma']):
            return 'float'
        elif any(pattern in column_name for pattern in ['id', 'cid', 'n_', 'count', 'num']):
            return 'int'
        else:
            return 'string'  # Default to string for unknown patterns
    
    def _validate_value_type(self, value: str, expected_type: str) -> bool:
        """
        Validate if a value matches the expected type
        
        Args:
            value: String value to validate
            expected_type: Expected type ('float', 'int', 'string')
            
        Returns:
            True if value matches expected type
        """
        value = value.strip()
        
        if not value or value.lower() in ['nan', 'null', 'none', '-', '--']:
            return True  # Allow missing values
        
        try:
            if expected_type == 'float':
                float(value)
                return True
            elif expected_type == 'int':
                int(value)
                return True
            elif expected_type == 'string':
                return True  # Any string is valid
        except ValueError:
            return False
        
        return False
    
    def _verify_json_schema(self, file_path: Path, schema_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify JSON format schema
        
        Args:
            file_path: Path to JSON file
            schema_config: Schema configuration
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        suggestions = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Basic JSON structure validation
            expected_keys = schema_config.get("required_keys", [])
            if expected_keys:
                missing_keys = [key for key in expected_keys if key not in data]
                if missing_keys:
                    errors.append(f"Missing required keys: {missing_keys}")
                    suggestions.append("Check JSON structure and required fields")
            
            # Validate data types for specific keys
            key_types = schema_config.get("key_types", {})
            for key, expected_type in key_types.items():
                if key in data:
                    actual_type = type(data[key]).__name__
                    if actual_type != expected_type:
                        errors.append(f"Key '{key}' has wrong type: expected {expected_type}, got {actual_type}")
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {e}")
            suggestions.append("Check JSON syntax and encoding")
        except Exception as e:
            errors.append(f"Failed to validate JSON schema: {e}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def _verify_fits_schema(self, file_path: Path, schema_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify FITS format schema (basic implementation)
        
        Args:
            file_path: Path to FITS file
            schema_config: Schema configuration
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Try to import astropy for FITS support
            try:
                from astropy.io import fits
            except ImportError:
                errors.append("FITS validation requires astropy package")
                suggestions.append("Install astropy: pip install astropy")
                return {"valid": False, "errors": errors, "warnings": warnings, "suggestions": suggestions}
            
            # Open and validate FITS file
            with fits.open(file_path) as hdul:
                # Check number of HDUs
                expected_hdus = schema_config.get("expected_hdus")
                if expected_hdus is not None:
                    actual_hdus = len(hdul)
                    if actual_hdus != expected_hdus:
                        errors.append(f"HDU count mismatch: expected {expected_hdus}, found {actual_hdus}")
                
                # Check for required extensions
                required_extensions = schema_config.get("required_extensions", [])
                for ext_name in required_extensions:
                    try:
                        hdul[ext_name]
                    except KeyError:
                        errors.append(f"Missing required extension: {ext_name}")
                
                # Basic header validation
                primary_header = hdul[0].header
                required_keywords = schema_config.get("required_keywords", [])
                for keyword in required_keywords:
                    if keyword not in primary_header:
                        errors.append(f"Missing required header keyword: {keyword}")
        
        except Exception as e:
            errors.append(f"Failed to validate FITS file: {e}")
            suggestions.append("Check FITS file format and accessibility")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def re_verify_dataset(self, dataset_name: str, file_path: Path, 
                         verification_config: Dict[str, Any]) -> VerificationResult:
        """
        Re-verify an existing dataset (alias for verify_dataset)
        
        Args:
            dataset_name: Name of the dataset
            file_path: Path to dataset file
            verification_config: Verification configuration
            
        Returns:
            VerificationResult with current verification status
        """
        return self.verify_dataset(dataset_name, file_path, verification_config)
    
    def batch_verify(self, datasets: List[Tuple[str, Path, Dict[str, Any]]]) -> List[VerificationResult]:
        """
        Verify multiple datasets in batch
        
        Args:
            datasets: List of (dataset_name, file_path, verification_config) tuples
            
        Returns:
            List of VerificationResult objects
        """
        results = []
        for dataset_name, file_path, verification_config in datasets:
            try:
                result = self.verify_dataset(dataset_name, file_path, verification_config)
                results.append(result)
            except Exception as e:
                # Create failed result for unexpected errors
                failed_result = VerificationResult(
                    dataset_name=dataset_name,
                    file_path=file_path,
                    verification_time=datetime.now(),
                    is_valid=False,
                    status=VerificationStatus.FAILED,
                    sha256_match=False,
                    sha256_expected=None,
                    sha256_actual=None,
                    size_match=False,
                    size_expected=None,
                    size_actual=None,
                    schema_valid=False,
                    schema_errors=[],
                    schema_warnings=[],
                    errors=[f"Unexpected verification error: {e}"],
                    warnings=[],
                    suggestions=["Check dataset configuration and file accessibility"]
                )
                results.append(failed_result)
        
        return results
    
    def get_verification_summary(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """
        Generate summary statistics for verification results
        
        Args:
            results: List of verification results
            
        Returns:
            Dictionary with summary statistics
        """
        total_datasets = len(results)
        if total_datasets == 0:
            return {"total_datasets": 0, "all_valid": True}
        
        valid_datasets = sum(1 for r in results if r.is_valid)
        failed_datasets = sum(1 for r in results if r.status == VerificationStatus.FAILED)
        partial_datasets = sum(1 for r in results if r.status == VerificationStatus.PARTIAL)
        
        checksum_failures = sum(1 for r in results if not r.sha256_match and r.sha256_expected)
        size_failures = sum(1 for r in results if not r.size_match and r.size_expected)
        schema_failures = sum(1 for r in results if not r.schema_valid)
        
        return {
            "total_datasets": total_datasets,
            "valid_datasets": valid_datasets,
            "failed_datasets": failed_datasets,
            "partial_datasets": partial_datasets,
            "success_rate": valid_datasets / total_datasets,
            "checksum_failures": checksum_failures,
            "size_failures": size_failures,
            "schema_failures": schema_failures,
            "all_valid": valid_datasets == total_datasets
        }
    
    def generate_verification_report(self, results: List[VerificationResult], 
                                   output_format: str = "text") -> str:
        """
        Generate comprehensive verification report
        
        Args:
            results: List of verification results
            output_format: Output format ('text', 'json', 'html')
            
        Returns:
            Formatted verification report
        """
        if output_format == "json":
            return self._generate_json_report(results)
        elif output_format == "html":
            return self._generate_html_report(results)
        else:
            return self._generate_text_report(results)
    
    def _generate_text_report(self, results: List[VerificationResult]) -> str:
        """Generate text format verification report"""
        summary = self.get_verification_summary(results)
        
        report_lines = [
            "=" * 80,
            "DATASET VERIFICATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY:",
            f"  Total datasets: {summary['total_datasets']}",
            f"  Valid datasets: {summary['valid_datasets']}",
            f"  Failed datasets: {summary['failed_datasets']}",
            f"  Partial datasets: {summary['partial_datasets']}",
            f"  Success rate: {summary['success_rate']:.1%}",
            "",
            "FAILURE BREAKDOWN:",
            f"  Checksum failures: {summary['checksum_failures']}",
            f"  Size failures: {summary['size_failures']}",
            f"  Schema failures: {summary['schema_failures']}",
            "",
            "DETAILED RESULTS:",
            "-" * 80
        ]
        
        for result in results:
            status_symbol = "✓" if result.is_valid else "✗"
            report_lines.extend([
                f"{status_symbol} {result.dataset_name} ({result.status.value.upper()})",
                f"  File: {result.file_path}",
                f"  Verified: {result.verification_time.strftime('%Y-%m-%d %H:%M:%S')}"
            ])
            
            if result.sha256_expected:
                checksum_symbol = "✓" if result.sha256_match else "✗"
                report_lines.append(f"  Checksum: {checksum_symbol} SHA256")
            
            if result.size_expected:
                size_symbol = "✓" if result.size_match else "✗"
                report_lines.append(f"  Size: {size_symbol} {result.size_actual} bytes")
            
            schema_symbol = "✓" if result.schema_valid else "✗"
            report_lines.append(f"  Schema: {schema_symbol}")
            
            if result.errors:
                report_lines.append("  ERRORS:")
                for error in result.errors:
                    report_lines.append(f"    - {error}")
            
            if result.warnings:
                report_lines.append("  WARNINGS:")
                for warning in result.warnings:
                    report_lines.append(f"    - {warning}")
            
            if result.suggestions:
                report_lines.append("  SUGGESTIONS:")
                for suggestion in result.suggestions:
                    report_lines.append(f"    - {suggestion}")
            
            report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            f"Report completed: {summary['all_valid'] and 'ALL DATASETS VALID' or 'ISSUES FOUND'}",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def _generate_json_report(self, results: List[VerificationResult]) -> str:
        """Generate JSON format verification report"""
        summary = self.get_verification_summary(results)
        
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "dataset_verification",
                "format_version": "1.0"
            },
            "summary": summary,
            "results": [result.to_dict() for result in results]
        }
        
        return json.dumps(report_data, indent=2, sort_keys=True)
    
    def _generate_html_report(self, results: List[VerificationResult]) -> str:
        """Generate HTML format verification report"""
        summary = self.get_verification_summary(results)
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Verification Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        .summary { background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .dataset { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .valid { border-left: 5px solid #4CAF50; }
        .invalid { border-left: 5px solid #f44336; }
        .partial { border-left: 5px solid #ff9800; }
        .error { color: #d32f2f; }
        .warning { color: #f57c00; }
        .suggestion { color: #1976d2; }
        .checkmark { color: #4CAF50; }
        .crossmark { color: #f44336; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Dataset Verification Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <ul>
            <li>Total datasets: {total_datasets}</li>
            <li>Valid datasets: {valid_datasets}</li>
            <li>Failed datasets: {failed_datasets}</li>
            <li>Success rate: {success_rate:.1%}</li>
        </ul>
    </div>
    
    <h2>Detailed Results</h2>
    {dataset_results}
</body>
</html>
        """.strip()
        
        dataset_html_parts = []
        for result in results:
            status_class = "valid" if result.is_valid else ("partial" if result.status == VerificationStatus.PARTIAL else "invalid")
            status_symbol = "✓" if result.is_valid else "✗"
            
            errors_html = ""
            if result.errors:
                errors_html = "<h4>Errors:</h4><ul>" + "".join(f'<li class="error">{error}</li>' for error in result.errors) + "</ul>"
            
            warnings_html = ""
            if result.warnings:
                warnings_html = "<h4>Warnings:</h4><ul>" + "".join(f'<li class="warning">{warning}</li>' for warning in result.warnings) + "</ul>"
            
            suggestions_html = ""
            if result.suggestions:
                suggestions_html = "<h4>Suggestions:</h4><ul>" + "".join(f'<li class="suggestion">{suggestion}</li>' for suggestion in result.suggestions) + "</ul>"
            
            dataset_html = f"""
            <div class="dataset {status_class}">
                <h3>{status_symbol} {result.dataset_name}</h3>
                <p><strong>File:</strong> {result.file_path}</p>
                <p><strong>Status:</strong> {result.status.value.upper()}</p>
                <p><strong>Verified:</strong> {result.verification_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                {errors_html}
                {warnings_html}
                {suggestions_html}
            </div>
            """
            dataset_html_parts.append(dataset_html)
        
        return html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_datasets=summary['total_datasets'],
            valid_datasets=summary['valid_datasets'],
            failed_datasets=summary['failed_datasets'],
            success_rate=summary['success_rate'],
            dataset_results="".join(dataset_html_parts)
        )
    
    def diagnose_verification_failure(self, result: VerificationResult) -> Dict[str, Any]:
        """
        Provide detailed diagnosis for verification failures
        
        Args:
            result: Failed verification result
            
        Returns:
            Dictionary with diagnostic information and recovery strategies
        """
        diagnosis = {
            "dataset_name": result.dataset_name,
            "failure_type": [],
            "root_causes": [],
            "recovery_strategies": [],
            "priority": "low"
        }
        
        if not result.is_valid:
            # Analyze failure types
            if not result.sha256_match and result.sha256_expected:
                diagnosis["failure_type"].append("checksum_mismatch")
                diagnosis["root_causes"].extend([
                    "File corruption during download or storage",
                    "Incorrect expected checksum in manifest",
                    "File modified after download"
                ])
                diagnosis["recovery_strategies"].extend([
                    "Re-download the dataset from primary source",
                    "Try alternative mirror sources",
                    "Verify expected checksum with dataset provider",
                    "Check disk integrity and file system errors"
                ])
                diagnosis["priority"] = "high"
            
            if not result.size_match and result.size_expected:
                diagnosis["failure_type"].append("size_mismatch")
                diagnosis["root_causes"].extend([
                    "Incomplete download",
                    "File truncation or corruption",
                    "Incorrect expected size in manifest"
                ])
                diagnosis["recovery_strategies"].extend([
                    "Re-download with resume capability",
                    "Check network stability during download",
                    "Verify expected size with dataset provider"
                ])
                if diagnosis["priority"] != "high":
                    diagnosis["priority"] = "medium"
            
            if not result.schema_valid:
                diagnosis["failure_type"].append("schema_validation")
                diagnosis["root_causes"].extend([
                    "Dataset format changed from expected",
                    "Incorrect schema configuration",
                    "File encoding issues"
                ])
                diagnosis["recovery_strategies"].extend([
                    "Update schema configuration in manifest",
                    "Check dataset documentation for format changes",
                    "Verify file encoding (UTF-8, ASCII, etc.)"
                ])
                if diagnosis["priority"] == "low":
                    diagnosis["priority"] = "medium"
        
        # Add specific error analysis
        for error in result.errors:
            if "permission" in error.lower():
                diagnosis["root_causes"].append("File permission issues")
                diagnosis["recovery_strategies"].append("Check file and directory permissions")
            elif "not found" in error.lower():
                diagnosis["root_causes"].append("Missing file or incorrect path")
                diagnosis["recovery_strategies"].append("Verify file path and download status")
            elif "encoding" in error.lower():
                diagnosis["root_causes"].append("File encoding problems")
                diagnosis["recovery_strategies"].append("Check file encoding and character set")
        
        return diagnosis
    
    def suggest_recovery_actions(self, results: List[VerificationResult]) -> List[Dict[str, Any]]:
        """
        Suggest recovery actions for failed verifications
        
        Args:
            results: List of verification results
            
        Returns:
            List of suggested recovery actions with priorities
        """
        recovery_actions = []
        
        failed_results = [r for r in results if not r.is_valid]
        
        for result in failed_results:
            diagnosis = self.diagnose_verification_failure(result)
            
            action = {
                "dataset_name": result.dataset_name,
                "priority": diagnosis["priority"],
                "failure_types": diagnosis["failure_type"],
                "recommended_actions": diagnosis["recovery_strategies"][:3],  # Top 3 actions
                "estimated_effort": self._estimate_recovery_effort(diagnosis["failure_type"]),
                "automation_possible": self._can_automate_recovery(diagnosis["failure_type"])
            }
            
            recovery_actions.append(action)
        
        # Sort by priority (high -> medium -> low)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recovery_actions.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recovery_actions
    
    def _estimate_recovery_effort(self, failure_types: List[str]) -> str:
        """Estimate effort required for recovery"""
        if "checksum_mismatch" in failure_types:
            return "medium"  # Re-download required
        elif "size_mismatch" in failure_types:
            return "low"     # Usually just re-download
        elif "schema_validation" in failure_types:
            return "high"    # May require configuration changes
        else:
            return "medium"
    
    def _can_automate_recovery(self, failure_types: List[str]) -> bool:
        """Check if recovery can be automated"""
        # Checksum and size mismatches can often be auto-recovered by re-downloading
        auto_recoverable = {"checksum_mismatch", "size_mismatch"}
        return any(ft in auto_recoverable for ft in failure_types)
    
    def export_verification_results(self, results: List[VerificationResult], 
                                  output_path: Path, format_type: str = "json") -> bool:
        """
        Export verification results to file
        
        Args:
            results: List of verification results
            output_path: Path to output file
            format_type: Export format ('json', 'csv', 'html')
            
        Returns:
            True if export successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type == "json":
                report_content = self._generate_json_report(results)
            elif format_type == "html":
                report_content = self._generate_html_report(results)
            elif format_type == "csv":
                report_content = self._generate_csv_report(results)
            else:
                report_content = self._generate_text_report(results)
            
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            return True
            
        except Exception as e:
            print(f"Failed to export verification results: {e}")
            return False
    
    def _generate_csv_report(self, results: List[VerificationResult]) -> str:
        """Generate CSV format verification report"""
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "dataset_name", "file_path", "verification_time", "is_valid", "status",
            "sha256_match", "size_match", "schema_valid", "errors", "warnings", "suggestions"
        ])
        
        # Write data rows
        for result in results:
            writer.writerow([
                result.dataset_name,
                str(result.file_path),
                result.verification_time.isoformat(),
                result.is_valid,
                result.status.value,
                result.sha256_match,
                result.size_match,
                result.schema_valid,
                "; ".join(result.errors),
                "; ".join(result.warnings),
                "; ".join(result.suggestions)
            ])
        
        return output.getvalue()
    
    def create_verification_checkpoint(self, results: List[VerificationResult], 
                                    checkpoint_path: Path) -> bool:
        """
        Create verification checkpoint for resuming operations
        
        Args:
            results: Current verification results
            checkpoint_path: Path to save checkpoint
            
        Returns:
            True if checkpoint created successfully
        """
        try:
            checkpoint_data = {
                "checkpoint_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_datasets": len(results),
                "verified_datasets": [r.dataset_name for r in results if r.is_valid],
                "failed_datasets": [r.dataset_name for r in results if not r.is_valid],
                "results": [result.to_dict() for result in results]
            }
            
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Failed to create verification checkpoint: {e}")
            return False
    
    def load_verification_checkpoint(self, checkpoint_path: Path) -> Optional[List[VerificationResult]]:
        """
        Load verification results from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            List of verification results or None if loading fails
        """
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            results = []
            for result_data in checkpoint_data.get("results", []):
                result = VerificationResult.from_dict(result_data)
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Failed to load verification checkpoint: {e}")
            return None