"""
Reproduction diagnostics and error handling

This module provides comprehensive diagnostics for reproduction failures,
detailed error analysis, and recovery suggestions to help users troubleshoot
and resolve dataset reproduction issues.
"""

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import traceback
import urllib.parse
import urllib.request

from ..core.registry_manager import RegistryManager
from ..core.manifest_schema import DatasetManifest


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostic issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DiagnosticCategory(Enum):
    """Categories of diagnostic issues"""
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    VERIFICATION = "verification"
    CONFIGURATION = "configuration"
    ENVIRONMENT = "environment"
    MANIFEST = "manifest"
    REGISTRY = "registry"
    PERMISSIONS = "permissions"


@dataclass
class DiagnosticIssue:
    """Individual diagnostic issue"""
    category: DiagnosticCategory
    severity: DiagnosticSeverity
    title: str
    description: str
    dataset_name: Optional[str] = None
    error_code: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    technical_details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "dataset_name": self.dataset_name,
            "error_code": self.error_code,
            "suggestions": self.suggestions,
            "technical_details": self.technical_details,
            "timestamp": self.timestamp
        }


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report"""
    system_status: str
    overall_health: str
    total_issues: int
    critical_issues: int
    error_issues: int
    warning_issues: int
    info_issues: int
    issues: List[DiagnosticIssue] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    recovery_steps: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "system_status": self.system_status,
            "overall_health": self.overall_health,
            "summary": {
                "total_issues": self.total_issues,
                "critical_issues": self.critical_issues,
                "error_issues": self.error_issues,
                "warning_issues": self.warning_issues,
                "info_issues": self.info_issues
            },
            "issues": [issue.to_dict() for issue in self.issues],
            "system_info": self.system_info,
            "recommendations": self.recommendations,
            "recovery_steps": self.recovery_steps,
            "timestamp": self.timestamp
        }


class ReproductionDiagnostics:
    """
    Comprehensive diagnostics for reproduction system
    
    Analyzes system state, identifies issues, and provides detailed
    recovery suggestions for reproduction failures.
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path] = "data/datasets_manifest.json",
        registry_path: Union[str, Path] = "data/registry"
    ):
        """
        Initialize diagnostics system
        
        Args:
            manifest_path: Path to dataset manifest file
            registry_path: Path to registry directory
        """
        self.manifest_path = Path(manifest_path)
        self.registry_path = Path(registry_path)
        
        # Initialize components (with error handling)
        self.manifest = None
        self.registry_manager = None
        
        try:
            if self.manifest_path.exists():
                self.manifest = DatasetManifest(self.manifest_path)
        except Exception:
            pass  # Will be caught in diagnostics
        
        try:
            self.registry_manager = RegistryManager(self.registry_path)
        except Exception:
            pass  # Will be caught in diagnostics
    
    def run_comprehensive_diagnostics(self) -> DiagnosticReport:
        """
        Run comprehensive system diagnostics
        
        Returns:
            DiagnosticReport with complete analysis and recommendations
        """
        issues = []
        
        # System-level diagnostics
        issues.extend(self._diagnose_system_environment())
        issues.extend(self._diagnose_filesystem())
        issues.extend(self._diagnose_network_connectivity())
        issues.extend(self._diagnose_permissions())
        
        # Component diagnostics
        issues.extend(self._diagnose_manifest())
        issues.extend(self._diagnose_registry())
        
        # Dataset-specific diagnostics
        issues.extend(self._diagnose_datasets())
        
        # Configuration diagnostics
        issues.extend(self._diagnose_configuration())
        
        # Count issues by severity
        critical_count = sum(1 for issue in issues if issue.severity == DiagnosticSeverity.CRITICAL)
        error_count = sum(1 for issue in issues if issue.severity == DiagnosticSeverity.ERROR)
        warning_count = sum(1 for issue in issues if issue.severity == DiagnosticSeverity.WARNING)
        info_count = sum(1 for issue in issues if issue.severity == DiagnosticSeverity.INFO)
        
        # Determine overall health
        if critical_count > 0:
            overall_health = "critical"
            system_status = "System has critical issues preventing reproduction"
        elif error_count > 0:
            overall_health = "degraded"
            system_status = "System has errors that may affect reproduction"
        elif warning_count > 0:
            overall_health = "warning"
            system_status = "System is functional but has warnings"
        else:
            overall_health = "healthy"
            system_status = "System is ready for reproduction"
        
        # Generate recommendations and recovery steps
        recommendations = self._generate_recommendations(issues)
        recovery_steps = self._generate_recovery_steps(issues)
        
        # Collect system information
        system_info = self._collect_system_info()
        
        return DiagnosticReport(
            system_status=system_status,
            overall_health=overall_health,
            total_issues=len(issues),
            critical_issues=critical_count,
            error_issues=error_count,
            warning_issues=warning_count,
            info_issues=info_count,
            issues=issues,
            system_info=system_info,
            recommendations=recommendations,
            recovery_steps=recovery_steps
        )
    
    def diagnose_dataset_failure(self, dataset_name: str, error_message: str, 
                                error_context: Optional[Dict[str, Any]] = None) -> List[DiagnosticIssue]:
        """
        Diagnose specific dataset failure
        
        Args:
            dataset_name: Name of the failed dataset
            error_message: Error message from failure
            error_context: Additional context about the failure
            
        Returns:
            List of diagnostic issues specific to this failure
        """
        issues = []
        error_context = error_context or {}
        
        # Analyze error message patterns
        error_lower = error_message.lower()
        
        # Network-related errors
        if any(keyword in error_lower for keyword in ['network', 'connection', 'timeout', 'dns', 'resolve']):
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.NETWORK,
                severity=DiagnosticSeverity.ERROR,
                title=f"Network connectivity issue for {dataset_name}",
                description=f"Failed to download dataset due to network error: {error_message}",
                dataset_name=dataset_name,
                error_code="NETWORK_ERROR",
                suggestions=[
                    "Check internet connectivity",
                    "Verify DNS resolution is working",
                    "Try downloading from a different network",
                    "Check if the source URL is accessible in a web browser",
                    "Consider using a VPN if behind a restrictive firewall"
                ],
                technical_details={"original_error": error_message, **error_context}
            ))
        
        # HTTP errors
        elif any(keyword in error_lower for keyword in ['404', 'not found', '403', 'forbidden', '401', 'unauthorized']):
            status_code = self._extract_http_status(error_message)
            
            if '404' in error_lower or 'not found' in error_lower:
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.NETWORK,
                    severity=DiagnosticSeverity.ERROR,
                    title=f"Dataset source not found for {dataset_name}",
                    description=f"The dataset source URL returned a 404 Not Found error",
                    dataset_name=dataset_name,
                    error_code="SOURCE_NOT_FOUND",
                    suggestions=[
                        "Check if the dataset URL has changed",
                        "Try the mirror source if available",
                        "Contact the data provider for updated URLs",
                        "Check the dataset manifest for correct URLs",
                        "Search for alternative sources of the same dataset"
                    ],
                    technical_details={"http_status": status_code, "original_error": error_message, **error_context}
                ))
            
            elif '403' in error_lower or 'forbidden' in error_lower:
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.NETWORK,
                    severity=DiagnosticSeverity.ERROR,
                    title=f"Access forbidden for {dataset_name}",
                    description=f"Access to the dataset source is forbidden (403 error)",
                    dataset_name=dataset_name,
                    error_code="ACCESS_FORBIDDEN",
                    suggestions=[
                        "Check if authentication is required",
                        "Verify you have permission to access this dataset",
                        "Try accessing the URL in a web browser",
                        "Contact the data provider for access instructions",
                        "Check if registration is required"
                    ],
                    technical_details={"http_status": status_code, "original_error": error_message, **error_context}
                ))
        
        # Verification errors
        elif any(keyword in error_lower for keyword in ['checksum', 'sha256', 'verification', 'hash']):
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.VERIFICATION,
                severity=DiagnosticSeverity.ERROR,
                title=f"Verification failure for {dataset_name}",
                description=f"Dataset failed checksum or verification: {error_message}",
                dataset_name=dataset_name,
                error_code="VERIFICATION_FAILED",
                suggestions=[
                    "Try re-downloading the dataset with --force-refresh",
                    "Check if the dataset source has been updated",
                    "Verify the expected checksum in the manifest is correct",
                    "Check for network issues that might corrupt downloads",
                    "Try downloading from a mirror source if available"
                ],
                technical_details={"original_error": error_message, **error_context}
            ))
        
        # File system errors
        elif any(keyword in error_lower for keyword in ['permission', 'access denied', 'disk', 'space', 'directory']):
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.FILESYSTEM,
                severity=DiagnosticSeverity.ERROR,
                title=f"Filesystem issue for {dataset_name}",
                description=f"Filesystem error during dataset operation: {error_message}",
                dataset_name=dataset_name,
                error_code="FILESYSTEM_ERROR",
                suggestions=[
                    "Check available disk space",
                    "Verify write permissions to the data directory",
                    "Ensure the target directory exists and is writable",
                    "Check if the filesystem is full or read-only",
                    "Try running with elevated permissions if necessary"
                ],
                technical_details={"original_error": error_message, **error_context}
            ))
        
        # Generic error
        else:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.CONFIGURATION,
                severity=DiagnosticSeverity.ERROR,
                title=f"Unknown error for {dataset_name}",
                description=f"Unrecognized error during dataset operation: {error_message}",
                dataset_name=dataset_name,
                error_code="UNKNOWN_ERROR",
                suggestions=[
                    "Check the full error log for more details",
                    "Try running the operation again",
                    "Check system resources (disk space, memory)",
                    "Verify the dataset configuration in the manifest",
                    "Contact support with the full error message"
                ],
                technical_details={"original_error": error_message, **error_context}
            ))
        
        return issues
    
    def _diagnose_system_environment(self) -> List[DiagnosticIssue]:
        """Diagnose system environment issues"""
        issues = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 7):
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.ENVIRONMENT,
                severity=DiagnosticSeverity.CRITICAL,
                title="Unsupported Python version",
                description=f"Python {python_version.major}.{python_version.minor} is not supported. Minimum required: 3.7",
                suggestions=[
                    "Upgrade to Python 3.7 or later",
                    "Use a virtual environment with a supported Python version",
                    "Check your system's Python installation"
                ],
                technical_details={"python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}"}
            ))
        
        # Check required modules
        required_modules = ['requests', 'pathlib', 'json', 'hashlib']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.ENVIRONMENT,
                severity=DiagnosticSeverity.CRITICAL,
                title="Missing required Python modules",
                description=f"Required modules not found: {', '.join(missing_modules)}",
                suggestions=[
                    "Install missing modules with pip",
                    "Check your Python environment setup",
                    "Ensure all dependencies are installed"
                ],
                technical_details={"missing_modules": missing_modules}
            ))
        
        # Check PBUF environment
        try:
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.ENVIRONMENT,
                    severity=DiagnosticSeverity.WARNING,
                    title="Not in a Git repository",
                    description="PBUF commit tracking may not work properly outside a Git repository",
                    suggestions=[
                        "Run from within the PBUF Git repository",
                        "Initialize a Git repository if needed",
                        "Commit tracking will be disabled"
                    ]
                ))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.ENVIRONMENT,
                severity=DiagnosticSeverity.WARNING,
                title="Git not available",
                description="Git is not installed or not in PATH. Commit tracking will be disabled.",
                suggestions=[
                    "Install Git for commit tracking",
                    "Add Git to your system PATH",
                    "Commit tracking will be disabled"
                ]
            ))
        
        return issues
    
    def _diagnose_filesystem(self) -> List[DiagnosticIssue]:
        """Diagnose filesystem issues"""
        issues = []
        
        # Check data directory
        data_dir = Path("data")
        
        if not data_dir.exists():
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.FILESYSTEM,
                severity=DiagnosticSeverity.WARNING,
                title="Data directory does not exist",
                description="The data directory 'data/' does not exist and will be created",
                suggestions=[
                    "The directory will be created automatically",
                    "Ensure you have write permissions in the current directory"
                ],
                technical_details={"data_dir": str(data_dir.absolute())}
            ))
        else:
            # Check write permissions
            try:
                test_file = data_dir / ".write_test"
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.PERMISSIONS,
                    severity=DiagnosticSeverity.CRITICAL,
                    title="No write permission to data directory",
                    description="Cannot write to the data directory",
                    suggestions=[
                        "Check file permissions on the data directory",
                        "Run with appropriate user permissions",
                        "Change directory ownership if necessary"
                    ],
                    technical_details={"data_dir": str(data_dir.absolute())}
                ))
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(data_dir.parent if data_dir.exists() else Path.cwd())
            free_gb = free / (1024**3)
            
            if free_gb < 1.0:  # Less than 1 GB free
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.FILESYSTEM,
                    severity=DiagnosticSeverity.CRITICAL,
                    title="Low disk space",
                    description=f"Only {free_gb:.1f} GB of free disk space available",
                    suggestions=[
                        "Free up disk space before downloading datasets",
                        "Clean up old or unnecessary files",
                        "Consider using a different storage location"
                    ],
                    technical_details={"free_space_gb": free_gb, "total_space_gb": total / (1024**3)}
                ))
            elif free_gb < 5.0:  # Less than 5 GB free
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.FILESYSTEM,
                    severity=DiagnosticSeverity.WARNING,
                    title="Limited disk space",
                    description=f"Only {free_gb:.1f} GB of free disk space available",
                    suggestions=[
                        "Monitor disk space during dataset downloads",
                        "Consider freeing up additional space",
                        "Large datasets may require more space"
                    ],
                    technical_details={"free_space_gb": free_gb}
                ))
        except Exception:
            pass  # Disk space check is optional
        
        return issues
    
    def _diagnose_network_connectivity(self) -> List[DiagnosticIssue]:
        """Diagnose network connectivity issues"""
        issues = []
        
        # Test basic internet connectivity
        test_urls = [
            "https://www.google.com",
            "https://github.com",
            "https://zenodo.org"
        ]
        
        connectivity_failures = []
        
        for url in test_urls:
            try:
                req = urllib.request.Request(url, method='HEAD')
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status != 200:
                        connectivity_failures.append(f"{url} (HTTP {response.status})")
            except Exception as e:
                connectivity_failures.append(f"{url} ({str(e)})")
        
        if len(connectivity_failures) == len(test_urls):
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.NETWORK,
                severity=DiagnosticSeverity.CRITICAL,
                title="No internet connectivity",
                description="Cannot reach any test URLs. Internet connection may be down.",
                suggestions=[
                    "Check your internet connection",
                    "Verify network settings",
                    "Check if you're behind a firewall or proxy",
                    "Try connecting to a different network"
                ],
                technical_details={"failed_urls": connectivity_failures}
            ))
        elif connectivity_failures:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.NETWORK,
                severity=DiagnosticSeverity.WARNING,
                title="Limited internet connectivity",
                description=f"Some test URLs are not reachable: {', '.join(connectivity_failures)}",
                suggestions=[
                    "Check if specific sites are blocked",
                    "Verify firewall or proxy settings",
                    "Some dataset sources may not be accessible"
                ],
                technical_details={"failed_urls": connectivity_failures}
            ))
        
        # Test dataset source connectivity if manifest is available
        if self.manifest:
            dataset_source_failures = []
            
            for dataset_name in self.manifest.list_datasets()[:3]:  # Test first 3 datasets
                try:
                    dataset_info = self.manifest.get_dataset_info(dataset_name)
                    if dataset_info and dataset_info.sources:
                        primary_source = dataset_info.sources.get("primary")
                        if primary_source:
                            url = primary_source.get("url")
                            if url:
                                try:
                                    req = urllib.request.Request(url, method='HEAD')
                                    with urllib.request.urlopen(req, timeout=10) as response:
                                        if response.status not in [200, 302, 301]:
                                            dataset_source_failures.append(f"{dataset_name}: HTTP {response.status}")
                                except Exception as e:
                                    dataset_source_failures.append(f"{dataset_name}: {str(e)}")
                except Exception:
                    continue
            
            if dataset_source_failures:
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.NETWORK,
                    severity=DiagnosticSeverity.WARNING,
                    title="Dataset source connectivity issues",
                    description=f"Some dataset sources are not reachable",
                    suggestions=[
                        "Check if dataset source URLs are still valid",
                        "Try using mirror sources if available",
                        "Contact dataset providers for updated URLs"
                    ],
                    technical_details={"failed_sources": dataset_source_failures}
                ))
        
        return issues
    
    def _diagnose_permissions(self) -> List[DiagnosticIssue]:
        """Diagnose permission issues"""
        issues = []
        
        # Check current user permissions
        import os
        
        # Check if running as root (which might cause issues)
        if os.geteuid() == 0:  # Unix-like systems
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.PERMISSIONS,
                severity=DiagnosticSeverity.WARNING,
                title="Running as root user",
                description="Running as root may cause permission issues with created files",
                suggestions=[
                    "Consider running as a regular user",
                    "Ensure proper file ownership after completion",
                    "Be careful with file permissions"
                ]
            ))
        
        return issues
    
    def _diagnose_manifest(self) -> List[DiagnosticIssue]:
        """Diagnose manifest-related issues"""
        issues = []
        
        if not self.manifest_path.exists():
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.MANIFEST,
                severity=DiagnosticSeverity.CRITICAL,
                title="Dataset manifest not found",
                description=f"Dataset manifest file not found at {self.manifest_path}",
                suggestions=[
                    "Ensure you're running from the correct directory",
                    "Check if the manifest file exists",
                    "Verify the manifest path is correct",
                    "Create a manifest file if needed"
                ],
                technical_details={"manifest_path": str(self.manifest_path.absolute())}
            ))
            return issues
        
        if self.manifest is None:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.MANIFEST,
                severity=DiagnosticSeverity.CRITICAL,
                title="Cannot load dataset manifest",
                description="Dataset manifest file exists but cannot be loaded",
                suggestions=[
                    "Check manifest file syntax (JSON format)",
                    "Verify file is not corrupted",
                    "Check file permissions",
                    "Validate manifest schema"
                ],
                technical_details={"manifest_path": str(self.manifest_path.absolute())}
            ))
            return issues
        
        # Validate manifest integrity
        try:
            validation_result = self.manifest.validate_manifest_integrity()
            
            if not validation_result.get("valid", False):
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.MANIFEST,
                    severity=DiagnosticSeverity.ERROR,
                    title="Invalid dataset manifest",
                    description="Dataset manifest failed validation",
                    suggestions=[
                        "Check manifest syntax and schema",
                        "Verify all required fields are present",
                        "Fix validation errors in the manifest"
                    ],
                    technical_details=validation_result
                ))
            
            if validation_result.get("warnings"):
                for warning in validation_result["warnings"]:
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.MANIFEST,
                        severity=DiagnosticSeverity.WARNING,
                        title="Manifest validation warning",
                        description=warning,
                        suggestions=[
                            "Review manifest configuration",
                            "Fix non-critical issues for better reliability"
                        ]
                    ))
        
        except Exception as e:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.MANIFEST,
                severity=DiagnosticSeverity.ERROR,
                title="Manifest validation failed",
                description=f"Error during manifest validation: {str(e)}",
                suggestions=[
                    "Check manifest file format",
                    "Verify file is not corrupted",
                    "Review manifest syntax"
                ],
                technical_details={"error": str(e)}
            ))
        
        return issues
    
    def _diagnose_registry(self) -> List[DiagnosticIssue]:
        """Diagnose registry-related issues"""
        issues = []
        
        if self.registry_manager is None:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.REGISTRY,
                severity=DiagnosticSeverity.ERROR,
                title="Cannot initialize registry manager",
                description="Registry manager could not be initialized",
                suggestions=[
                    "Check registry directory permissions",
                    "Verify registry path is correct",
                    "Ensure registry directory is writable"
                ],
                technical_details={"registry_path": str(self.registry_path.absolute())}
            ))
            return issues
        
        # Validate registry integrity
        try:
            validation_result = self.registry_manager.validate_registry_integrity()
            
            if not validation_result.get("valid", False):
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.REGISTRY,
                    severity=DiagnosticSeverity.ERROR,
                    title="Registry integrity issues",
                    description="Registry has integrity problems",
                    suggestions=[
                        "Run registry integrity check",
                        "Fix corrupted registry entries",
                        "Consider rebuilding the registry"
                    ],
                    technical_details=validation_result
                ))
            
            if validation_result.get("warnings"):
                for warning in validation_result["warnings"]:
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.REGISTRY,
                        severity=DiagnosticSeverity.WARNING,
                        title="Registry warning",
                        description=warning,
                        suggestions=[
                            "Review registry entries",
                            "Fix non-critical issues for better reliability"
                        ]
                    ))
        
        except Exception as e:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.REGISTRY,
                severity=DiagnosticSeverity.ERROR,
                title="Registry validation failed",
                description=f"Error during registry validation: {str(e)}",
                suggestions=[
                    "Check registry directory and files",
                    "Verify file permissions",
                    "Consider reinitializing the registry"
                ],
                technical_details={"error": str(e)}
            ))
        
        return issues
    
    def _diagnose_datasets(self) -> List[DiagnosticIssue]:
        """Diagnose dataset-specific issues"""
        issues = []
        
        if not self.manifest or not self.registry_manager:
            return issues  # Cannot diagnose without both components
        
        # Check each dataset
        for dataset_name in self.manifest.list_datasets():
            try:
                dataset_info = self.manifest.get_dataset_info(dataset_name)
                if not dataset_info:
                    continue
                
                # Check if dataset is in registry
                has_registry_entry = self.registry_manager.has_registry_entry(dataset_name)
                
                if not has_registry_entry:
                    issues.append(DiagnosticIssue(
                        category=DiagnosticCategory.CONFIGURATION,
                        severity=DiagnosticSeverity.INFO,
                        title=f"Dataset {dataset_name} not yet downloaded",
                        description=f"Dataset {dataset_name} is available in manifest but not yet downloaded",
                        dataset_name=dataset_name,
                        suggestions=[
                            f"Run 'dataset-cli fetch-all' to download all datasets",
                            f"Run dataset fetch for {dataset_name} specifically"
                        ]
                    ))
                else:
                    # Check registry entry status
                    registry_entry = self.registry_manager.get_registry_entry(dataset_name)
                    if registry_entry and not registry_entry.verification.is_valid:
                        issues.append(DiagnosticIssue(
                            category=DiagnosticCategory.VERIFICATION,
                            severity=DiagnosticSeverity.ERROR,
                            title=f"Dataset {dataset_name} verification failed",
                            description=f"Dataset {dataset_name} is downloaded but failed verification",
                            dataset_name=dataset_name,
                            suggestions=[
                                f"Re-download {dataset_name} with --force-refresh",
                                f"Check dataset source integrity",
                                f"Verify expected checksums in manifest"
                            ],
                            technical_details={
                                "sha256_match": registry_entry.verification.sha256_verified,
                                "size_match": registry_entry.verification.size_verified,
                                "schema_valid": registry_entry.verification.schema_verified
                            }
                        ))
                
                # Check dataset source URLs
                if dataset_info.sources:
                    for source_name, source_config in dataset_info.sources.items():
                        url = source_config.get("url")
                        if url:
                            # Basic URL validation
                            parsed = urllib.parse.urlparse(url)
                            if not parsed.scheme or not parsed.netloc:
                                issues.append(DiagnosticIssue(
                                    category=DiagnosticCategory.CONFIGURATION,
                                    severity=DiagnosticSeverity.ERROR,
                                    title=f"Invalid URL for {dataset_name}",
                                    description=f"Invalid URL in {source_name} source: {url}",
                                    dataset_name=dataset_name,
                                    suggestions=[
                                        "Check URL format in manifest",
                                        "Verify URL is complete and correct",
                                        "Test URL in web browser"
                                    ],
                                    technical_details={"source": source_name, "url": url}
                                ))
            
            except Exception as e:
                issues.append(DiagnosticIssue(
                    category=DiagnosticCategory.CONFIGURATION,
                    severity=DiagnosticSeverity.WARNING,
                    title=f"Error checking dataset {dataset_name}",
                    description=f"Could not check dataset {dataset_name}: {str(e)}",
                    dataset_name=dataset_name,
                    suggestions=[
                        "Check dataset configuration in manifest",
                        "Verify dataset name is correct"
                    ],
                    technical_details={"error": str(e)}
                ))
        
        return issues
    
    def _diagnose_configuration(self) -> List[DiagnosticIssue]:
        """Diagnose configuration issues"""
        issues = []
        
        # Check for common configuration problems
        
        # Check if running from correct directory
        expected_files = ["pipelines", "data", "requirements.txt"]
        missing_files = [f for f in expected_files if not Path(f).exists()]
        
        if missing_files:
            issues.append(DiagnosticIssue(
                category=DiagnosticCategory.CONFIGURATION,
                severity=DiagnosticSeverity.WARNING,
                title="May not be running from PBUF root directory",
                description=f"Expected files/directories not found: {', '.join(missing_files)}",
                suggestions=[
                    "Ensure you're running from the PBUF root directory",
                    "Check if the project structure is correct",
                    "Verify all required files are present"
                ],
                technical_details={"missing_files": missing_files, "current_dir": str(Path.cwd())}
            ))
        
        return issues
    
    def _generate_recommendations(self, issues: List[DiagnosticIssue]) -> List[str]:
        """Generate high-level recommendations based on issues"""
        recommendations = []
        
        # Count issues by category and severity
        critical_issues = [i for i in issues if i.severity == DiagnosticSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == DiagnosticSeverity.ERROR]
        
        if critical_issues:
            recommendations.append("🚨 Address critical issues before attempting dataset reproduction")
            
            # Specific recommendations for critical issues
            if any(i.category == DiagnosticCategory.ENVIRONMENT for i in critical_issues):
                recommendations.append("📋 Fix Python environment and dependency issues first")
            
            if any(i.category == DiagnosticCategory.NETWORK for i in critical_issues):
                recommendations.append("🌐 Resolve network connectivity issues before downloading datasets")
            
            if any(i.category == DiagnosticCategory.FILESYSTEM for i in critical_issues):
                recommendations.append("💾 Fix filesystem and permission issues before proceeding")
        
        elif error_issues:
            recommendations.append("⚠️ Resolve error-level issues for reliable reproduction")
            
            # Category-specific recommendations
            network_errors = [i for i in error_issues if i.category == DiagnosticCategory.NETWORK]
            if network_errors:
                recommendations.append("🔗 Check network connectivity and dataset source availability")
            
            verification_errors = [i for i in error_issues if i.category == DiagnosticCategory.VERIFICATION]
            if verification_errors:
                recommendations.append("✅ Re-download datasets with verification failures using --force-refresh")
        
        else:
            recommendations.append("✅ System appears ready for dataset reproduction")
        
        # General recommendations
        if len(issues) > 5:
            recommendations.append("📊 Consider running diagnostics again after fixing major issues")
        
        recommendations.append("📖 Use 'dataset-cli fetch-all --help' for detailed usage information")
        
        return recommendations
    
    def _generate_recovery_steps(self, issues: List[DiagnosticIssue]) -> List[str]:
        """Generate step-by-step recovery instructions"""
        recovery_steps = []
        
        # Prioritize steps by severity and category
        critical_issues = [i for i in issues if i.severity == DiagnosticSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == DiagnosticSeverity.ERROR]
        
        step_num = 1
        
        # Critical issues first
        for issue in critical_issues:
            if issue.category == DiagnosticCategory.ENVIRONMENT:
                recovery_steps.append(f"{step_num}. Fix Python environment: {issue.description}")
                step_num += 1
            elif issue.category == DiagnosticCategory.NETWORK:
                recovery_steps.append(f"{step_num}. Resolve network issue: {issue.description}")
                step_num += 1
            elif issue.category == DiagnosticCategory.FILESYSTEM:
                recovery_steps.append(f"{step_num}. Fix filesystem issue: {issue.description}")
                step_num += 1
        
        # Error issues next
        for issue in error_issues:
            if issue.category == DiagnosticCategory.MANIFEST:
                recovery_steps.append(f"{step_num}. Fix manifest: {issue.description}")
                step_num += 1
            elif issue.category == DiagnosticCategory.REGISTRY:
                recovery_steps.append(f"{step_num}. Fix registry: {issue.description}")
                step_num += 1
        
        # Final steps
        if not critical_issues:
            recovery_steps.append(f"{step_num}. Run 'dataset-cli fetch-all' to download all datasets")
            step_num += 1
            recovery_steps.append(f"{step_num}. Verify successful completion with 'dataset-cli reproduction-status'")
        
        return recovery_steps
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "current_directory": str(Path.cwd()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Git information
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info["git_commit"] = result.stdout.strip()
        except Exception:
            info["git_commit"] = None
        
        # Disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(Path.cwd())
            info["disk_space"] = {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3)
            }
        except Exception:
            info["disk_space"] = None
        
        # Registry stats
        if self.registry_manager:
            try:
                registry_summary = self.registry_manager.get_registry_summary()
                info["registry_stats"] = registry_summary
            except Exception:
                info["registry_stats"] = None
        
        # Manifest stats
        if self.manifest:
            try:
                info["manifest_stats"] = {
                    "total_datasets": len(self.manifest.list_datasets()),
                    "manifest_version": self.manifest.manifest_data.get("manifest_version", "unknown")
                }
            except Exception:
                info["manifest_stats"] = None
        
        return info
    
    def _extract_http_status(self, error_message: str) -> Optional[int]:
        """Extract HTTP status code from error message"""
        import re
        
        # Look for HTTP status codes in error message
        match = re.search(r'HTTP\s+(\d{3})', error_message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Look for just status codes
        match = re.search(r'\b(\d{3})\b', error_message)
        if match:
            code = int(match.group(1))
            if 100 <= code <= 599:  # Valid HTTP status code range
                return code
        
        return None