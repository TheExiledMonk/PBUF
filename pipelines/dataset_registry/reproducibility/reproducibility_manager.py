"""
Reproducibility manager for one-command dataset preparation

This module provides the ReproducibilityManager class for comprehensive
dataset fetching, verification, and reproduction diagnostics to enable
complete workflow reproducibility.
"""

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Set
import traceback

from ..core.manifest_schema import DatasetManifest
from ..core.registry_manager import RegistryManager
from ..integration.dataset_integration import DatasetRegistry


class ReproductionStatus(Enum):
    """Status of reproduction process"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    FETCHING = "fetching"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DatasetProgress:
    """Progress information for individual dataset"""
    name: str
    status: ReproductionStatus = ReproductionStatus.PENDING
    progress_percent: float = 0.0
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    verification_passed: bool = False
    source_used: Optional[str] = None
    file_size: Optional[int] = None
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Calculate elapsed time in seconds"""
        if self.start_time:
            end = self.end_time or time.time()
            return end - self.start_time
        return None


@dataclass
class ReproductionProgress:
    """Overall progress information for reproduction process"""
    total_datasets: int = 0
    completed_datasets: int = 0
    failed_datasets: int = 0
    cancelled_datasets: int = 0
    status: ReproductionStatus = ReproductionStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    datasets: Dict[str, DatasetProgress] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    @property
    def progress_percent(self) -> float:
        """Calculate overall progress percentage"""
        if self.total_datasets == 0:
            return 0.0
        return (self.completed_datasets / self.total_datasets) * 100
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Calculate elapsed time in seconds"""
        if self.start_time:
            end = self.end_time or time.time()
            return end - self.start_time
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if reproduction is complete"""
        return self.status in [ReproductionStatus.COMPLETED, ReproductionStatus.FAILED, ReproductionStatus.CANCELLED]


@dataclass
class ReproductionResult:
    """Final result of reproduction process"""
    success: bool
    total_datasets: int
    successful_datasets: List[str]
    failed_datasets: List[str]
    cancelled_datasets: List[str]
    total_time: float
    total_size: int
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    environment_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "summary": {
                "total_datasets": self.total_datasets,
                "successful_datasets": len(self.successful_datasets),
                "failed_datasets": len(self.failed_datasets),
                "cancelled_datasets": len(self.cancelled_datasets),
                "success_rate": len(self.successful_datasets) / self.total_datasets if self.total_datasets > 0 else 0.0,
                "total_time_seconds": self.total_time,
                "total_size_bytes": self.total_size
            },
            "datasets": {
                "successful": self.successful_datasets,
                "failed": self.failed_datasets,
                "cancelled": self.cancelled_datasets
            },
            "diagnostics": {
                "errors": self.errors,
                "warnings": self.warnings,
                "suggestions": self.suggestions
            },
            "environment": self.environment_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class ReproductionError(Exception):
    """Exception raised during reproduction process"""
    def __init__(self, message: str, dataset_name: Optional[str] = None, original_error: Optional[Exception] = None):
        self.dataset_name = dataset_name
        self.original_error = original_error
        super().__init__(message)


class ReproducibilityManager:
    """
    Manager for one-command dataset preparation and reproduction
    
    Provides comprehensive dataset fetching, verification, and diagnostics
    for complete workflow reproducibility.
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path] = "data/datasets_manifest.json",
        registry_path: Union[str, Path] = "data/registry",
        max_workers: int = 4,
        timeout_seconds: int = 300
    ):
        """
        Initialize reproducibility manager
        
        Args:
            manifest_path: Path to dataset manifest file
            registry_path: Path to registry directory
            max_workers: Maximum number of parallel download workers
            timeout_seconds: Timeout for individual dataset operations
        """
        self.manifest_path = Path(manifest_path)
        self.registry_path = Path(registry_path)
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        
        # Initialize components
        self.dataset_registry = DatasetRegistry(manifest_path, registry_path)
        self.manifest = self.dataset_registry.manifest
        self.registry_manager = self.dataset_registry.registry_manager
        
        # Progress tracking
        self._progress_lock = threading.Lock()
        self._cancellation_token = {'cancelled': False}
    
    def fetch_all_datasets(
        self,
        dataset_names: Optional[List[str]] = None,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[ReproductionProgress], None]] = None,
        parallel: bool = True
    ) -> ReproductionResult:
        """
        Fetch and verify all required datasets for complete workflow reproduction
        
        Args:
            dataset_names: List of dataset names to fetch (all if None)
            force_refresh: Force re-download even if datasets exist
            progress_callback: Optional callback for progress updates
            parallel: Enable parallel downloading
            
        Returns:
            ReproductionResult with comprehensive status and diagnostics
            
        Raises:
            ReproductionError: If critical errors occur during reproduction
        """
        # Initialize progress tracking
        if dataset_names is None:
            dataset_names = self.manifest.list_datasets()
        
        progress = ReproductionProgress(
            total_datasets=len(dataset_names),
            status=ReproductionStatus.INITIALIZING,
            start_time=time.time()
        )
        
        # Initialize dataset progress tracking
        for name in dataset_names:
            progress.datasets[name] = DatasetProgress(name=name)
        
        if progress_callback:
            progress_callback(progress)
        
        try:
            # Validate datasets exist in manifest
            self._validate_datasets(dataset_names, progress)
            
            # Update status to fetching
            progress.status = ReproductionStatus.FETCHING
            if progress_callback:
                progress_callback(progress)
            
            # Fetch datasets (parallel or sequential)
            if parallel and len(dataset_names) > 1:
                self._fetch_datasets_parallel(dataset_names, force_refresh, progress, progress_callback)
            else:
                self._fetch_datasets_sequential(dataset_names, force_refresh, progress, progress_callback)
            
            # Update status to verifying
            progress.status = ReproductionStatus.VERIFYING
            if progress_callback:
                progress_callback(progress)
            
            # Perform final verification
            self._verify_all_datasets(dataset_names, progress, progress_callback)
            
            # Complete reproduction
            progress.status = ReproductionStatus.COMPLETED
            progress.end_time = time.time()
            
            if progress_callback:
                progress_callback(progress)
            
            return self._create_result(progress)
            
        except Exception as e:
            progress.status = ReproductionStatus.FAILED
            progress.end_time = time.time()
            progress.error_message = str(e)
            
            if progress_callback:
                progress_callback(progress)
            
            # Create result even for failures to provide diagnostics
            result = self._create_result(progress)
            result.success = False
            result.errors.append(f"Reproduction failed: {str(e)}")
            
            return result
    
    def _validate_datasets(self, dataset_names: List[str], progress: ReproductionProgress) -> None:
        """Validate that all requested datasets exist in manifest"""
        missing_datasets = []
        
        for name in dataset_names:
            if not self.manifest.has_dataset(name):
                missing_datasets.append(name)
                progress.datasets[name].status = ReproductionStatus.FAILED
                progress.datasets[name].error_message = "Dataset not found in manifest"
        
        if missing_datasets:
            raise ReproductionError(f"Datasets not found in manifest: {missing_datasets}")
    
    def _fetch_datasets_parallel(
        self,
        dataset_names: List[str],
        force_refresh: bool,
        progress: ReproductionProgress,
        progress_callback: Optional[Callable[[ReproductionProgress], None]]
    ) -> None:
        """Fetch datasets in parallel using ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_dataset = {
                executor.submit(self._fetch_single_dataset, name, force_refresh, progress): name
                for name in dataset_names
            }
            
            # Process completed downloads
            for future in as_completed(future_to_dataset, timeout=self.timeout_seconds * len(dataset_names)):
                dataset_name = future_to_dataset[future]
                
                try:
                    # Get result (will raise exception if download failed)
                    future.result()
                    
                    with self._progress_lock:
                        progress.completed_datasets += 1
                        progress.datasets[dataset_name].status = ReproductionStatus.COMPLETED
                        progress.datasets[dataset_name].end_time = time.time()
                    
                except Exception as e:
                    with self._progress_lock:
                        progress.failed_datasets += 1
                        progress.datasets[dataset_name].status = ReproductionStatus.FAILED
                        progress.datasets[dataset_name].error_message = str(e)
                        progress.datasets[dataset_name].end_time = time.time()
                
                # Update progress
                if progress_callback:
                    with self._progress_lock:
                        progress_callback(progress)
                
                # Check for cancellation
                if self._cancellation_token.get('cancelled', False):
                    # Cancel remaining futures
                    for remaining_future in future_to_dataset:
                        if not remaining_future.done():
                            remaining_future.cancel()
                    break
    
    def _fetch_datasets_sequential(
        self,
        dataset_names: List[str],
        force_refresh: bool,
        progress: ReproductionProgress,
        progress_callback: Optional[Callable[[ReproductionProgress], None]]
    ) -> None:
        """Fetch datasets sequentially"""
        for dataset_name in dataset_names:
            if self._cancellation_token.get('cancelled', False):
                progress.datasets[dataset_name].status = ReproductionStatus.CANCELLED
                progress.cancelled_datasets += 1
                continue
            
            try:
                self._fetch_single_dataset(dataset_name, force_refresh, progress)
                progress.completed_datasets += 1
                progress.datasets[dataset_name].status = ReproductionStatus.COMPLETED
                progress.datasets[dataset_name].end_time = time.time()
                
            except Exception as e:
                progress.failed_datasets += 1
                progress.datasets[dataset_name].status = ReproductionStatus.FAILED
                progress.datasets[dataset_name].error_message = str(e)
                progress.datasets[dataset_name].end_time = time.time()
            
            # Update progress
            if progress_callback:
                progress_callback(progress)
    
    def _fetch_single_dataset(
        self,
        dataset_name: str,
        force_refresh: bool,
        progress: ReproductionProgress
    ) -> None:
        """Fetch a single dataset with error handling"""
        dataset_progress = progress.datasets[dataset_name]
        dataset_progress.start_time = time.time()
        dataset_progress.status = ReproductionStatus.FETCHING
        
        try:
            # Fetch dataset using registry
            dataset_info = self.dataset_registry.fetch_dataset(dataset_name, force_refresh)
            
            # Update progress with results
            dataset_progress.verification_passed = dataset_info.is_verified
            dataset_progress.source_used = dataset_info.provenance.source_used if dataset_info.provenance else None
            
            # Get file size
            if dataset_info.local_path.exists():
                dataset_progress.file_size = dataset_info.local_path.stat().st_size
            
            if not dataset_info.is_verified:
                raise ReproductionError(f"Dataset verification failed for {dataset_name}")
            
        except Exception as e:
            raise ReproductionError(f"Failed to fetch dataset {dataset_name}: {str(e)}", dataset_name, e)
    
    def _verify_all_datasets(
        self,
        dataset_names: List[str],
        progress: ReproductionProgress,
        progress_callback: Optional[Callable[[ReproductionProgress], None]]
    ) -> None:
        """Perform final verification of all datasets"""
        verification_failures = []
        
        for dataset_name in dataset_names:
            if progress.datasets[dataset_name].status != ReproductionStatus.COMPLETED:
                continue  # Skip failed datasets
            
            try:
                verification_result = self.dataset_registry.verify_dataset(dataset_name)
                if not verification_result.is_valid:
                    verification_failures.append(dataset_name)
                    progress.datasets[dataset_name].verification_passed = False
                    progress.datasets[dataset_name].error_message = "Final verification failed"
                
            except Exception as e:
                verification_failures.append(dataset_name)
                progress.datasets[dataset_name].verification_passed = False
                progress.datasets[dataset_name].error_message = f"Verification error: {str(e)}"
        
        if verification_failures:
            raise ReproductionError(f"Final verification failed for datasets: {verification_failures}")
    
    def _create_result(self, progress: ReproductionProgress) -> ReproductionResult:
        """Create final reproduction result from progress"""
        successful_datasets = []
        failed_datasets = []
        cancelled_datasets = []
        errors = []
        warnings = []
        suggestions = []
        total_size = 0
        
        # Analyze dataset results
        for name, dataset_progress in progress.datasets.items():
            if dataset_progress.status == ReproductionStatus.COMPLETED and dataset_progress.verification_passed:
                successful_datasets.append(name)
                if dataset_progress.file_size:
                    total_size += dataset_progress.file_size
            elif dataset_progress.status == ReproductionStatus.CANCELLED:
                cancelled_datasets.append(name)
            else:
                failed_datasets.append(name)
                if dataset_progress.error_message:
                    errors.append(f"{name}: {dataset_progress.error_message}")
        
        # Generate warnings and suggestions
        if failed_datasets:
            warnings.append(f"{len(failed_datasets)} datasets failed to download or verify")
            suggestions.extend(self._generate_failure_suggestions(failed_datasets, progress))
        
        if cancelled_datasets:
            warnings.append(f"{len(cancelled_datasets)} datasets were cancelled")
        
        # Collect environment information
        environment_info = self._collect_environment_info()
        
        return ReproductionResult(
            success=len(failed_datasets) == 0 and len(cancelled_datasets) == 0,
            total_datasets=progress.total_datasets,
            successful_datasets=successful_datasets,
            failed_datasets=failed_datasets,
            cancelled_datasets=cancelled_datasets,
            total_time=progress.elapsed_time or 0.0,
            total_size=total_size,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            environment_info=environment_info
        )
    
    def _generate_failure_suggestions(self, failed_datasets: List[str], progress: ReproductionProgress) -> List[str]:
        """Generate suggestions for handling failed datasets using diagnostics"""
        suggestions = []
        
        # Use diagnostics system for detailed analysis
        try:
            from .diagnostics import ReproductionDiagnostics
            
            diagnostics = ReproductionDiagnostics(
                manifest_path=self.dataset_registry.manifest_path,
                registry_path=self.dataset_registry.registry_path
            )
            
            # Analyze each failed dataset
            all_diagnostic_issues = []
            
            for dataset_name in failed_datasets:
                dataset_progress = progress.datasets[dataset_name]
                error_msg = dataset_progress.error_message or ""
                
                # Get diagnostic issues for this failure
                issues = diagnostics.diagnose_dataset_failure(
                    dataset_name, 
                    error_msg,
                    {
                        "status": dataset_progress.status.value,
                        "elapsed_time": dataset_progress.elapsed_time,
                        "source_used": dataset_progress.source_used
                    }
                )
                
                all_diagnostic_issues.extend(issues)
            
            # Extract suggestions from diagnostic issues
            for issue in all_diagnostic_issues:
                suggestions.extend(issue.suggestions[:2])  # Take first 2 suggestions per issue
            
            # Remove duplicates while preserving order
            seen = set()
            unique_suggestions = []
            for suggestion in suggestions:
                if suggestion not in seen:
                    seen.add(suggestion)
                    unique_suggestions.append(suggestion)
            
            suggestions = unique_suggestions
            
        except Exception:
            # Fallback to basic analysis if diagnostics fail
            # Analyze failure patterns
            network_failures = []
            verification_failures = []
            missing_datasets = []
            
            for dataset_name in failed_datasets:
                dataset_progress = progress.datasets[dataset_name]
                error_msg = dataset_progress.error_message or ""
                
                if "not found" in error_msg.lower() or "404" in error_msg:
                    missing_datasets.append(dataset_name)
                elif "verification" in error_msg.lower() or "checksum" in error_msg.lower():
                    verification_failures.append(dataset_name)
                elif "network" in error_msg.lower() or "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                    network_failures.append(dataset_name)
            
            # Generate specific suggestions
            if network_failures:
                suggestions.append(f"Network issues detected for {len(network_failures)} datasets. Check internet connection and try again.")
            
            if verification_failures:
                suggestions.append(f"Verification failures for {len(verification_failures)} datasets. Files may be corrupted - try re-downloading with --force-refresh.")
            
            if missing_datasets:
                suggestions.append(f"Missing datasets: {missing_datasets}. Check if URLs are still valid or contact data providers.")
        
        # Add general suggestions
        if len(failed_datasets) > 1:
            suggestions.append("For multiple failures, try running with --parallel=false for better error diagnostics.")
        
        suggestions.append("Use 'dataset-cli diagnostics' for comprehensive system analysis.")
        suggestions.append("Use 'dataset-cli status <dataset_name>' for detailed error information.")
        suggestions.append("Check the audit trail with 'dataset-cli audit' for historical issues.")
        
        return suggestions
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information for reproducibility"""
        import platform
        import sys
        import subprocess
        
        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Get PBUF commit hash
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                env_info["pbuf_commit"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            env_info["pbuf_commit"] = None
        
        # Get registry statistics
        try:
            registry_summary = self.registry_manager.get_registry_summary()
            env_info["registry_stats"] = {
                "total_datasets": registry_summary["total_datasets"],
                "verified_datasets": registry_summary["verified_datasets"],
                "failed_datasets": registry_summary["failed_datasets"]
            }
        except Exception:
            env_info["registry_stats"] = None
        
        return env_info
    
    def cancel_reproduction(self) -> None:
        """Cancel ongoing reproduction process"""
        self._cancellation_token['cancelled'] = True
    
    def get_reproduction_status(self) -> Dict[str, Any]:
        """Get current status of reproduction system"""
        try:
            # Get registry summary
            registry_summary = self.registry_manager.get_registry_summary()
            
            # Get manifest info
            manifest_datasets = self.manifest.list_datasets()
            
            # Check which datasets are ready for reproduction
            ready_datasets = []
            missing_datasets = []
            
            for dataset_name in manifest_datasets:
                if self.registry_manager.has_registry_entry(dataset_name):
                    entry = self.registry_manager.get_registry_entry(dataset_name)
                    if entry and entry.verification.is_valid:
                        ready_datasets.append(dataset_name)
                    else:
                        missing_datasets.append(dataset_name)
                else:
                    missing_datasets.append(dataset_name)
            
            return {
                "system_status": "ready" if len(missing_datasets) == 0 else "incomplete",
                "total_datasets_available": len(manifest_datasets),
                "datasets_ready": len(ready_datasets),
                "datasets_missing": len(missing_datasets),
                "ready_datasets": ready_datasets,
                "missing_datasets": missing_datasets,
                "registry_summary": registry_summary,
                "environment": self._collect_environment_info()
            }
            
        except Exception as e:
            return {
                "system_status": "error",
                "error": str(e),
                "environment": self._collect_environment_info()
            }
    
    def prepare_reproduction_environment(
        self,
        output_dir: Optional[Path] = None,
        include_logs: bool = True,
        include_provenance: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare complete reproduction environment with all necessary files
        
        Args:
            output_dir: Directory to prepare reproduction files (current dir if None)
            include_logs: Include audit logs and verification history
            include_provenance: Include complete provenance records
            
        Returns:
            Dictionary with preparation results and file locations
        """
        if output_dir is None:
            output_dir = Path.cwd() / "reproduction_bundle"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        preparation_result = {
            "success": True,
            "output_directory": str(output_dir),
            "files_created": [],
            "errors": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Copy manifest file
            manifest_dest = output_dir / "datasets_manifest.json"
            if self.manifest_path.exists():
                import shutil
                shutil.copy2(self.manifest_path, manifest_dest)
                preparation_result["files_created"].append(str(manifest_dest))
            
            # Export registry summary
            registry_summary = self.registry_manager.export_provenance_summary()
            registry_file = output_dir / "registry_summary.json"
            with open(registry_file, 'w') as f:
                json.dump(registry_summary, f, indent=2)
            preparation_result["files_created"].append(str(registry_file))
            
            # Export reproduction script
            script_content = self._generate_reproduction_script()
            script_file = output_dir / "reproduce_datasets.py"
            with open(script_file, 'w') as f:
                f.write(script_content)
            preparation_result["files_created"].append(str(script_file))
            
            # Include audit logs if requested
            if include_logs:
                audit_entries = self.registry_manager.get_audit_trail()
                if audit_entries:
                    audit_file = output_dir / "audit_trail.jsonl"
                    with open(audit_file, 'w') as f:
                        for entry in audit_entries:
                            json.dump(entry, f)
                            f.write('\n')
                    preparation_result["files_created"].append(str(audit_file))
            
            # Include provenance records if requested
            if include_provenance:
                provenance_dir = output_dir / "provenance"
                provenance_dir.mkdir(exist_ok=True)
                
                for dataset_name in self.registry_manager.list_datasets():
                    entry = self.registry_manager.get_registry_entry(dataset_name)
                    if entry:
                        provenance_file = provenance_dir / f"{dataset_name}_provenance.json"
                        with open(provenance_file, 'w') as f:
                            json.dump(entry.to_dict(), f, indent=2)
                        preparation_result["files_created"].append(str(provenance_file))
            
            # Create README
            readme_content = self._generate_reproduction_readme()
            readme_file = output_dir / "README.md"
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            preparation_result["files_created"].append(str(readme_file))
            
        except Exception as e:
            preparation_result["success"] = False
            preparation_result["errors"].append(str(e))
        
        return preparation_result
    
    def _generate_reproduction_script(self) -> str:
        """Generate Python script for reproducing datasets"""
        return '''#!/usr/bin/env python3
"""
Dataset Reproduction Script

This script reproduces all datasets required for PBUF workflow execution.
Generated automatically by the dataset registry reproducibility system.
"""

import sys
from pathlib import Path

# Add PBUF to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.dataset_registry.reproducibility import fetch_all_datasets


def main():
    """Main reproduction function"""
    print("Starting dataset reproduction...")
    
    # Fetch all datasets
    result = fetch_all_datasets(
        force_refresh=False,
        parallel=True,
        progress_callback=print_progress
    )
    
    # Print results
    if result.success:
        print(f"\\n✓ Reproduction completed successfully!")
        print(f"  - {len(result.successful_datasets)} datasets ready")
        print(f"  - Total size: {result.total_size / 1024 / 1024:.1f} MB")
        print(f"  - Total time: {result.total_time:.1f} seconds")
    else:
        print(f"\\n✗ Reproduction failed!")
        print(f"  - {len(result.successful_datasets)} datasets succeeded")
        print(f"  - {len(result.failed_datasets)} datasets failed")
        
        if result.suggestions:
            print("\\nSuggestions:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")
        
        sys.exit(1)


def print_progress(progress):
    """Print progress updates"""
    print(f"Progress: {progress.progress_percent:.1f}% "
          f"({progress.completed_datasets}/{progress.total_datasets} datasets)")


if __name__ == "__main__":
    main()
'''
    
    def _generate_reproduction_readme(self) -> str:
        """Generate README for reproduction bundle"""
        return '''# Dataset Reproduction Bundle

This bundle contains all necessary files to reproduce the PBUF dataset environment.

## Contents

- `datasets_manifest.json` - Dataset definitions and sources
- `registry_summary.json` - Current registry state and provenance
- `reproduce_datasets.py` - Automated reproduction script
- `audit_trail.jsonl` - Complete audit history (if included)
- `provenance/` - Individual dataset provenance records (if included)

## Usage

### Quick Reproduction

Run the automated script:

```bash
python reproduce_datasets.py
```

### Manual Reproduction

Use the dataset registry CLI:

```bash
# Fetch all datasets
python -m pipelines.dataset_registry.cli fetch-all

# Check status
python -m pipelines.dataset_registry.cli list --show-details

# Verify all datasets
python -m pipelines.dataset_registry.cli verify-all
```

### Programmatic Usage

```python
from pipelines.dataset_registry.reproducibility import fetch_all_datasets

# Fetch all datasets with progress reporting
result = fetch_all_datasets(
    force_refresh=False,
    parallel=True,
    progress_callback=lambda p: print(f"Progress: {p.progress_percent:.1f}%")
)

if result.success:
    print("All datasets ready for analysis!")
else:
    print("Some datasets failed - check result.errors for details")
```

## Requirements

- Python 3.7+
- Internet connection (for downloading datasets)
- Sufficient disk space (check registry_summary.json for size estimates)

## Troubleshooting

If reproduction fails:

1. Check internet connectivity
2. Verify disk space availability
3. Review error messages in the output
4. Try with `force_refresh=True` to re-download corrupted files
5. Use `parallel=False` for better error diagnostics

For detailed diagnostics, use:

```bash
python -m pipelines.dataset_registry.cli status <dataset_name>
python -m pipelines.dataset_registry.cli integrity
```
'''


# Convenience functions for external use
def fetch_all_datasets(
    dataset_names: Optional[List[str]] = None,
    force_refresh: bool = False,
    progress_callback: Optional[Callable[[ReproductionProgress], None]] = None,
    parallel: bool = True,
    manifest_path: Union[str, Path] = "data/datasets_manifest.json",
    registry_path: Union[str, Path] = "data/registry"
) -> ReproductionResult:
    """
    Convenience function for fetching all datasets
    
    Args:
        dataset_names: List of dataset names to fetch (all if None)
        force_refresh: Force re-download even if datasets exist
        progress_callback: Optional callback for progress updates
        parallel: Enable parallel downloading
        manifest_path: Path to dataset manifest file
        registry_path: Path to registry directory
        
    Returns:
        ReproductionResult with comprehensive status and diagnostics
    """
    manager = ReproducibilityManager(manifest_path, registry_path)
    return manager.fetch_all_datasets(dataset_names, force_refresh, progress_callback, parallel)


def prepare_reproduction_environment(
    output_dir: Optional[Path] = None,
    include_logs: bool = True,
    include_provenance: bool = True,
    manifest_path: Union[str, Path] = "data/datasets_manifest.json",
    registry_path: Union[str, Path] = "data/registry"
) -> Dict[str, Any]:
    """
    Convenience function for preparing reproduction environment
    
    Args:
        output_dir: Directory to prepare reproduction files
        include_logs: Include audit logs and verification history
        include_provenance: Include complete provenance records
        manifest_path: Path to dataset manifest file
        registry_path: Path to registry directory
        
    Returns:
        Dictionary with preparation results and file locations
    """
    manager = ReproducibilityManager(manifest_path, registry_path)
    return manager.prepare_reproduction_environment(output_dir, include_logs, include_provenance)