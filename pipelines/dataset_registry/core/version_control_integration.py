"""
Version control integration for dataset registry

This module provides integration with PBUF version control system including
automatic commit hash recording, environment fingerprinting, and version-aware
dataset compatibility checking.
"""

import os
import subprocess
import sys
import platform
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class EnvironmentFingerprint:
    """Complete environment fingerprint for reproducibility"""
    pbuf_commit: Optional[str]
    pbuf_branch: Optional[str]
    pbuf_remote_url: Optional[str]
    pbuf_status: str  # clean, dirty, unknown
    python_version: str
    python_executable: str
    platform: str
    hostname: str
    working_directory: str
    environment_variables: Dict[str, str]
    installed_packages: Dict[str, str]
    timestamp: str
    fingerprint_hash: str
    
    def __post_init__(self):
        """Calculate fingerprint hash after initialization"""
        if not self.fingerprint_hash:
            self.fingerprint_hash = self._calculate_fingerprint_hash()
    
    def _calculate_fingerprint_hash(self) -> str:
        """Calculate SHA256 hash of environment fingerprint"""
        # Create deterministic string representation
        fingerprint_data = {
            "pbuf_commit": self.pbuf_commit,
            "pbuf_branch": self.pbuf_branch,
            "python_version": self.python_version,
            "platform": self.platform,
            "working_directory": self.working_directory,
            # Include only essential environment variables
            "essential_env": {k: v for k, v in self.environment_variables.items() 
                            if k in ["PATH", "PYTHONPATH", "HOME", "USER"]}
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()


@dataclass
class VersionCompatibility:
    """Version compatibility information"""
    is_compatible: bool
    compatibility_level: str  # exact, compatible, warning, incompatible
    dataset_environment: EnvironmentFingerprint
    current_environment: EnvironmentFingerprint
    compatibility_issues: List[str]
    compatibility_warnings: List[str]
    recommendations: List[str]


class VersionControlIntegration:
    """
    Integration with PBUF version control system
    
    Provides automatic PBUF commit hash recording, environment fingerprinting,
    and version-aware dataset compatibility checking.
    """
    
    def __init__(self, repository_path: Optional[Path] = None):
        """
        Initialize version control integration
        
        Args:
            repository_path: Path to PBUF repository (defaults to current directory)
        """
        self.repository_path = repository_path or Path.cwd()
        self._git_available = self._check_git_availability()
        self._current_environment = None
    
    def _check_git_availability(self) -> bool:
        """Check if git is available and repository exists"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_pbuf_commit_info(self) -> Dict[str, Optional[str]]:
        """
        Get current PBUF git commit information
        
        Returns:
            Dictionary with commit hash, branch, and status information
        """
        commit_info = {
            "commit_hash": None,
            "branch": None,
            "remote_url": None,
            "status": "unknown"
        }
        
        if not self._git_available:
            return commit_info
        
        try:
            # Get commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                commit_info["commit_hash"] = result.stdout.strip()
            
            # Get branch name
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                commit_info["branch"] = result.stdout.strip()
            
            # Get remote URL
            result = subprocess.run(
                ['git', 'config', '--get', 'remote.origin.url'],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                commit_info["remote_url"] = result.stdout.strip()
            
            # Check if repository is clean
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repository_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                commit_info["status"] = "clean" if not result.stdout.strip() else "dirty"
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return commit_info
    
    def get_python_environment_info(self) -> Dict[str, Any]:
        """
        Get Python environment information
        
        Returns:
            Dictionary with Python version, executable, and package information
        """
        env_info = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "python_path": sys.path.copy(),
            "installed_packages": {}
        }
        
        # Get installed packages (basic version)
        try:
            import pkg_resources
            installed_packages = {}
            for dist in pkg_resources.working_set:
                installed_packages[dist.project_name] = dist.version
            env_info["installed_packages"] = installed_packages
        except ImportError:
            # Fallback to pip list if pkg_resources not available
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'list', '--format=json'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    packages = json.loads(result.stdout)
                    env_info["installed_packages"] = {
                        pkg["name"]: pkg["version"] for pkg in packages
                    }
            except (subprocess.TimeoutExpired, json.JSONDecodeError):
                pass
        
        return env_info
    
    def get_system_environment_info(self) -> Dict[str, Any]:
        """
        Get system environment information
        
        Returns:
            Dictionary with platform, hostname, and environment variables
        """
        # Get essential environment variables
        essential_env_vars = [
            "PATH", "PYTHONPATH", "HOME", "USER", "USERNAME", 
            "CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "PWD"
        ]
        
        env_vars = {}
        for var in essential_env_vars:
            value = os.environ.get(var)
            if value:
                env_vars[var] = value
        
        return {
            "platform": platform.platform(),
            "hostname": platform.node(),
            "working_directory": str(Path.cwd()),
            "environment_variables": env_vars
        }
    
    def collect_environment_fingerprint(self) -> EnvironmentFingerprint:
        """
        Collect complete environment fingerprint
        
        Returns:
            EnvironmentFingerprint with all environment information
        """
        commit_info = self.get_pbuf_commit_info()
        python_info = self.get_python_environment_info()
        system_info = self.get_system_environment_info()
        
        fingerprint = EnvironmentFingerprint(
            pbuf_commit=commit_info["commit_hash"],
            pbuf_branch=commit_info["branch"],
            pbuf_remote_url=commit_info["remote_url"],
            pbuf_status=commit_info["status"],
            python_version=python_info["python_version"],
            python_executable=python_info["python_executable"],
            platform=system_info["platform"],
            hostname=system_info["hostname"],
            working_directory=system_info["working_directory"],
            environment_variables=system_info["environment_variables"],
            installed_packages=python_info["installed_packages"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            fingerprint_hash=""  # Will be calculated in __post_init__
        )
        
        return fingerprint
    
    def get_current_environment(self) -> EnvironmentFingerprint:
        """
        Get current environment fingerprint (cached)
        
        Returns:
            Current EnvironmentFingerprint
        """
        if self._current_environment is None:
            self._current_environment = self.collect_environment_fingerprint()
        return self._current_environment
    
    def check_dataset_compatibility(
        self, 
        dataset_environment: EnvironmentFingerprint,
        strict_mode: bool = False
    ) -> VersionCompatibility:
        """
        Check compatibility between dataset environment and current environment
        
        Args:
            dataset_environment: Environment fingerprint from dataset registration
            strict_mode: If True, require exact environment match
            
        Returns:
            VersionCompatibility with detailed compatibility information
        """
        current_env = self.get_current_environment()
        
        compatibility_issues = []
        compatibility_warnings = []
        recommendations = []
        
        # Check PBUF commit compatibility
        if dataset_environment.pbuf_commit != current_env.pbuf_commit:
            if strict_mode:
                compatibility_issues.append(
                    f"PBUF commit mismatch: dataset uses {dataset_environment.pbuf_commit}, "
                    f"current is {current_env.pbuf_commit}"
                )
            else:
                compatibility_warnings.append(
                    f"PBUF commit difference: dataset uses {dataset_environment.pbuf_commit}, "
                    f"current is {current_env.pbuf_commit}"
                )
                recommendations.append("Consider checking for dataset compatibility with current PBUF version")
        
        # Check Python version compatibility
        if dataset_environment.python_version != current_env.python_version:
            if strict_mode:
                compatibility_issues.append(
                    f"Python version mismatch: dataset uses {dataset_environment.python_version}, "
                    f"current is {current_env.python_version}"
                )
            else:
                compatibility_warnings.append(
                    f"Python version difference: dataset uses {dataset_environment.python_version}, "
                    f"current is {current_env.python_version}"
                )
        
        # Check platform compatibility
        if dataset_environment.platform != current_env.platform:
            compatibility_warnings.append(
                f"Platform difference: dataset registered on {dataset_environment.platform}, "
                f"current is {current_env.platform}"
            )
        
        # Check critical package versions
        critical_packages = ["numpy", "scipy", "pandas", "astropy"]
        for package in critical_packages:
            dataset_version = dataset_environment.installed_packages.get(package)
            current_version = current_env.installed_packages.get(package)
            
            if dataset_version and current_version and dataset_version != current_version:
                if strict_mode:
                    compatibility_issues.append(
                        f"Package version mismatch for {package}: "
                        f"dataset uses {dataset_version}, current is {current_version}"
                    )
                else:
                    compatibility_warnings.append(
                        f"Package version difference for {package}: "
                        f"dataset uses {dataset_version}, current is {current_version}"
                    )
        
        # Determine compatibility level
        if compatibility_issues:
            compatibility_level = "incompatible"
            is_compatible = False
        elif compatibility_warnings:
            compatibility_level = "warning"
            is_compatible = True
        elif dataset_environment.fingerprint_hash == current_env.fingerprint_hash:
            compatibility_level = "exact"
            is_compatible = True
        else:
            compatibility_level = "compatible"
            is_compatible = True
        
        return VersionCompatibility(
            is_compatible=is_compatible,
            compatibility_level=compatibility_level,
            dataset_environment=dataset_environment,
            current_environment=current_env,
            compatibility_issues=compatibility_issues,
            compatibility_warnings=compatibility_warnings,
            recommendations=recommendations
        )
    
    def get_datasets_by_commit(self, commit_hash: str) -> List[str]:
        """
        Get datasets registered with a specific PBUF commit
        
        Args:
            commit_hash: PBUF git commit hash
            
        Returns:
            List of dataset names registered with that commit
        """
        # This would integrate with the registry manager
        from .registry_manager import RegistryManager
        
        registry_manager = RegistryManager("data/registry")
        return registry_manager.get_datasets_by_commit(commit_hash)
    
    def get_commit_history_for_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Get commit history for a dataset (all versions registered)
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of commit information for all versions of the dataset
        """
        from .registry_manager import RegistryManager
        
        registry_manager = RegistryManager("data/registry")
        audit_trail = registry_manager.get_audit_trail(dataset_name)
        
        commit_history = []
        for entry in audit_trail:
            if "pbuf_commit" in entry:
                commit_history.append({
                    "timestamp": entry["timestamp"],
                    "event": entry["event"],
                    "pbuf_commit": entry["pbuf_commit"],
                    "agent": entry.get("agent", "unknown")
                })
        
        return commit_history
    
    def validate_reproducibility_requirements(self, dataset_names: List[str]) -> Dict[str, Any]:
        """
        Validate that all datasets meet reproducibility requirements
        
        Args:
            dataset_names: List of dataset names to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "total_datasets": len(dataset_names),
            "datasets_with_commit": 0,
            "datasets_without_commit": 0,
            "environment_consistency": True,
            "issues": [],
            "warnings": [],
            "dataset_details": {}
        }
        
        from .registry_manager import RegistryManager
        registry_manager = RegistryManager("data/registry")
        
        environments = {}
        
        for dataset_name in dataset_names:
            try:
                registry_entry = registry_manager.get_registry_entry(dataset_name)
                if registry_entry:
                    pbuf_commit = registry_entry.environment.pbuf_commit
                    
                    if pbuf_commit:
                        validation_results["datasets_with_commit"] += 1
                        
                        # Track environment fingerprints
                        env_key = f"{pbuf_commit}_{registry_entry.environment.python_version}"
                        if env_key not in environments:
                            environments[env_key] = []
                        environments[env_key].append(dataset_name)
                        
                        validation_results["dataset_details"][dataset_name] = {
                            "has_commit": True,
                            "pbuf_commit": pbuf_commit,
                            "python_version": registry_entry.environment.python_version,
                            "platform": registry_entry.environment.platform,
                            "timestamp": registry_entry.download_timestamp
                        }
                    else:
                        validation_results["datasets_without_commit"] += 1
                        validation_results["issues"].append(
                            f"Dataset '{dataset_name}' missing PBUF commit hash"
                        )
                        validation_results["dataset_details"][dataset_name] = {
                            "has_commit": False,
                            "error": "Missing PBUF commit hash"
                        }
                else:
                    validation_results["issues"].append(
                        f"Dataset '{dataset_name}' not found in registry"
                    )
                    validation_results["dataset_details"][dataset_name] = {
                        "has_commit": False,
                        "error": "Not found in registry"
                    }
            except Exception as e:
                validation_results["issues"].append(
                    f"Failed to validate dataset '{dataset_name}': {str(e)}"
                )
                validation_results["dataset_details"][dataset_name] = {
                    "has_commit": False,
                    "error": str(e)
                }
        
        # Check environment consistency
        if len(environments) > 1:
            validation_results["environment_consistency"] = False
            validation_results["warnings"].append(
                f"Datasets registered with {len(environments)} different environments"
            )
            for env_key, datasets in environments.items():
                validation_results["warnings"].append(
                    f"Environment {env_key}: {', '.join(datasets)}"
                )
        
        # Overall validation
        if validation_results["issues"] or validation_results["datasets_without_commit"] > 0:
            validation_results["valid"] = False
        
        return validation_results
    
    def export_environment_summary(self) -> Dict[str, Any]:
        """
        Export current environment summary for documentation
        
        Returns:
            Dictionary with environment summary for publication materials
        """
        current_env = self.get_current_environment()
        
        return {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "pbuf_version": {
                "commit_hash": current_env.pbuf_commit,
                "branch": current_env.pbuf_branch,
                "remote_url": current_env.pbuf_remote_url,
                "status": current_env.pbuf_status
            },
            "python_environment": {
                "version": current_env.python_version,
                "executable": current_env.python_executable
            },
            "system_environment": {
                "platform": current_env.platform,
                "hostname": current_env.hostname,
                "working_directory": current_env.working_directory
            },
            "environment_fingerprint": current_env.fingerprint_hash,
            "critical_packages": {
                name: version for name, version in current_env.installed_packages.items()
                if name in ["numpy", "scipy", "pandas", "astropy", "matplotlib", "requests"]
            }
        }
    
    def create_reproducibility_manifest(self, dataset_names: List[str]) -> Dict[str, Any]:
        """
        Create reproducibility manifest for a set of datasets
        
        Args:
            dataset_names: List of dataset names to include
            
        Returns:
            Dictionary with complete reproducibility information
        """
        manifest = {
            "manifest_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "current_environment": self.export_environment_summary(),
            "datasets": {},
            "reproducibility_summary": {}
        }
        
        validation_results = self.validate_reproducibility_requirements(dataset_names)
        manifest["reproducibility_summary"] = validation_results
        
        # Add detailed dataset information
        from .registry_manager import RegistryManager
        registry_manager = RegistryManager("data/registry")
        
        for dataset_name in dataset_names:
            try:
                registry_entry = registry_manager.get_registry_entry(dataset_name)
                if registry_entry:
                    manifest["datasets"][dataset_name] = {
                        "provenance": asdict(registry_entry),
                        "compatibility": asdict(self.check_dataset_compatibility(
                            EnvironmentFingerprint(**asdict(registry_entry.environment))
                        ))
                    }
            except Exception as e:
                manifest["datasets"][dataset_name] = {
                    "error": str(e)
                }
        
        return manifest


# Convenience functions

def get_current_pbuf_commit() -> Optional[str]:
    """Get current PBUF commit hash"""
    vc_integration = VersionControlIntegration()
    commit_info = vc_integration.get_pbuf_commit_info()
    return commit_info["commit_hash"]


def collect_current_environment() -> EnvironmentFingerprint:
    """Collect current environment fingerprint"""
    vc_integration = VersionControlIntegration()
    return vc_integration.collect_environment_fingerprint()


def check_dataset_compatibility(dataset_name: str, strict_mode: bool = False) -> VersionCompatibility:
    """Check compatibility of a dataset with current environment"""
    from .registry_manager import RegistryManager
    
    registry_manager = RegistryManager("data/registry")
    registry_entry = registry_manager.get_registry_entry(dataset_name)
    
    if not registry_entry:
        raise ValueError(f"Dataset '{dataset_name}' not found in registry")
    
    vc_integration = VersionControlIntegration()
    dataset_env = EnvironmentFingerprint(**asdict(registry_entry.environment))
    
    return vc_integration.check_dataset_compatibility(dataset_env, strict_mode)


def validate_reproducibility(dataset_names: List[str]) -> Dict[str, Any]:
    """Validate reproducibility requirements for datasets"""
    vc_integration = VersionControlIntegration()
    return vc_integration.validate_reproducibility_requirements(dataset_names)


def create_reproducibility_report(dataset_names: List[str]) -> Dict[str, Any]:
    """Create complete reproducibility report"""
    vc_integration = VersionControlIntegration()
    return vc_integration.create_reproducibility_manifest(dataset_names)