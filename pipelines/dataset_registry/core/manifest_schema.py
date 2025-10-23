"""
Dataset manifest schema definition and validation

This module defines the JSON schema for dataset manifests and provides
validation and parsing functionality.
"""

import json
import jsonschema
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


# JSON Schema for dataset manifest validation
MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["manifest_version", "datasets"],
    "properties": {
        "manifest_version": {
            "type": "string",
            "pattern": "^[0-9]+\\.[0-9]+$"
        },
        "datasets": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9_-]+$": {
                    "type": "object",
                    "required": ["canonical_name", "description", "citation", "sources", "verification"],
                    "properties": {
                        "canonical_name": {"type": "string"},
                        "description": {"type": "string"},
                        "citation": {"type": "string"},
                        "license": {"type": "string"},
                        "sources": {
                            "type": "object",
                            "required": ["primary"],
                            "properties": {
                                "primary": {"$ref": "#/definitions/source"},
                                "mirror": {"$ref": "#/definitions/source"}
                            },
                            "additionalProperties": {"$ref": "#/definitions/source"}
                        },
                        "verification": {
                            "type": "object",
                            "required": ["sha256"],
                            "properties": {
                                "sha256": {"type": "string", "pattern": "^[a-fA-F0-9]{64}$"},
                                "size_bytes": {"type": "integer", "minimum": 0},
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "format": {"type": "string"},
                                        "columns": {"type": "array", "items": {"type": "string"}},
                                        "expected_rows": {"type": "integer", "minimum": 0}
                                    }
                                }
                            }
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "dataset_type": {"type": "string"},
                                "redshift_range": {"type": "array", "items": {"type": "number"}},
                                "observables": {"type": "array", "items": {"type": "string"}},
                                "n_data_points": {"type": "integer", "minimum": 0},
                                "covariance_included": {"type": "boolean"}
                            }
                        }
                    }
                }
            }
        }
    },
    "definitions": {
        "source": {
            "type": "object",
            "required": ["url", "protocol"],
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "protocol": {"type": "string", "enum": ["https", "zenodo", "arxiv", "local", "manual"]},
                "extraction": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "string", "enum": ["zip", "tar", "tar.gz", "tar.bz2"]},
                        "target_files": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        }
    }
}


@dataclass
class DatasetInfo:
    """Information about a dataset from the manifest"""
    name: str
    canonical_name: str
    description: str
    citation: str
    license: Optional[str]
    sources: Dict[str, Dict[str, Any]]
    verification: Dict[str, Any]
    metadata: Optional[Dict[str, Any]]
    local_path: Optional[Path] = None
    verification_status: Optional[str] = None
    last_verified: Optional[datetime] = None


class ManifestValidationError(Exception):
    """Raised when manifest validation fails"""
    def __init__(self, message: str, errors: List[str] = None):
        self.message = message
        self.errors = errors or []
        super().__init__(message)


class DatasetManifest:
    """
    Dataset manifest parser and validator
    
    Manages the central dataset definition file with schema validation,
    dataset lookup, and manifest versioning support.
    """
    
    def __init__(self, manifest_path: Union[str, Path]):
        """
        Initialize manifest with validation
        
        Args:
            manifest_path: Path to the manifest JSON file
            
        Raises:
            ManifestValidationError: If manifest is invalid
            FileNotFoundError: If manifest file doesn't exist
        """
        self.manifest_path = Path(manifest_path)
        self._manifest_data = None
        self._load_and_validate()
    
    def _load_and_validate(self):
        """Load and validate the manifest file"""
        try:
            with open(self.manifest_path, 'r') as f:
                self._manifest_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        except json.JSONDecodeError as e:
            raise ManifestValidationError(f"Invalid JSON in manifest: {e}")
        
        # Validate against schema
        try:
            jsonschema.validate(self._manifest_data, MANIFEST_SCHEMA)
        except jsonschema.ValidationError as e:
            errors = [str(e)]
            # Collect all validation errors
            validator = jsonschema.Draft7Validator(MANIFEST_SCHEMA)
            all_errors = [error.message for error in validator.iter_errors(self._manifest_data)]
            raise ManifestValidationError("Manifest validation failed", all_errors)
    
    @property
    def version(self) -> str:
        """Get manifest version"""
        return self._manifest_data["manifest_version"]
    
    def list_datasets(self) -> List[str]:
        """Get list of all dataset names in manifest"""
        return list(self._manifest_data["datasets"].keys())
    
    def get_dataset_info(self, name: str) -> DatasetInfo:
        """
        Get dataset information by name
        
        Args:
            name: Dataset name
            
        Returns:
            DatasetInfo object with dataset metadata
            
        Raises:
            KeyError: If dataset not found in manifest
        """
        if name not in self._manifest_data["datasets"]:
            available = ", ".join(self.list_datasets())
            raise KeyError(f"Dataset '{name}' not found in manifest. Available: {available}")
        
        dataset_data = self._manifest_data["datasets"][name]
        
        return DatasetInfo(
            name=name,
            canonical_name=dataset_data["canonical_name"],
            description=dataset_data["description"],
            citation=dataset_data["citation"],
            license=dataset_data.get("license"),
            sources=dataset_data["sources"],
            verification=dataset_data["verification"],
            metadata=dataset_data.get("metadata")
        )
    
    def has_dataset(self, name: str) -> bool:
        """Check if dataset exists in manifest"""
        return name in self._manifest_data["datasets"]
    
    def validate_dataset_name(self, name: str) -> bool:
        """
        Validate dataset name format
        
        Args:
            name: Dataset name to validate
            
        Returns:
            True if name is valid format
        """
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))
    
    def get_datasets_by_type(self, dataset_type: str) -> List[str]:
        """
        Get datasets filtered by type
        
        Args:
            dataset_type: Type to filter by (e.g., 'cmb', 'bao', 'sn')
            
        Returns:
            List of dataset names matching the type
        """
        matching_datasets = []
        for name, data in self._manifest_data["datasets"].items():
            metadata = data.get("metadata", {})
            if metadata.get("dataset_type") == dataset_type:
                matching_datasets.append(name)
        return matching_datasets
    
    def export_summary(self) -> Dict[str, Any]:
        """
        Export manifest summary for publication materials
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            "manifest_version": self.version,
            "total_datasets": len(self.list_datasets()),
            "datasets_by_type": {},
            "datasets": []
        }
        
        # Count by type
        for name in self.list_datasets():
            dataset_info = self.get_dataset_info(name)
            if dataset_info.metadata:
                dataset_type = dataset_info.metadata.get("dataset_type", "unknown")
                summary["datasets_by_type"][dataset_type] = summary["datasets_by_type"].get(dataset_type, 0) + 1
            
            # Add dataset summary
            summary["datasets"].append({
                "name": name,
                "canonical_name": dataset_info.canonical_name,
                "citation": dataset_info.citation,
                "type": dataset_info.metadata.get("dataset_type") if dataset_info.metadata else None
            })
        
        return summary
    
    def reload(self):
        """Reload manifest from file"""
        self._load_and_validate()
    
    def update_dataset(self, name: str, dataset_data: Dict[str, Any]) -> bool:
        """
        Update or add a dataset to the manifest
        
        Args:
            name: Dataset name
            dataset_data: Dataset definition dictionary
            
        Returns:
            True if update was successful
            
        Raises:
            ManifestValidationError: If updated manifest would be invalid
        """
        if not self.validate_dataset_name(name):
            raise ManifestValidationError(f"Invalid dataset name format: {name}")
        
        # Create a copy of manifest data for validation
        updated_manifest = self._manifest_data.copy()
        updated_manifest["datasets"] = self._manifest_data["datasets"].copy()
        updated_manifest["datasets"][name] = dataset_data
        
        # Validate the updated manifest
        try:
            jsonschema.validate(updated_manifest, MANIFEST_SCHEMA)
        except jsonschema.ValidationError as e:
            raise ManifestValidationError(f"Dataset update would create invalid manifest: {e}")
        
        # If validation passes, update the actual manifest
        self._manifest_data["datasets"][name] = dataset_data
        return True
    
    def remove_dataset(self, name: str) -> bool:
        """
        Remove a dataset from the manifest
        
        Args:
            name: Dataset name to remove
            
        Returns:
            True if dataset was removed, False if not found
        """
        if name in self._manifest_data["datasets"]:
            del self._manifest_data["datasets"][name]
            return True
        return False
    
    def save_manifest(self, backup: bool = True) -> None:
        """
        Save manifest to file
        
        Args:
            backup: Whether to create a backup of the existing file
        """
        if backup and self.manifest_path.exists():
            backup_path = self.manifest_path.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            import shutil
            shutil.copy2(self.manifest_path, backup_path)
        
        with open(self.manifest_path, 'w') as f:
            json.dump(self._manifest_data, f, indent=2, sort_keys=True)
    
    def migrate_version(self, target_version: str) -> bool:
        """
        Migrate manifest to a different version
        
        Args:
            target_version: Target manifest version
            
        Returns:
            True if migration was successful
            
        Note:
            Currently only supports version 1.0. Future versions will
            implement migration logic here.
        """
        current_version = self.version
        
        if current_version == target_version:
            return True
        
        # Version migration logic would go here
        # For now, only version 1.0 is supported
        if target_version != "1.0":
            raise ManifestValidationError(f"Migration to version {target_version} not supported")
        
        return True
    
    def search_datasets(self, query: str, fields: List[str] = None) -> List[str]:
        """
        Search datasets by text query
        
        Args:
            query: Search query string
            fields: Fields to search in (default: name, canonical_name, description)
            
        Returns:
            List of dataset names matching the query
        """
        if fields is None:
            fields = ["name", "canonical_name", "description"]
        
        query_lower = query.lower()
        matching_datasets = []
        
        for name in self.list_datasets():
            dataset_info = self.get_dataset_info(name)
            
            # Search in specified fields
            search_text = ""
            if "name" in fields:
                search_text += f" {name}"
            if "canonical_name" in fields:
                search_text += f" {dataset_info.canonical_name}"
            if "description" in fields:
                search_text += f" {dataset_info.description}"
            if "citation" in fields:
                search_text += f" {dataset_info.citation}"
            
            if query_lower in search_text.lower():
                matching_datasets.append(name)
        
        return matching_datasets
    
    def get_datasets_by_observable(self, observable: str) -> List[str]:
        """
        Get datasets that include a specific observable
        
        Args:
            observable: Observable name (e.g., 'DM_over_rd', 'MU')
            
        Returns:
            List of dataset names that include the observable
        """
        matching_datasets = []
        for name in self.list_datasets():
            dataset_info = self.get_dataset_info(name)
            if dataset_info.metadata:
                observables = dataset_info.metadata.get("observables", [])
                if observable in observables:
                    matching_datasets.append(name)
        return matching_datasets
    
    def get_datasets_by_redshift_range(self, z_min: float, z_max: float) -> List[str]:
        """
        Get datasets within a redshift range
        
        Args:
            z_min: Minimum redshift
            z_max: Maximum redshift
            
        Returns:
            List of dataset names with overlapping redshift ranges
        """
        matching_datasets = []
        for name in self.list_datasets():
            dataset_info = self.get_dataset_info(name)
            if dataset_info.metadata:
                z_range = dataset_info.metadata.get("redshift_range")
                if z_range and len(z_range) >= 2:
                    dataset_z_min, dataset_z_max = z_range[0], z_range[1]
                    # Check for overlap
                    if not (z_max < dataset_z_min or z_min > dataset_z_max):
                        matching_datasets.append(name)
        return matching_datasets
    
    def validate_manifest_integrity(self) -> Dict[str, Any]:
        """
        Perform comprehensive manifest integrity checks
        
        Returns:
            Dictionary with validation results and any issues found
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "dataset_count": len(self.list_datasets()),
            "schema_valid": True
        }
        
        try:
            # Re-validate schema
            jsonschema.validate(self._manifest_data, MANIFEST_SCHEMA)
        except jsonschema.ValidationError as e:
            results["valid"] = False
            results["schema_valid"] = False
            results["errors"].append(f"Schema validation failed: {e}")
        
        # Check for duplicate canonical names
        canonical_names = {}
        for name in self.list_datasets():
            dataset_info = self.get_dataset_info(name)
            canonical = dataset_info.canonical_name
            if canonical in canonical_names:
                results["warnings"].append(f"Duplicate canonical name '{canonical}' in datasets: {canonical_names[canonical]}, {name}")
            else:
                canonical_names[canonical] = name
        
        # Check for missing required metadata fields
        for name in self.list_datasets():
            dataset_info = self.get_dataset_info(name)
            if not dataset_info.metadata:
                results["warnings"].append(f"Dataset '{name}' missing metadata section")
            elif not dataset_info.metadata.get("dataset_type"):
                results["warnings"].append(f"Dataset '{name}' missing dataset_type in metadata")
        
        return results