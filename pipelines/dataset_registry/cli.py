#!/usr/bin/env python3
"""
Dataset Registry CLI Interface

Command-line tool for dataset listing, status checking, re-verification,
integrity checking, and cleanup operations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import textwrap

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_registry.core.registry_manager import RegistryManager, RegistryError
from dataset_registry.core.manifest_schema import DatasetManifest, ManifestValidationError
from dataset_registry.verification.verification_engine import VerificationEngine


class CLIError(Exception):
    """CLI-specific error"""
    pass


class DatasetCLI:
    """
    Command-line interface for dataset registry management
    
    Provides commands for listing datasets, checking status, re-verification,
    integrity checking, and cleanup operations.
    """
    
    def __init__(self, registry_path: str = "data/registry", manifest_path: str = "data/datasets_manifest.json"):
        """
        Initialize CLI with registry and manifest paths
        
        Args:
            registry_path: Path to registry directory
            manifest_path: Path to manifest file
        """
        self.registry_path = Path(registry_path)
        self.manifest_path = Path(manifest_path)
        
        # Initialize components
        try:
            self.registry_manager = RegistryManager(self.registry_path)
            if self.manifest_path.exists():
                self.manifest = DatasetManifest(self.manifest_path)
            else:
                self.manifest = None
            self.verification_engine = VerificationEngine()

        except Exception as e:
            raise CLIError(f"Failed to initialize CLI: {e}")
    
    def list_datasets(self, format_type: str = "table", filter_status: Optional[str] = None, 
                     filter_type: Optional[str] = None, show_details: bool = False) -> None:
        """
        List all datasets with optional filtering
        
        Args:
            format_type: Output format ('table', 'json', 'csv')
            filter_status: Filter by status ('verified', 'failed', 'corrupted')
            filter_type: Filter by dataset type ('cmb', 'bao', 'sn')
            show_details: Show detailed information
        """
        try:
            # Get registry summary
            summary = self.registry_manager.get_registry_summary()
            datasets = summary["datasets"]
            
            # Apply filters
            if filter_status:
                datasets = [d for d in datasets if d["status"] == filter_status]
            
            if filter_type and self.manifest:
                # Filter by dataset type from manifest metadata
                type_datasets = []
                for d in datasets:
                    try:
                        dataset_info = self.manifest.get_dataset_info(d["name"])
                        if dataset_info.metadata and dataset_info.metadata.get("dataset_type") == filter_type:
                            type_datasets.append(d)
                    except KeyError:
                        continue
                datasets = type_datasets
            
            if not datasets:
                print("No datasets found matching the criteria.")
                return
            
            # Output in requested format
            if format_type == "json":
                print(json.dumps(datasets, indent=2))
            elif format_type == "csv":
                self._print_csv(datasets, show_details)
            else:  # table format
                self._print_table(datasets, show_details)
                
        except Exception as e:
            raise CLIError(f"Failed to list datasets: {e}")
    
    def show_dataset_status(self, dataset_name: str) -> None:
        """
        Show detailed status for a specific dataset
        
        Args:
            dataset_name: Name of the dataset
        """
        try:
            # Get registry entry
            entry = self.registry_manager.get_registry_entry(dataset_name)
            if not entry:
                print(f"Dataset '{dataset_name}' not found in registry.")
                return
            
            # Get manifest info if available
            manifest_info = None
            if self.manifest and self.manifest.has_dataset(dataset_name):
                manifest_info = self.manifest.get_dataset_info(dataset_name)
            
            # Print detailed status
            print(f"\n=== Dataset Status: {dataset_name} ===")
            print(f"Status: {'✓ VERIFIED' if entry.verification.is_valid else '✗ FAILED'}")
            print(f"Last Verified: {entry.verification.verification_timestamp}")
            print(f"Source Type: {entry.source_used}")
            print(f"File Path: {entry.file_info['local_path']}")
            
            if manifest_info:
                print(f"Canonical Name: {manifest_info.canonical_name}")
                print(f"Description: {manifest_info.description}")
                print(f"Citation: {manifest_info.citation}")
                if manifest_info.metadata:
                    print(f"Dataset Type: {manifest_info.metadata.get('dataset_type', 'Unknown')}")
            
            print(f"\n--- Verification Details ---")
            print(f"SHA256 Match: {'✓' if entry.verification.sha256_verified else '✗'}")
            if entry.verification.sha256_expected:
                print(f"Expected SHA256: {entry.verification.sha256_expected}")
            if entry.verification.sha256_actual:
                print(f"Actual SHA256: {entry.verification.sha256_actual}")
            
            print(f"Size Match: {'✓' if entry.verification.size_verified else '✗'}")
            if entry.verification.size_expected:
                print(f"Expected Size: {entry.verification.size_expected:,} bytes")
            if entry.verification.size_actual:
                print(f"Actual Size: {entry.verification.size_actual:,} bytes")
            
            print(f"Schema Valid: {'✓' if entry.verification.schema_verified else '✗'}")
            if entry.verification.schema_errors:
                print("Schema Errors:")
                for error in entry.verification.schema_errors:
                    print(f"  - {error}")
            
            print(f"\n--- Environment ---")
            print(f"PBUF Commit: {entry.environment.pbuf_commit or 'Unknown'}")
            print(f"Python Version: {entry.environment.python_version}")
            print(f"Platform: {entry.environment.platform}")
            print(f"Download Agent: {entry.download_agent}")
            print(f"Download Time: {entry.download_timestamp}")
            
        except Exception as e:
            raise CLIError(f"Failed to show dataset status: {e}")
    
    def verify_dataset(self, dataset_name: str, force: bool = False) -> None:
        """
        Re-verify a specific dataset
        
        Args:
            dataset_name: Name of the dataset to verify
            force: Force verification even if recently verified
        """
        try:
            print(f"Verifying dataset: {dataset_name}")
            
            # Check if dataset exists in registry
            entry = self.registry_manager.get_registry_entry(dataset_name)
            if not entry:
                print(f"Dataset '{dataset_name}' not found in registry.")
                return
            
            # Check if verification is recent (unless forced)
            if not force:
                last_verified = datetime.fromisoformat(entry.verification.verification_timestamp)
                hours_since = (datetime.now() - last_verified).total_seconds() / 3600
                if hours_since < 1 and entry.verification.is_valid:
                    print(f"Dataset was verified {hours_since:.1f} hours ago and is valid. Use --force to re-verify.")
                    return
            
            # Get file path
            file_path = Path(entry.file_info["local_path"])
            
            # Get verification config
            verification_config = {}
            if self.manifest and self.manifest.has_dataset(dataset_name):
                manifest_info = self.manifest.get_dataset_info(dataset_name)
                verification_config = manifest_info.verification
            else:
                # Use stored verification info
                verification_config = {
                    "sha256": entry.verification.sha256_expected,
                    "size_bytes": entry.verification.size_expected
                }
            
            # Perform verification
            result = self.verification_engine.verify_dataset(dataset_name, file_path, verification_config)
            
            # Update registry
            from dataset_registry.core.registry_manager import VerificationResult as RegistryVerificationResult
            registry_result = RegistryVerificationResult(
                sha256_verified=result.sha256_match,
                sha256_expected=result.sha256_expected,
                sha256_actual=result.sha256_actual,
                size_verified=result.size_match,
                size_expected=result.size_expected,
                size_actual=result.size_actual,
                schema_verified=result.schema_valid,
                schema_errors=result.schema_errors,
                verification_timestamp=result.verification_time.isoformat()
            )
            
            self.registry_manager.update_verification(dataset_name, registry_result)
            
            # Print results
            if result.is_valid:
                print(f"✓ Dataset '{dataset_name}' verification PASSED")
            else:
                print(f"✗ Dataset '{dataset_name}' verification FAILED")
                for error in result.errors:
                    print(f"  Error: {error}")
                for suggestion in result.suggestions:
                    print(f"  Suggestion: {suggestion}")
            
        except Exception as e:
            raise CLIError(f"Failed to verify dataset: {e}")
    
    def verify_all_datasets(self, force: bool = False) -> None:
        """
        Re-verify all datasets in the registry
        
        Args:
            force: Force verification even if recently verified
        """
        try:
            datasets = self.registry_manager.list_datasets()
            if not datasets:
                print("No datasets found in registry.")
                return
            
            print(f"Verifying {len(datasets)} datasets...")
            
            passed = 0
            failed = 0
            
            for dataset_name in datasets:
                try:
                    print(f"\nVerifying {dataset_name}...")
                    self.verify_dataset(dataset_name, force=force)
                    
                    # Check result
                    entry = self.registry_manager.get_registry_entry(dataset_name)
                    if entry and entry.verification.is_valid:
                        passed += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    print(f"✗ Failed to verify {dataset_name}: {e}")
                    failed += 1
            
            print(f"\n=== Verification Summary ===")
            print(f"Passed: {passed}")
            print(f"Failed: {failed}")
            print(f"Total: {len(datasets)}")
            
        except Exception as e:
            raise CLIError(f"Failed to verify all datasets: {e}")
    
    def check_integrity(self) -> None:
        """
        Check integrity of the registry and datasets
        """
        try:
            print("Checking registry integrity...")
            
            # Check registry integrity
            registry_results = self.registry_manager.validate_registry_integrity()
            
            print(f"\n=== Registry Integrity ===")
            print(f"Status: {'✓ VALID' if registry_results['valid'] else '✗ INVALID'}")
            print(f"Total Entries: {registry_results['total_entries']}")
            print(f"Valid Entries: {registry_results['valid_entries']}")
            print(f"Corrupted Entries: {registry_results['corrupted_entries']}")
            
            if registry_results['errors']:
                print("\nErrors:")
                for error in registry_results['errors']:
                    print(f"  - {error}")
            
            if registry_results['warnings']:
                print("\nWarnings:")
                for warning in registry_results['warnings']:
                    print(f"  - {warning}")
            
            # Check manifest integrity if available
            if self.manifest:
                print(f"\n=== Manifest Integrity ===")
                manifest_results = self.manifest.validate_manifest_integrity()
                print(f"Status: {'✓ VALID' if manifest_results['valid'] else '✗ INVALID'}")
                print(f"Schema Valid: {'✓' if manifest_results['schema_valid'] else '✗'}")
                print(f"Dataset Count: {manifest_results['dataset_count']}")
                
                if manifest_results['errors']:
                    print("\nErrors:")
                    for error in manifest_results['errors']:
                        print(f"  - {error}")
                
                if manifest_results['warnings']:
                    print("\nWarnings:")
                    for warning in manifest_results['warnings']:
                        print(f"  - {warning}")
            
            # Check for orphaned files
            print(f"\n=== Orphaned Files Check ===")
            orphaned_files = self._find_orphaned_files()
            if orphaned_files:
                print(f"Found {len(orphaned_files)} orphaned files:")
                for file_path in orphaned_files:
                    print(f"  - {file_path}")
            else:
                print("No orphaned files found.")
            
        except Exception as e:
            raise CLIError(f"Failed to check integrity: {e}")
    
    def cleanup_orphaned_files(self, dry_run: bool = True) -> None:
        """
        Clean up orphaned dataset files
        
        Args:
            dry_run: If True, only show what would be deleted
        """
        try:
            orphaned_files = self._find_orphaned_files()
            
            if not orphaned_files:
                print("No orphaned files found.")
                return
            
            print(f"Found {len(orphaned_files)} orphaned files:")
            
            total_size = 0
            for file_path in orphaned_files:
                try:
                    size = file_path.stat().st_size
                    total_size += size
                    print(f"  - {file_path} ({size:,} bytes)")
                except OSError:
                    print(f"  - {file_path} (size unknown)")
            
            print(f"\nTotal size: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
            
            if dry_run:
                print("\nDry run mode - no files deleted. Use --no-dry-run to actually delete files.")
            else:
                # Confirm deletion
                response = input("\nDelete these files? [y/N]: ")
                if response.lower() in ['y', 'yes']:
                    deleted = 0
                    for file_path in orphaned_files:
                        try:
                            file_path.unlink()
                            deleted += 1
                            print(f"Deleted: {file_path}")
                        except OSError as e:
                            print(f"Failed to delete {file_path}: {e}")
                    
                    print(f"\nDeleted {deleted} out of {len(orphaned_files)} files.")
                else:
                    print("Cleanup cancelled.")
            
        except Exception as e:
            raise CLIError(f"Failed to cleanup orphaned files: {e}")
    
    def export_summary(self, output_file: Optional[str] = None, format_type: str = "json", 
                      export_type: str = "registry") -> None:
        """
        Export registry summary for publication materials
        
        Args:
            output_file: Output file path (stdout if None)
            format_type: Output format ('json', 'markdown', 'csv', 'latex', 'yaml')
            export_type: Type of export ('registry', 'manifest', 'provenance', 'audit')
        """
        try:
            from dataset_registry.export.export_manager import ExportManager, ExportFormat, ExportOptions
            from dataset_registry.export.summary_generator import SummaryGenerator, SummaryType
            
            # Create export manager
            export_manager = ExportManager(self.registry_manager, self.manifest)
            
            # Convert format string to enum
            format_enum = ExportFormat(format_type.lower())
            
            if export_type == "registry":
                # Export registry summary
                options = ExportOptions(
                    include_metadata=True,
                    include_provenance=True,
                    include_verification=True,
                    include_environment=True
                )
                output = export_manager.export_registry_summary(format_enum, options, output_file)
            elif export_type == "manifest":
                # Export manifest table
                output = export_manager.export_manifest_table(format_enum, output_file)
            elif export_type == "provenance":
                # Export provenance bundle
                output = export_manager.export_provenance_bundle(output_file=output_file)
            elif export_type == "audit":
                # Export audit trail
                output = export_manager.export_audit_trail(format_enum, output_file=output_file)
            else:
                raise CLIError(f"Unsupported export type: {export_type}")
            
            if not output_file:
                print(output)
            else:
                print(f"Summary exported to: {output_file}")
                
        except Exception as e:
            raise CLIError(f"Failed to export summary: {e}")
    
    def show_audit_trail(self, dataset_name: Optional[str] = None, limit: int = 50) -> None:
        """
        Show audit trail for datasets
        
        Args:
            dataset_name: Show trail for specific dataset (all if None)
            limit: Maximum number of entries to show
        """
        try:
            entries = self.registry_manager.get_audit_trail(dataset_name)
            
            if not entries:
                if dataset_name:
                    print(f"No audit trail found for dataset: {dataset_name}")
                else:
                    print("No audit trail entries found.")
                return
            
            # Sort by timestamp (newest first) and limit
            entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            entries = entries[:limit]
            
            print(f"\n=== Audit Trail ===")
            if dataset_name:
                print(f"Dataset: {dataset_name}")
            print(f"Showing {len(entries)} most recent entries:\n")
            
            for entry in entries:
                timestamp = entry.get("timestamp", "Unknown")
                event = entry.get("event", "Unknown")
                dataset = entry.get("dataset_name", "Unknown")
                
                print(f"{timestamp} | {event} | {dataset}")
                
                # Show additional details based on event type
                if event == "registry_create":
                    agent = entry.get("agent", "Unknown")
                    commit = entry.get("pbuf_commit", "Unknown")
                    print(f"  Agent: {agent}, PBUF Commit: {commit}")
                elif event == "verification_update":
                    status = entry.get("verification_status", "Unknown")
                    source_type = entry.get("source_type", "Unknown")
                    print(f"  Status: {status}, Source: {source_type}")
                
                print()
            
        except Exception as e:
            raise CLIError(f"Failed to show audit trail: {e}")
    
    def generate_summary(self, summary_type: str, output_file: Optional[str] = None, format_type: str = "json") -> None:
        """
        Generate specialized summary reports
        
        Args:
            summary_type: Type of summary ('publication', 'compliance', 'operational', 'executive', 'technical')
            output_file: Output file path (stdout if None)
            format_type: Output format ('json', 'markdown')
        """
        try:
            from dataset_registry.export.summary_generator import SummaryGenerator, SummaryType, SummaryOptions
            
            # Create summary generator
            summary_generator = SummaryGenerator(self.registry_manager, self.manifest)
            
            # Convert type string to enum
            type_enum = SummaryType(summary_type.lower())
            
            # Generate summary
            options = SummaryOptions(
                include_statistics=True,
                include_trends=True,
                include_issues=True,
                include_recommendations=True
            )
            
            summary_data = summary_generator.generate_summary(type_enum, options)
            
            # Format output
            if format_type == "json":
                output = json.dumps(summary_data, indent=2)
            elif format_type == "markdown":
                output = self._format_summary_markdown(summary_data, summary_type)
            else:
                raise CLIError(f"Unsupported format for summary: {format_type}")
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(output)
                print(f"Summary generated: {output_file}")
            else:
                print(output)
                
        except Exception as e:
            raise CLIError(f"Failed to generate summary: {e}")
    
    def _format_summary_markdown(self, summary_data: Dict[str, Any], summary_type: str) -> str:
        """Format summary data as Markdown"""
        lines = [
            f"# {summary_type.title()} Summary",
            "",
            f"**Generated:** {summary_data.get('generated_at', 'Unknown')}",
            ""
        ]
        
        # Add type-specific formatting
        if summary_type == "executive":
            key_metrics = summary_data.get("key_metrics", {})
            lines.extend([
                "## Key Metrics",
                "",
                f"- **Total Datasets:** {key_metrics.get('total_datasets_managed', 'Unknown')}",
                f"- **Data Integrity Rate:** {key_metrics.get('data_integrity_rate', 'Unknown')}",
                f"- **System Status:** {key_metrics.get('system_status', 'Unknown')}",
                ""
            ])
            
            if "executive_summary" in summary_data:
                lines.extend([
                    "## Executive Summary",
                    "",
                    summary_data["executive_summary"],
                    ""
                ])
        
        elif summary_type == "operational":
            system_status = summary_data.get("system_status", {})
            lines.extend([
                "## System Status",
                "",
                f"- **Total Datasets:** {system_status.get('total_datasets', 'Unknown')}",
                f"- **Operational Datasets:** {system_status.get('operational_datasets', 'Unknown')}",
                f"- **Failed Datasets:** {system_status.get('failed_datasets', 'Unknown')}",
                f"- **System Health:** {system_status.get('system_health', 'Unknown')}",
                ""
            ])
            
            if "operational_issues" in summary_data:
                issues = summary_data["operational_issues"]
                if issues:
                    lines.extend(["## Issues", ""])
                    for issue in issues:
                        lines.append(f"- **{issue.get('type', 'Unknown')}** ({issue.get('severity', 'Unknown')}): {issue.get('description', 'No description')}")
                    lines.append("")
        
        # Add raw data as JSON for other types
        else:
            lines.extend([
                "## Summary Data",
                "",
                "```json",
                json.dumps(summary_data, indent=2),
                "```"
            ])
        
        return "\n".join(lines)
    
    def fetch_all_datasets(self, force_refresh: bool = False, parallel: bool = True, 
                          output_file: Optional[str] = None) -> None:
        """
        Fetch and verify all datasets for complete workflow reproduction
        
        Args:
            force_refresh: Force re-download even if datasets exist
            parallel: Enable parallel downloading
            output_file: Save detailed results to file
        """
        try:
            from dataset_registry.reproducibility import ReproducibilityManager
            
            print("Starting comprehensive dataset reproduction...")
            
            # Create reproducibility manager
            manager = ReproducibilityManager(
                manifest_path=self.manifest_path,
                registry_path=self.registry_path
            )
            
            # Progress callback for CLI output
            def progress_callback(progress):
                status_icon = {
                    "pending": "⏳",
                    "initializing": "🔄", 
                    "fetching": "⬇️",
                    "verifying": "✅",
                    "completed": "✓",
                    "failed": "✗",
                    "cancelled": "⚠️"
                }.get(progress.status.value, "?")
                
                print(f"\r{status_icon} {progress.status.value.title()}: "
                      f"{progress.progress_percent:.1f}% "
                      f"({progress.completed_datasets}/{progress.total_datasets} datasets) "
                      f"[{progress.elapsed_time:.1f}s]", end="", flush=True)
            
            # Fetch all datasets
            result = manager.fetch_all_datasets(
                force_refresh=force_refresh,
                parallel=parallel,
                progress_callback=progress_callback
            )
            
            print()  # New line after progress
            
            # Print results
            if result.success:
                print(f"\n✓ Dataset reproduction completed successfully!")
                print(f"  📊 Summary:")
                print(f"    - Total datasets: {result.total_datasets}")
                print(f"    - Successful: {len(result.successful_datasets)}")
                print(f"    - Failed: {len(result.failed_datasets)}")
                print(f"    - Total size: {result.total_size / 1024 / 1024:.1f} MB")
                print(f"    - Total time: {result.total_time:.1f} seconds")
                
                if result.successful_datasets:
                    print(f"\n  ✓ Ready datasets:")
                    for dataset in result.successful_datasets:
                        print(f"    - {dataset}")
            else:
                print(f"\n✗ Dataset reproduction failed!")
                print(f"  📊 Summary:")
                print(f"    - Total datasets: {result.total_datasets}")
                print(f"    - Successful: {len(result.successful_datasets)}")
                print(f"    - Failed: {len(result.failed_datasets)}")
                
                if result.failed_datasets:
                    print(f"\n  ✗ Failed datasets:")
                    for dataset in result.failed_datasets:
                        print(f"    - {dataset}")
                
                if result.errors:
                    print(f"\n  🔍 Errors:")
                    for error in result.errors[:5]:  # Show first 5 errors
                        print(f"    - {error}")
                    if len(result.errors) > 5:
                        print(f"    ... and {len(result.errors) - 5} more errors")
                
                if result.suggestions:
                    print(f"\n  💡 Suggestions:")
                    for suggestion in result.suggestions:
                        print(f"    - {suggestion}")
            
            # Save detailed results if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"\n📄 Detailed results saved to: {output_file}")
            
        except Exception as e:
            raise CLIError(f"Failed to fetch all datasets: {e}")
    
    def reproduction_status(self) -> None:
        """
        Show current reproduction system status
        """
        try:
            from dataset_registry.reproducibility import ReproducibilityManager
            
            # Create reproducibility manager
            manager = ReproducibilityManager(
                manifest_path=self.manifest_path,
                registry_path=self.registry_path
            )
            
            # Get status
            status = manager.get_reproduction_status()
            
            print(f"\n=== Reproduction System Status ===")
            
            system_status = status.get("system_status", "unknown")
            status_icon = "✓" if system_status == "ready" else "⚠️" if system_status == "incomplete" else "✗"
            
            print(f"System Status: {status_icon} {system_status.upper()}")
            print(f"Total Datasets Available: {status.get('total_datasets_available', 0)}")
            print(f"Datasets Ready: {status.get('datasets_ready', 0)}")
            print(f"Datasets Missing: {status.get('datasets_missing', 0)}")
            
            # Show ready datasets
            ready_datasets = status.get("ready_datasets", [])
            if ready_datasets:
                print(f"\n✓ Ready for reproduction ({len(ready_datasets)} datasets):")
                for dataset in ready_datasets:
                    print(f"  - {dataset}")
            
            # Show missing datasets
            missing_datasets = status.get("missing_datasets", [])
            if missing_datasets:
                print(f"\n⚠️ Missing datasets ({len(missing_datasets)} datasets):")
                for dataset in missing_datasets:
                    print(f"  - {dataset}")
                
                print(f"\n💡 To fetch missing datasets:")
                print(f"   dataset-cli fetch-all")
            
            # Show environment info
            env_info = status.get("environment", {})
            if env_info:
                print(f"\n--- Environment ---")
                print(f"PBUF Commit: {env_info.get('pbuf_commit', 'Unknown')}")
                print(f"Python Version: {env_info.get('python_version', 'Unknown')}")
                print(f"Platform: {env_info.get('platform', 'Unknown')}")
            
            # Show registry stats
            registry_stats = env_info.get("registry_stats")
            if registry_stats:
                print(f"\n--- Registry Statistics ---")
                print(f"Total Datasets: {registry_stats.get('total_datasets', 0)}")
                print(f"Verified Datasets: {registry_stats.get('verified_datasets', 0)}")
                print(f"Failed Datasets: {registry_stats.get('failed_datasets', 0)}")
            
        except Exception as e:
            raise CLIError(f"Failed to get reproduction status: {e}")
    
    def run_diagnostics(self, output_file: Optional[str] = None, format_type: str = "text") -> None:
        """
        Run comprehensive reproduction diagnostics
        
        Args:
            output_file: Save detailed results to file
            format_type: Output format ('text', 'json')
        """
        try:
            from dataset_registry.reproducibility.diagnostics import ReproductionDiagnostics
            
            print("Running comprehensive reproduction diagnostics...")
            
            # Create diagnostics system
            diagnostics = ReproductionDiagnostics(
                manifest_path=self.manifest_path,
                registry_path=self.registry_path
            )
            
            # Run diagnostics
            report = diagnostics.run_comprehensive_diagnostics()
            
            # Print results based on format
            if format_type == "json":
                output = json.dumps(report.to_dict(), indent=2)
                print(output)
            else:
                # Text format
                self._print_diagnostic_report(report)
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    if format_type == "json":
                        json.dump(report.to_dict(), f, indent=2)
                    else:
                        f.write(self._format_diagnostic_report_text(report))
                print(f"\n📄 Detailed diagnostics saved to: {output_file}")
            
        except Exception as e:
            raise CLIError(f"Failed to run diagnostics: {e}")
    
    def _print_diagnostic_report(self, report: 'DiagnosticReport') -> None:
        """Print diagnostic report in human-readable format"""
        # Header
        health_icon = {
            "healthy": "✅",
            "warning": "⚠️", 
            "degraded": "🔶",
            "critical": "🚨"
        }.get(report.overall_health, "❓")
        
        print(f"\n=== Reproduction System Diagnostics ===")
        print(f"Overall Health: {health_icon} {report.overall_health.upper()}")
        print(f"System Status: {report.system_status}")
        print(f"Timestamp: {report.timestamp}")
        
        # Summary
        print(f"\n📊 Issue Summary:")
        print(f"  Total Issues: {report.total_issues}")
        if report.critical_issues > 0:
            print(f"  🚨 Critical: {report.critical_issues}")
        if report.error_issues > 0:
            print(f"  ❌ Errors: {report.error_issues}")
        if report.warning_issues > 0:
            print(f"  ⚠️ Warnings: {report.warning_issues}")
        if report.info_issues > 0:
            print(f"  ℹ️ Info: {report.info_issues}")
        
        # Issues by category
        if report.issues:
            print(f"\n🔍 Issues Found:")
            
            # Group by category
            issues_by_category = {}
            for issue in report.issues:
                category = issue.category.value
                if category not in issues_by_category:
                    issues_by_category[category] = []
                issues_by_category[category].append(issue)
            
            for category, issues in issues_by_category.items():
                print(f"\n  📂 {category.title()} Issues:")
                for issue in issues:
                    severity_icon = {
                        "critical": "🚨",
                        "error": "❌",
                        "warning": "⚠️",
                        "info": "ℹ️"
                    }.get(issue.severity.value, "❓")
                    
                    print(f"    {severity_icon} {issue.title}")
                    print(f"       {issue.description}")
                    
                    if issue.dataset_name:
                        print(f"       Dataset: {issue.dataset_name}")
                    
                    if issue.suggestions:
                        print(f"       Suggestions:")
                        for suggestion in issue.suggestions[:2]:  # Show first 2 suggestions
                            print(f"         - {suggestion}")
                        if len(issue.suggestions) > 2:
                            print(f"         ... and {len(issue.suggestions) - 2} more")
                    print()
        
        # Recommendations
        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
        
        # Recovery steps
        if report.recovery_steps:
            print(f"\n🔧 Recovery Steps:")
            for step in report.recovery_steps:
                print(f"  {step}")
        
        # System info summary
        if report.system_info:
            print(f"\n🖥️ System Information:")
            if "python_version" in report.system_info:
                python_ver = report.system_info["python_version"].split()[0]
                print(f"  Python: {python_ver}")
            if "platform" in report.system_info:
                print(f"  Platform: {report.system_info['platform']}")
            if "git_commit" in report.system_info and report.system_info["git_commit"]:
                commit = report.system_info["git_commit"][:8]
                print(f"  Git Commit: {commit}")
            if "disk_space" in report.system_info and report.system_info["disk_space"]:
                free_gb = report.system_info["disk_space"]["free_gb"]
                print(f"  Free Disk Space: {free_gb:.1f} GB")
    
    def _format_diagnostic_report_text(self, report: 'DiagnosticReport') -> str:
        """Format diagnostic report as text for file output"""
        lines = [
            "# Reproduction System Diagnostics Report",
            f"",
            f"**Generated:** {report.timestamp}",
            f"**Overall Health:** {report.overall_health.upper()}",
            f"**System Status:** {report.system_status}",
            f"",
            f"## Summary",
            f"",
            f"- Total Issues: {report.total_issues}",
            f"- Critical Issues: {report.critical_issues}",
            f"- Error Issues: {report.error_issues}",
            f"- Warning Issues: {report.warning_issues}",
            f"- Info Issues: {report.info_issues}",
            f""
        ]
        
        # Add issues
        if report.issues:
            lines.extend([
                "## Issues",
                ""
            ])
            
            for i, issue in enumerate(report.issues, 1):
                lines.extend([
                    f"### {i}. {issue.title}",
                    f"",
                    f"**Category:** {issue.category.value.title()}",
                    f"**Severity:** {issue.severity.value.title()}",
                    f"**Description:** {issue.description}",
                ])
                
                if issue.dataset_name:
                    lines.append(f"**Dataset:** {issue.dataset_name}")
                
                if issue.suggestions:
                    lines.extend([
                        f"**Suggestions:**",
                        ""
                    ])
                    for suggestion in issue.suggestions:
                        lines.append(f"- {suggestion}")
                
                lines.append("")
        
        # Add recommendations
        if report.recommendations:
            lines.extend([
                "## Recommendations",
                ""
            ])
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Add recovery steps
        if report.recovery_steps:
            lines.extend([
                "## Recovery Steps",
                ""
            ])
            for step in report.recovery_steps:
                lines.append(f"{step}")
            lines.append("")
        
        return "\n".join(lines)
    
    def prepare_reproduction_bundle(self, output_dir: Optional[str] = None, 
                                   include_logs: bool = True, include_provenance: bool = True) -> None:
        """
        Prepare complete reproduction bundle with all necessary files
        
        Args:
            output_dir: Output directory for reproduction bundle
            include_logs: Include audit logs and verification history
            include_provenance: Include complete provenance records
        """
        try:
            from dataset_registry.reproducibility import ReproducibilityManager
            
            # Create reproducibility manager
            manager = ReproducibilityManager(
                manifest_path=self.manifest_path,
                registry_path=self.registry_path
            )
            
            # Prepare bundle
            output_path = Path(output_dir) if output_dir else None
            result = manager.prepare_reproduction_environment(
                output_dir=output_path,
                include_logs=include_logs,
                include_provenance=include_provenance
            )
            
            if result["success"]:
                print(f"✓ Reproduction bundle prepared successfully!")
                print(f"  📁 Output directory: {result['output_directory']}")
                print(f"  📄 Files created: {len(result['files_created'])}")
                
                print(f"\n  📋 Bundle contents:")
                for file_path in result["files_created"]:
                    file_name = Path(file_path).name
                    print(f"    - {file_name}")
                
                print(f"\n  🚀 To reproduce datasets:")
                print(f"    cd {result['output_directory']}")
                print(f"    python reproduce_datasets.py")
                
            else:
                print(f"✗ Failed to prepare reproduction bundle!")
                if result["errors"]:
                    print(f"  Errors:")
                    for error in result["errors"]:
                        print(f"    - {error}")
            
        except Exception as e:
            raise CLIError(f"Failed to prepare reproduction bundle: {e}")
    
    def _find_orphaned_files(self) -> List[Path]:
        """
        Find files in data directory that are not referenced in registry
        
        Returns:
            List of orphaned file paths
        """
        orphaned_files = []
        
        # Get all registered file paths
        registered_paths = set()
        for dataset_name in self.registry_manager.list_datasets():
            try:
                entry = self.registry_manager.get_registry_entry(dataset_name)
                if entry:
                    registered_paths.add(Path(entry.file_info["local_path"]))
            except Exception:
                continue
        
        # Check data directory for unregistered files
        data_dir = Path("data")
        if data_dir.exists():
            for file_path in data_dir.rglob("*"):
                if file_path.is_file() and file_path not in registered_paths:
                    # Skip known system files
                    if file_path.name in ["datasets_manifest.json", ".gitkeep", "README.md"]:
                        continue
                    if file_path.suffix in [".json", ".jsonl"] and "registry" in str(file_path):
                        continue
                    
                    orphaned_files.append(file_path)
        
        return orphaned_files
    
    def _print_table(self, datasets: List[Dict[str, Any]], show_details: bool = False) -> None:
        """Print datasets in table format"""
        if not datasets:
            return
        
        # Calculate column widths
        name_width = max(len(d["name"]) for d in datasets) + 2
        status_width = 12
        source_width = 12
        
        # Header
        print(f"{'Name':<{name_width}} {'Status':<{status_width}} {'Source':<{source_width}}", end="")
        if show_details:
            print(f" {'Last Verified':<20} {'PBUF Commit':<12}")
        else:
            print()
        
        print("-" * (name_width + status_width + source_width + (32 if show_details else 0)))
        
        # Rows
        for dataset in datasets:
            status_icon = "✓" if dataset["status"] == "verified" else "✗"
            status_text = f"{status_icon} {dataset['status'].upper()}"
            
            print(f"{dataset['name']:<{name_width}} {status_text:<{status_width}} {dataset['source_type']:<{source_width}}", end="")
            
            if show_details:
                last_verified = dataset.get("last_verified", "Unknown")
                if last_verified and last_verified != "Unknown":
                    try:
                        dt = datetime.fromisoformat(last_verified.replace('Z', '+00:00'))
                        last_verified = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                
                pbuf_commit = dataset.get("pbuf_commit", "Unknown")
                if pbuf_commit and len(pbuf_commit) > 8:
                    pbuf_commit = pbuf_commit[:8]
                
                print(f" {last_verified:<20} {pbuf_commit:<12}")
            else:
                print()
    
    def _print_csv(self, datasets: List[Dict[str, Any]], show_details: bool = False) -> None:
        """Print datasets in CSV format"""
        import csv
        import io
        
        output = io.StringIO()
        
        if show_details:
            fieldnames = ["name", "status", "source_type", "last_verified", "pbuf_commit"]
        else:
            fieldnames = ["name", "status", "source_type"]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for dataset in datasets:
            row = {field: dataset.get(field, "") for field in fieldnames}
            writer.writerow(row)
        
        print(output.getvalue().strip())
    
    def _format_markdown_summary(self, summary: Dict[str, Any]) -> str:
        """Format summary as Markdown table"""
        lines = [
            "# Dataset Registry Summary",
            f"",
            f"**Export Date:** {summary['export_timestamp']}",
            f"**Total Datasets:** {summary['total_datasets']}",
            f"",
            "## Datasets",
            "",
            "| Dataset | Source Type | Status | SHA256 | PBUF Commit |",
            "|---------|-------------|--------|--------|-------------|"
        ]
        
        for name, data in summary["datasets"].items():
            status = "✓" if data.get("verification_status") == "verified" else "✗"
            sha256 = data.get("sha256", "")[:16] + "..." if data.get("sha256") else "N/A"
            commit = data.get("pbuf_commit", "N/A")[:8] if data.get("pbuf_commit") else "N/A"
            
            lines.append(f"| {name} | {data.get('source_type', 'Unknown')} | {status} | `{sha256}` | `{commit}` |")
        
        return "\n".join(lines)
    
    def _format_csv_summary(self, summary: Dict[str, Any]) -> str:
        """Format summary as CSV"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["dataset_name", "source_type", "verification_status", "sha256", "pbuf_commit", "download_timestamp"])
        
        # Data rows
        for name, data in summary["datasets"].items():
            writer.writerow([
                name,
                data.get("source_type", ""),
                data.get("verification_status", ""),
                data.get("sha256", ""),
                data.get("pbuf_commit", ""),
                data.get("download_timestamp", "")
            ])
        
        return output.getvalue().strip()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Dataset Registry Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          %(prog)s list                          # List all datasets
          %(prog)s list --status verified        # List only verified datasets
          %(prog)s list --type cmb --details     # List CMB datasets with details
          %(prog)s status cmb_planck2018         # Show detailed status
          %(prog)s verify cmb_planck2018         # Re-verify specific dataset
          %(prog)s verify-all --force            # Re-verify all datasets
          %(prog)s integrity                     # Check registry integrity
          %(prog)s cleanup --dry-run             # Show orphaned files (dry run)
          %(prog)s export --format markdown      # Export summary as Markdown
          %(prog)s audit --dataset cmb_planck2018 # Show audit trail for dataset
          %(prog)s fetch-all                     # Fetch all datasets for reproduction
          %(prog)s fetch-all --force-refresh     # Force re-download all datasets
          %(prog)s reproduction-status           # Check reproduction system status
          %(prog)s prepare-bundle --output-dir ./bundle # Prepare reproduction bundle
          %(prog)s diagnostics                   # Run comprehensive system diagnostics
          %(prog)s diagnostics --format json --output diag.json # Save diagnostics as JSON
        """)
    )
    
    # Global options
    parser.add_argument("--registry-path", default="data/registry",
                       help="Path to registry directory (default: data/registry)")
    parser.add_argument("--manifest-path", default="data/datasets_manifest.json",
                       help="Path to manifest file (default: data/datasets_manifest.json)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List datasets")
    list_parser.add_argument("--format", choices=["table", "json", "csv"], default="table",
                            help="Output format (default: table)")
    list_parser.add_argument("--status", choices=["verified", "failed", "corrupted"],
                            help="Filter by verification status")
    list_parser.add_argument("--type", help="Filter by dataset type (e.g., cmb, bao, sn)")
    list_parser.add_argument("--details", action="store_true",
                            help="Show detailed information")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show dataset status")
    status_parser.add_argument("dataset", help="Dataset name")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Re-verify dataset")
    verify_parser.add_argument("dataset", help="Dataset name")
    verify_parser.add_argument("--force", action="store_true",
                              help="Force verification even if recently verified")
    
    # Verify-all command
    verify_all_parser = subparsers.add_parser("verify-all", help="Re-verify all datasets")
    verify_all_parser.add_argument("--force", action="store_true",
                                  help="Force verification even if recently verified")
    
    # Integrity command
    subparsers.add_parser("integrity", help="Check registry and manifest integrity")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up orphaned files")
    cleanup_parser.add_argument("--dry-run", action="store_true", default=True,
                               help="Show what would be deleted without deleting (default)")
    cleanup_parser.add_argument("--no-dry-run", action="store_true",
                               help="Actually delete orphaned files")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export registry summary")
    export_parser.add_argument("--output", help="Output file path (stdout if not specified)")
    export_parser.add_argument("--format", choices=["json", "markdown", "csv", "latex", "yaml"], default="json",
                              help="Output format (default: json)")
    export_parser.add_argument("--type", choices=["registry", "manifest", "provenance", "audit"], default="registry",
                              help="Type of export (default: registry)")
    
    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Show audit trail")
    audit_parser.add_argument("--dataset", help="Show trail for specific dataset")
    audit_parser.add_argument("--limit", type=int, default=50,
                             help="Maximum number of entries to show (default: 50)")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Generate specialized summaries")
    summary_parser.add_argument("--type", choices=["publication", "compliance", "operational", "executive", "technical"], 
                               default="operational", help="Type of summary (default: operational)")
    summary_parser.add_argument("--output", help="Output file path (stdout if not specified)")
    summary_parser.add_argument("--format", choices=["json", "markdown"], default="json",
                               help="Output format (default: json)")
    
    # Fetch-all command (reproducibility)
    fetch_all_parser = subparsers.add_parser("fetch-all", help="Fetch and verify all datasets for reproduction")
    fetch_all_parser.add_argument("--force-refresh", action="store_true",
                                 help="Force re-download even if datasets exist")
    fetch_all_parser.add_argument("--parallel", action="store_true", default=True,
                                 help="Enable parallel downloading (default: true)")
    fetch_all_parser.add_argument("--no-parallel", action="store_true",
                                 help="Disable parallel downloading")
    fetch_all_parser.add_argument("--output", help="Save detailed results to file")
    
    # Reproduction-status command
    subparsers.add_parser("reproduction-status", help="Show reproduction system status")
    
    # Prepare-bundle command
    bundle_parser = subparsers.add_parser("prepare-bundle", help="Prepare reproduction bundle")
    bundle_parser.add_argument("--output-dir", help="Output directory for bundle")
    bundle_parser.add_argument("--no-logs", action="store_true",
                              help="Exclude audit logs from bundle")
    bundle_parser.add_argument("--no-provenance", action="store_true",
                              help="Exclude provenance records from bundle")
    
    # Diagnostics command
    diagnostics_parser = subparsers.add_parser("diagnostics", help="Run comprehensive system diagnostics")
    diagnostics_parser.add_argument("--output", help="Save detailed results to file")
    diagnostics_parser.add_argument("--format", choices=["text", "json"], default="text",
                                   help="Output format (default: text)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        cli = DatasetCLI(args.registry_path, args.manifest_path)
        
        if args.command == "list":
            cli.list_datasets(
                format_type=args.format,
                filter_status=args.status,
                filter_type=args.type,
                show_details=args.details
            )
        elif args.command == "status":
            cli.show_dataset_status(args.dataset)
        elif args.command == "verify":
            cli.verify_dataset(args.dataset, args.force)
        elif args.command == "verify-all":
            cli.verify_all_datasets(args.force)
        elif args.command == "integrity":
            cli.check_integrity()
        elif args.command == "cleanup":
            dry_run = args.dry_run and not args.no_dry_run
            cli.cleanup_orphaned_files(dry_run)
        elif args.command == "export":
            cli.export_summary(args.output, args.format, args.type)
        elif args.command == "audit":
            cli.show_audit_trail(args.dataset, args.limit)
        elif args.command == "summary":
            cli.generate_summary(args.type, args.output, args.format)
        elif args.command == "fetch-all":
            parallel = args.parallel and not args.no_parallel
            cli.fetch_all_datasets(args.force_refresh, parallel, args.output)
        elif args.command == "reproduction-status":
            cli.reproduction_status()
        elif args.command == "prepare-bundle":
            include_logs = not args.no_logs
            include_provenance = not args.no_provenance
            cli.prepare_reproduction_bundle(args.output_dir, include_logs, include_provenance)
        elif args.command == "diagnostics":
            cli.run_diagnostics(args.output, args.format)
        
        return 0
        
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())