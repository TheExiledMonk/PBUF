"""
Export manager for dataset registry data

This module provides comprehensive export functionality for registry data,
audit trails, and provenance information in various formats suitable for
publication materials and external systems.
"""

import json
import csv
import io
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TextIO
from datetime import datetime
from dataclasses import dataclass

from ..core.registry_manager import RegistryManager
from ..core.manifest_schema import DatasetManifest


class ExportFormat(Enum):
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    LATEX = "latex"
    YAML = "yaml"


@dataclass
class ExportOptions:
    """Options for export operations"""
    include_metadata: bool = True
    include_provenance: bool = True
    include_verification: bool = True
    include_environment: bool = True
    include_audit_trail: bool = False
    filter_datasets: Optional[List[str]] = None
    filter_status: Optional[str] = None  # 'verified', 'failed', 'corrupted'
    filter_source_type: Optional[str] = None  # 'downloaded', 'manual'
    sort_by: str = "name"  # 'name', 'timestamp', 'status'
    sort_reverse: bool = False


class ExportManager:
    """
    Comprehensive export manager for dataset registry
    
    Provides export functionality for registry data, audit trails, and
    provenance information in multiple formats for publication materials,
    external monitoring systems, and compliance reporting.
    """
    
    def __init__(self, registry_manager: RegistryManager, manifest: Optional[DatasetManifest] = None):
        """
        Initialize export manager
        
        Args:
            registry_manager: Registry manager instance
            manifest: Dataset manifest (optional, for enhanced metadata)
        """
        self.registry_manager = registry_manager
        self.manifest = manifest
    
    def export_registry_summary(
        self,
        format_type: ExportFormat,
        options: Optional[ExportOptions] = None,
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Export comprehensive registry summary
        
        Args:
            format_type: Export format
            options: Export options (uses defaults if None)
            output_file: Output file path (returns string if None)
            
        Returns:
            Exported data as string (if output_file is None)
        """
        if options is None:
            options = ExportOptions()
        
        # Get registry data
        summary_data = self._collect_registry_data(options)
        
        # Format data according to requested format
        if format_type == ExportFormat.JSON:
            output = self._format_json(summary_data, options)
        elif format_type == ExportFormat.CSV:
            output = self._format_csv(summary_data, options)
        elif format_type == ExportFormat.MARKDOWN:
            output = self._format_markdown(summary_data, options)
        elif format_type == ExportFormat.LATEX:
            output = self._format_latex(summary_data, options)
        elif format_type == ExportFormat.YAML:
            output = self._format_yaml(summary_data, options)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        # Write to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)
        
        return output
    
    def export_audit_trail(
        self,
        format_type: ExportFormat,
        dataset_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Export audit trail data
        
        Args:
            format_type: Export format
            dataset_name: Filter by dataset name (all if None)
            start_date: Filter by start date (no limit if None)
            end_date: Filter by end date (no limit if None)
            output_file: Output file path (returns string if None)
            
        Returns:
            Exported audit trail as string (if output_file is None)
        """
        # Get audit trail entries
        entries = self.registry_manager.get_audit_trail(dataset_name)
        
        # Apply date filters
        if start_date or end_date:
            filtered_entries = []
            for entry in entries:
                try:
                    entry_date = datetime.fromisoformat(entry.get("timestamp", ""))
                    if start_date and entry_date < start_date:
                        continue
                    if end_date and entry_date > end_date:
                        continue
                    filtered_entries.append(entry)
                except (ValueError, TypeError):
                    # Include entries with invalid timestamps
                    filtered_entries.append(entry)
            entries = filtered_entries
        
        # Sort by timestamp (newest first)
        entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Format data
        audit_data = {
            "export_metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "dataset_filter": dataset_name,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "total_entries": len(entries)
            },
            "audit_entries": entries
        }
        
        if format_type == ExportFormat.JSON:
            output = json.dumps(audit_data, indent=2, sort_keys=True)
        elif format_type == ExportFormat.CSV:
            output = self._format_audit_csv(entries)
        elif format_type == ExportFormat.MARKDOWN:
            output = self._format_audit_markdown(audit_data)
        else:
            raise ValueError(f"Unsupported format for audit trail: {format_type}")
        
        # Write to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)
        
        return output
    
    def export_provenance_bundle(
        self,
        dataset_names: Optional[List[str]] = None,
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Export complete provenance bundle for publication materials
        
        Args:
            dataset_names: List of datasets to include (all if None)
            output_file: Output file path (returns string if None)
            
        Returns:
            Provenance bundle as JSON string (if output_file is None)
        """
        if dataset_names is None:
            dataset_names = self.registry_manager.list_datasets()
        
        # Get provenance summary from registry manager
        provenance_data = self.registry_manager.export_provenance_summary(dataset_names)
        
        # Enhance with manifest data if available
        if self.manifest:
            for dataset_name in dataset_names:
                if dataset_name in provenance_data["datasets"]:
                    try:
                        manifest_info = self.manifest.get_dataset_info(dataset_name)
                        dataset_entry = provenance_data["datasets"][dataset_name]
                        
                        # Add manifest metadata
                        dataset_entry.update({
                            "manifest_canonical_name": manifest_info.canonical_name,
                            "manifest_description": manifest_info.description,
                            "manifest_citation": manifest_info.citation,
                            "manifest_license": manifest_info.license,
                            "manifest_metadata": manifest_info.metadata
                        })
                    except KeyError:
                        # Dataset not in manifest (probably manual)
                        pass
        
        # Add publication-ready metadata
        provenance_data.update({
            "bundle_type": "dataset_provenance",
            "bundle_version": "1.0",
            "generated_for": "publication_materials",
            "compliance_note": "This bundle provides complete dataset provenance for scientific reproducibility"
        })
        
        output = json.dumps(provenance_data, indent=2, sort_keys=True)
        
        # Write to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)
        
        return output
    
    def export_manifest_table(
        self,
        format_type: ExportFormat,
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Export manifest summary table for publications
        
        Args:
            format_type: Export format (CSV, Markdown, or LaTeX)
            output_file: Output file path (returns string if None)
            
        Returns:
            Formatted table as string (if output_file is None)
        """
        if not self.manifest:
            raise ValueError("Manifest not available for table export")
        
        # Get manifest summary
        manifest_summary = self.manifest.export_summary()
        datasets = manifest_summary["datasets"]
        
        # Enhance with registry status if available
        for dataset in datasets:
            dataset_name = dataset["name"]
            if self.registry_manager.has_registry_entry(dataset_name):
                entry = self.registry_manager.get_registry_entry(dataset_name)
                if entry:
                    dataset["registry_status"] = "verified" if entry.verification.is_valid else "failed"
                    dataset["last_verified"] = entry.verification.verification_timestamp
                else:
                    dataset["registry_status"] = "not_registered"
            else:
                dataset["registry_status"] = "not_registered"
        
        # Format table
        if format_type == ExportFormat.CSV:
            output = self._format_manifest_csv(datasets)
        elif format_type == ExportFormat.MARKDOWN:
            output = self._format_manifest_markdown(datasets)
        elif format_type == ExportFormat.LATEX:
            output = self._format_manifest_latex(datasets)
        else:
            raise ValueError(f"Unsupported format for manifest table: {format_type}")
        
        # Write to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)
        
        return output
    
    def _collect_registry_data(self, options: ExportOptions) -> Dict[str, Any]:
        """Collect registry data according to export options"""
        # Get base registry summary
        registry_summary = self.registry_manager.get_registry_summary()
        datasets = registry_summary["datasets"]
        
        # Apply filters
        if options.filter_datasets:
            datasets = [d for d in datasets if d["name"] in options.filter_datasets]
        
        if options.filter_status:
            datasets = [d for d in datasets if d["status"] == options.filter_status]
        
        if options.filter_source_type:
            datasets = [d for d in datasets if d["source_type"] == options.filter_source_type]
        
        # Sort datasets
        if options.sort_by == "name":
            datasets.sort(key=lambda x: x["name"], reverse=options.sort_reverse)
        elif options.sort_by == "timestamp":
            datasets.sort(key=lambda x: x.get("last_verified", ""), reverse=options.sort_reverse)
        elif options.sort_by == "status":
            datasets.sort(key=lambda x: x["status"], reverse=options.sort_reverse)
        
        # Enhance with detailed information if requested
        if options.include_metadata or options.include_provenance or options.include_verification:
            enhanced_datasets = []
            for dataset in datasets:
                enhanced_dataset = dataset.copy()
                
                try:
                    entry = self.registry_manager.get_registry_entry(dataset["name"])
                    if entry:
                        if options.include_verification:
                            enhanced_dataset["verification_details"] = {
                                "sha256_verified": entry.verification.sha256_verified,
                                "sha256_expected": entry.verification.sha256_expected,
                                "sha256_actual": entry.verification.sha256_actual,
                                "size_verified": entry.verification.size_verified,
                                "size_expected": entry.verification.size_expected,
                                "size_actual": entry.verification.size_actual,
                                "schema_verified": entry.verification.schema_verified,
                                "schema_errors": entry.verification.schema_errors
                            }
                        
                        if options.include_environment:
                            enhanced_dataset["environment"] = {
                                "pbuf_commit": entry.environment.pbuf_commit,
                                "python_version": entry.environment.python_version,
                                "platform": entry.environment.platform,
                                "hostname": entry.environment.hostname
                            }
                        
                        if options.include_provenance:
                            enhanced_dataset["provenance"] = {
                                "download_timestamp": entry.download_timestamp,
                                "source_used": entry.source_used,
                                "download_agent": entry.download_agent,
                                "file_path": entry.file_info["local_path"]
                            }
                
                except Exception:
                    # Skip enhancement for corrupted entries
                    pass
                
                # Add manifest metadata if available
                if options.include_metadata and self.manifest:
                    try:
                        manifest_info = self.manifest.get_dataset_info(dataset["name"])
                        enhanced_dataset["manifest_metadata"] = {
                            "canonical_name": manifest_info.canonical_name,
                            "description": manifest_info.description,
                            "citation": manifest_info.citation,
                            "license": manifest_info.license,
                            "metadata": manifest_info.metadata
                        }
                    except KeyError:
                        # Dataset not in manifest (probably manual)
                        pass
                
                enhanced_datasets.append(enhanced_dataset)
            
            datasets = enhanced_datasets
        
        return {
            "export_metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "export_options": {
                    "include_metadata": options.include_metadata,
                    "include_provenance": options.include_provenance,
                    "include_verification": options.include_verification,
                    "include_environment": options.include_environment,
                    "filter_datasets": options.filter_datasets,
                    "filter_status": options.filter_status,
                    "filter_source_type": options.filter_source_type,
                    "sort_by": options.sort_by,
                    "sort_reverse": options.sort_reverse
                },
                "total_datasets": len(datasets)
            },
            "registry_summary": {
                "total_datasets": registry_summary["total_datasets"],
                "verified_datasets": registry_summary["verified_datasets"],
                "failed_datasets": registry_summary["failed_datasets"],
                "downloaded_datasets": registry_summary["downloaded_datasets"],
                "manual_datasets": registry_summary["manual_datasets"]
            },
            "datasets": datasets
        }
    
    def _format_json(self, data: Dict[str, Any], options: ExportOptions) -> str:
        """Format data as JSON"""
        return json.dumps(data, indent=2, sort_keys=True)
    
    def _format_csv(self, data: Dict[str, Any], options: ExportOptions) -> str:
        """Format data as CSV"""
        output = io.StringIO()
        
        if not data["datasets"]:
            return "No datasets found"
        
        # Determine columns based on options
        base_columns = ["name", "status", "source_type", "last_verified"]
        
        if options.include_provenance:
            base_columns.extend(["download_timestamp", "source_used", "download_agent"])
        
        if options.include_verification:
            base_columns.extend(["sha256_verified", "size_verified", "schema_verified"])
        
        if options.include_environment:
            base_columns.extend(["pbuf_commit", "python_version", "platform"])
        
        writer = csv.DictWriter(output, fieldnames=base_columns, extrasaction='ignore')
        writer.writeheader()
        
        for dataset in data["datasets"]:
            row = {col: dataset.get(col, "") for col in base_columns}
            
            # Flatten nested data
            if options.include_provenance and "provenance" in dataset:
                prov = dataset["provenance"]
                row.update({
                    "download_timestamp": prov.get("download_timestamp", ""),
                    "source_used": prov.get("source_used", ""),
                    "download_agent": prov.get("download_agent", "")
                })
            
            if options.include_verification and "verification_details" in dataset:
                verif = dataset["verification_details"]
                row.update({
                    "sha256_verified": verif.get("sha256_verified", ""),
                    "size_verified": verif.get("size_verified", ""),
                    "schema_verified": verif.get("schema_verified", "")
                })
            
            if options.include_environment and "environment" in dataset:
                env = dataset["environment"]
                row.update({
                    "pbuf_commit": env.get("pbuf_commit", ""),
                    "python_version": env.get("python_version", ""),
                    "platform": env.get("platform", "")
                })
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def _format_markdown(self, data: Dict[str, Any], options: ExportOptions) -> str:
        """Format data as Markdown"""
        lines = [
            "# Dataset Registry Summary",
            "",
            f"**Export Date:** {data['export_metadata']['export_timestamp']}",
            f"**Total Datasets:** {data['export_metadata']['total_datasets']}",
            ""
        ]
        
        # Add summary statistics
        summary = data["registry_summary"]
        lines.extend([
            "## Summary Statistics",
            "",
            f"- **Total Datasets:** {summary['total_datasets']}",
            f"- **Verified Datasets:** {summary['verified_datasets']}",
            f"- **Failed Datasets:** {summary['failed_datasets']}",
            f"- **Downloaded Datasets:** {summary['downloaded_datasets']}",
            f"- **Manual Datasets:** {summary['manual_datasets']}",
            ""
        ])
        
        # Add datasets table
        if data["datasets"]:
            lines.extend([
                "## Datasets",
                "",
                "| Dataset | Status | Source Type | Last Verified |",
                "|---------|--------|-------------|---------------|"
            ])
            
            for dataset in data["datasets"]:
                status_icon = "✓" if dataset["status"] == "verified" else "✗"
                last_verified = dataset.get("last_verified", "Unknown")
                if last_verified and last_verified != "Unknown":
                    try:
                        dt = datetime.fromisoformat(last_verified.replace('Z', '+00:00'))
                        last_verified = dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                lines.append(f"| {dataset['name']} | {status_icon} {dataset['status']} | {dataset['source_type']} | {last_verified} |")
        
        return "\n".join(lines)
    
    def _format_latex(self, data: Dict[str, Any], options: ExportOptions) -> str:
        """Format data as LaTeX table"""
        lines = [
            "% Dataset Registry Summary Table",
            "% Generated on " + data['export_metadata']['export_timestamp'],
            "",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Dataset Registry Summary}",
            "\\label{tab:dataset_registry}",
            "\\begin{tabular}{llll}",
            "\\toprule",
            "Dataset & Status & Source Type & Last Verified \\\\",
            "\\midrule"
        ]
        
        for dataset in data["datasets"]:
            status_symbol = "$\\checkmark$" if dataset["status"] == "verified" else "$\\times$"
            last_verified = dataset.get("last_verified", "Unknown")
            if last_verified and last_verified != "Unknown":
                try:
                    dt = datetime.fromisoformat(last_verified.replace('Z', '+00:00'))
                    last_verified = dt.strftime("%Y-%m-%d")
                except:
                    pass
            
            # Escape LaTeX special characters
            name = dataset['name'].replace('_', '\\_')
            source_type = dataset['source_type'].replace('_', '\\_')
            
            lines.append(f"{name} & {status_symbol} & {source_type} & {last_verified} \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
    def _format_yaml(self, data: Dict[str, Any], options: ExportOptions) -> str:
        """Format data as YAML"""
        try:
            import yaml
            return yaml.dump(data, default_flow_style=False, sort_keys=True)
        except ImportError:
            raise ValueError("YAML export requires PyYAML package: pip install PyYAML")
    
    def _format_audit_csv(self, entries: List[Dict[str, Any]]) -> str:
        """Format audit trail as CSV"""
        if not entries:
            return "timestamp,event,dataset_name,details\n"
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "event", "dataset_name", "details"])
        
        for entry in entries:
            details = []
            for key, value in entry.items():
                if key not in ["timestamp", "event", "dataset_name"]:
                    details.append(f"{key}={value}")
            
            writer.writerow([
                entry.get("timestamp", ""),
                entry.get("event", ""),
                entry.get("dataset_name", ""),
                "; ".join(details)
            ])
        
        return output.getvalue()
    
    def _format_audit_markdown(self, audit_data: Dict[str, Any]) -> str:
        """Format audit trail as Markdown"""
        lines = [
            "# Dataset Registry Audit Trail",
            "",
            f"**Export Date:** {audit_data['export_metadata']['export_timestamp']}",
            f"**Total Entries:** {audit_data['export_metadata']['total_entries']}"
        ]
        
        if audit_data['export_metadata']['dataset_filter']:
            lines.append(f"**Dataset Filter:** {audit_data['export_metadata']['dataset_filter']}")
        
        lines.extend(["", "## Audit Entries", ""])
        
        for entry in audit_data["audit_entries"]:
            timestamp = entry.get("timestamp", "Unknown")
            event = entry.get("event", "Unknown")
            dataset = entry.get("dataset_name", "Unknown")
            
            lines.append(f"### {timestamp}")
            lines.append(f"- **Event:** {event}")
            lines.append(f"- **Dataset:** {dataset}")
            
            # Add event-specific details
            for key, value in entry.items():
                if key not in ["timestamp", "event", "dataset_name"]:
                    lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_manifest_csv(self, datasets: List[Dict[str, Any]]) -> str:
        """Format manifest table as CSV"""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "name", "canonical_name", "type", "citation", "registry_status"
        ])
        writer.writeheader()
        
        for dataset in datasets:
            writer.writerow({
                "name": dataset["name"],
                "canonical_name": dataset["canonical_name"],
                "type": dataset.get("type", ""),
                "citation": dataset["citation"],
                "registry_status": dataset.get("registry_status", "unknown")
            })
        
        return output.getvalue()
    
    def _format_manifest_markdown(self, datasets: List[Dict[str, Any]]) -> str:
        """Format manifest table as Markdown"""
        lines = [
            "# Dataset Manifest Summary",
            "",
            "| Dataset | Canonical Name | Type | Citation | Status |",
            "|---------|----------------|------|----------|--------|"
        ]
        
        for dataset in datasets:
            status = dataset.get("registry_status", "unknown")
            status_icon = "✓" if status == "verified" else "✗" if status == "failed" else "?"
            
            lines.append(
                f"| {dataset['name']} | {dataset['canonical_name']} | "
                f"{dataset.get('type', '')} | {dataset['citation'][:50]}... | "
                f"{status_icon} {status} |"
            )
        
        return "\n".join(lines)
    
    def _format_manifest_latex(self, datasets: List[Dict[str, Any]]) -> str:
        """Format manifest table as LaTeX"""
        lines = [
            "% Dataset Manifest Summary Table",
            "",
            "\\begin{longtable}{p{3cm}p{4cm}p{2cm}p{5cm}p{2cm}}",
            "\\caption{Dataset Manifest Summary} \\\\",
            "\\toprule",
            "Dataset & Canonical Name & Type & Citation & Status \\\\",
            "\\midrule",
            "\\endfirsthead",
            "\\multicolumn{5}{c}{\\tablename\\ \\thetable\\ -- \\textit{Continued from previous page}} \\\\",
            "\\toprule",
            "Dataset & Canonical Name & Type & Citation & Status \\\\",
            "\\midrule",
            "\\endhead",
            "\\midrule",
            "\\multicolumn{5}{r}{\\textit{Continued on next page}} \\\\",
            "\\endfoot",
            "\\bottomrule",
            "\\endlastfoot"
        ]
        
        for dataset in datasets:
            status = dataset.get("registry_status", "unknown")
            status_symbol = "$\\checkmark$" if status == "verified" else "$\\times$" if status == "failed" else "?"
            
            # Escape LaTeX special characters
            name = dataset['name'].replace('_', '\\_')
            canonical = dataset['canonical_name'].replace('&', '\\&').replace('_', '\\_')
            citation = dataset['citation'][:50].replace('&', '\\&').replace('_', '\\_') + "..."
            dataset_type = dataset.get('type', '').replace('_', '\\_')
            
            lines.append(f"{name} & {canonical} & {dataset_type} & {citation} & {status_symbol} \\\\")
        
        lines.append("\\end{longtable}")
        
        return "\n".join(lines)