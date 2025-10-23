"""
Summary generator for dataset registry reports

This module provides specialized summary generation for different types of
reports including publication materials, compliance reports, and operational
dashboards.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

from ..core.registry_manager import RegistryManager
from ..core.manifest_schema import DatasetManifest


class SummaryType(Enum):
    """Types of summaries that can be generated"""
    PUBLICATION = "publication"  # For papers and reports
    COMPLIANCE = "compliance"    # For audit and compliance
    OPERATIONAL = "operational"  # For monitoring and operations
    EXECUTIVE = "executive"      # High-level overview
    TECHNICAL = "technical"      # Detailed technical information


@dataclass
class SummaryOptions:
    """Options for summary generation"""
    include_statistics: bool = True
    include_trends: bool = False
    include_issues: bool = True
    include_recommendations: bool = False
    time_period_days: Optional[int] = None  # Limit to recent activity
    group_by_type: bool = True
    group_by_source: bool = True


class SummaryGenerator:
    """
    Specialized summary generator for dataset registry
    
    Generates different types of summaries tailored for specific audiences
    and use cases including publication materials, compliance reports,
    and operational dashboards.
    """
    
    def __init__(self, registry_manager: RegistryManager, manifest: Optional[DatasetManifest] = None):
        """
        Initialize summary generator
        
        Args:
            registry_manager: Registry manager instance
            manifest: Dataset manifest (optional, for enhanced metadata)
        """
        self.registry_manager = registry_manager
        self.manifest = manifest
    
    def generate_summary(
        self,
        summary_type: SummaryType,
        options: Optional[SummaryOptions] = None
    ) -> Dict[str, Any]:
        """
        Generate summary based on type and options
        
        Args:
            summary_type: Type of summary to generate
            options: Summary options (uses defaults if None)
            
        Returns:
            Dictionary containing the generated summary
        """
        if options is None:
            options = SummaryOptions()
        
        # Get base data
        base_data = self._collect_base_data(options)
        
        # Generate summary based on type
        if summary_type == SummaryType.PUBLICATION:
            return self._generate_publication_summary(base_data, options)
        elif summary_type == SummaryType.COMPLIANCE:
            return self._generate_compliance_summary(base_data, options)
        elif summary_type == SummaryType.OPERATIONAL:
            return self._generate_operational_summary(base_data, options)
        elif summary_type == SummaryType.EXECUTIVE:
            return self._generate_executive_summary(base_data, options)
        elif summary_type == SummaryType.TECHNICAL:
            return self._generate_technical_summary(base_data, options)
        else:
            raise ValueError(f"Unsupported summary type: {summary_type}")
    
    def generate_publication_dataset_table(self) -> Dict[str, Any]:
        """
        Generate dataset table specifically formatted for publication materials
        
        Returns:
            Dictionary with publication-ready dataset information
        """
        datasets = []
        
        # Get all datasets from registry
        for dataset_name in self.registry_manager.list_datasets():
            try:
                entry = self.registry_manager.get_registry_entry(dataset_name)
                if not entry:
                    continue
                
                dataset_info = {
                    "name": dataset_name,
                    "status": "verified" if entry.verification.is_valid else "failed",
                    "source_type": "manual" if entry.source_used == "manual" else "downloaded",
                    "sha256": entry.verification.sha256_actual[:16] + "..." if entry.verification.sha256_actual else "N/A",
                    "size_mb": round(entry.verification.size_actual / 1024 / 1024, 2) if entry.verification.size_actual else "N/A"
                }
                
                # Add manifest information if available
                if self.manifest and self.manifest.has_dataset(dataset_name):
                    manifest_info = self.manifest.get_dataset_info(dataset_name)
                    dataset_info.update({
                        "canonical_name": manifest_info.canonical_name,
                        "citation": manifest_info.citation,
                        "description": manifest_info.description,
                        "license": manifest_info.license or "Not specified"
                    })
                    
                    if manifest_info.metadata:
                        dataset_info.update({
                            "type": manifest_info.metadata.get("dataset_type", "Unknown"),
                            "n_data_points": manifest_info.metadata.get("n_data_points", "Unknown"),
                            "observables": ", ".join(manifest_info.metadata.get("observables", []))
                        })
                else:
                    # For manual datasets, get info from registry
                    manual_info = self.registry_manager.get_manual_dataset_info(dataset_name)
                    if manual_info:
                        dataset_info.update({
                            "canonical_name": manual_info.get("canonical_name", dataset_name),
                            "citation": manual_info.get("citation", "Not specified"),
                            "description": manual_info.get("description", ""),
                            "license": manual_info.get("license", "Not specified"),
                            "type": manual_info.get("metadata", {}).get("dataset_type", "Unknown")
                        })
                
                datasets.append(dataset_info)
                
            except Exception:
                continue
        
        # Sort by dataset type and name
        datasets.sort(key=lambda x: (x.get("type", ""), x["name"]))
        
        return {
            "publication_table": {
                "title": "Dataset Summary for Publication",
                "generated_at": datetime.now().isoformat(),
                "total_datasets": len(datasets),
                "datasets": datasets
            }
        }
    
    def generate_compliance_report(self, include_audit_trail: bool = True) -> Dict[str, Any]:
        """
        Generate compliance report for audit purposes
        
        Args:
            include_audit_trail: Whether to include full audit trail
            
        Returns:
            Dictionary with compliance report data
        """
        # Get registry integrity results
        integrity_results = self.registry_manager.validate_registry_integrity()
        
        # Get audit trail if requested
        audit_trail = []
        if include_audit_trail:
            audit_trail = self.registry_manager.get_audit_trail()
        
        # Analyze compliance status
        compliance_status = self._analyze_compliance_status(integrity_results, audit_trail)
        
        return {
            "compliance_report": {
                "report_type": "dataset_registry_compliance",
                "generated_at": datetime.now().isoformat(),
                "report_period": "all_time",
                "compliance_status": compliance_status,
                "registry_integrity": integrity_results,
                "audit_summary": {
                    "total_audit_entries": len(audit_trail),
                    "audit_trail_available": include_audit_trail,
                    "audit_entries": audit_trail if include_audit_trail else []
                },
                "recommendations": self._generate_compliance_recommendations(integrity_results)
            }
        }
    
    def generate_operational_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate data for operational monitoring dashboard
        
        Returns:
            Dictionary with operational metrics and status
        """
        # Get current status
        registry_summary = self.registry_manager.get_registry_summary()
        
        # Calculate health metrics
        total_datasets = registry_summary["total_datasets"]
        verified_datasets = registry_summary["verified_datasets"]
        failed_datasets = registry_summary["failed_datasets"]
        
        health_score = (verified_datasets / total_datasets * 100) if total_datasets > 0 else 100
        
        # Get recent activity (last 7 days)
        recent_activity = self._get_recent_activity(days=7)
        
        # Identify issues
        issues = self._identify_operational_issues(registry_summary)
        
        return {
            "operational_dashboard": {
                "generated_at": datetime.now().isoformat(),
                "health_metrics": {
                    "overall_health_score": round(health_score, 1),
                    "total_datasets": total_datasets,
                    "verified_datasets": verified_datasets,
                    "failed_datasets": failed_datasets,
                    "health_status": "healthy" if health_score >= 95 else "warning" if health_score >= 80 else "critical"
                },
                "recent_activity": recent_activity,
                "active_issues": issues,
                "dataset_breakdown": {
                    "by_source": {
                        "downloaded": registry_summary["downloaded_datasets"],
                        "manual": registry_summary["manual_datasets"]
                    },
                    "by_status": {
                        "verified": verified_datasets,
                        "failed": failed_datasets
                    }
                }
            }
        }
    
    def _collect_base_data(self, options: SummaryOptions) -> Dict[str, Any]:
        """Collect base data for summary generation"""
        # Get registry summary
        registry_summary = self.registry_manager.get_registry_summary()
        
        # Get manifest summary if available
        manifest_summary = None
        if self.manifest:
            manifest_summary = self.manifest.export_summary()
        
        # Get audit trail if needed
        audit_trail = []
        if options.include_trends or options.time_period_days:
            audit_trail = self.registry_manager.get_audit_trail()
            
            # Filter by time period if specified
            if options.time_period_days:
                cutoff_date = datetime.now() - timedelta(days=options.time_period_days)
                audit_trail = [
                    entry for entry in audit_trail
                    if self._parse_timestamp(entry.get("timestamp")) >= cutoff_date
                ]
        
        return {
            "registry_summary": registry_summary,
            "manifest_summary": manifest_summary,
            "audit_trail": audit_trail,
            "collection_timestamp": datetime.now().isoformat()
        }
    
    def _generate_publication_summary(self, base_data: Dict[str, Any], options: SummaryOptions) -> Dict[str, Any]:
        """Generate summary for publication materials"""
        registry_summary = base_data["registry_summary"]
        
        # Create publication-focused summary
        summary = {
            "summary_type": "publication",
            "generated_at": base_data["collection_timestamp"],
            "dataset_overview": {
                "total_datasets": registry_summary["total_datasets"],
                "verified_datasets": registry_summary["verified_datasets"],
                "data_sources": {
                    "public_repositories": registry_summary["downloaded_datasets"],
                    "proprietary_datasets": registry_summary["manual_datasets"]
                }
            }
        }
        
        # Add dataset details
        if options.include_statistics:
            summary["dataset_statistics"] = self._calculate_dataset_statistics(base_data)
        
        # Add dataset groupings
        if options.group_by_type and base_data["manifest_summary"]:
            summary["datasets_by_type"] = base_data["manifest_summary"]["datasets_by_type"]
        
        # Add verification status
        summary["data_integrity"] = {
            "verification_status": "all_verified" if registry_summary["failed_datasets"] == 0 else "partial_verification",
            "verification_coverage": f"{registry_summary['verified_datasets']}/{registry_summary['total_datasets']} datasets verified"
        }
        
        return summary
    
    def _generate_compliance_summary(self, base_data: Dict[str, Any], options: SummaryOptions) -> Dict[str, Any]:
        """Generate summary for compliance reporting"""
        registry_summary = base_data["registry_summary"]
        
        # Validate registry integrity
        integrity_results = self.registry_manager.validate_registry_integrity()
        
        summary = {
            "summary_type": "compliance",
            "generated_at": base_data["collection_timestamp"],
            "compliance_status": {
                "registry_integrity": "compliant" if integrity_results["valid"] else "non_compliant",
                "audit_trail_available": len(base_data["audit_trail"]) > 0,
                "provenance_tracking": "enabled",
                "data_verification": "mandatory"
            },
            "integrity_assessment": integrity_results
        }
        
        if options.include_issues:
            summary["identified_issues"] = integrity_results.get("errors", [])
            summary["warnings"] = integrity_results.get("warnings", [])
        
        if options.include_recommendations:
            summary["recommendations"] = self._generate_compliance_recommendations(integrity_results)
        
        return summary
    
    def _generate_operational_summary(self, base_data: Dict[str, Any], options: SummaryOptions) -> Dict[str, Any]:
        """Generate summary for operational monitoring"""
        registry_summary = base_data["registry_summary"]
        
        summary = {
            "summary_type": "operational",
            "generated_at": base_data["collection_timestamp"],
            "system_status": {
                "total_datasets": registry_summary["total_datasets"],
                "operational_datasets": registry_summary["verified_datasets"],
                "failed_datasets": registry_summary["failed_datasets"],
                "system_health": "operational" if registry_summary["failed_datasets"] == 0 else "degraded"
            }
        }
        
        if options.include_trends and base_data["audit_trail"]:
            summary["activity_trends"] = self._analyze_activity_trends(base_data["audit_trail"])
        
        if options.include_issues:
            summary["operational_issues"] = self._identify_operational_issues(registry_summary)
        
        return summary
    
    def _generate_executive_summary(self, base_data: Dict[str, Any], options: SummaryOptions) -> Dict[str, Any]:
        """Generate high-level executive summary"""
        registry_summary = base_data["registry_summary"]
        
        # Calculate key metrics
        total_datasets = registry_summary["total_datasets"]
        success_rate = (registry_summary["verified_datasets"] / total_datasets * 100) if total_datasets > 0 else 100
        
        summary = {
            "summary_type": "executive",
            "generated_at": base_data["collection_timestamp"],
            "key_metrics": {
                "total_datasets_managed": total_datasets,
                "data_integrity_rate": f"{success_rate:.1f}%",
                "system_status": "operational" if success_rate >= 95 else "attention_required",
                "data_sources": {
                    "external_repositories": registry_summary["downloaded_datasets"],
                    "internal_datasets": registry_summary["manual_datasets"]
                }
            },
            "executive_summary": self._generate_executive_narrative(registry_summary, success_rate)
        }
        
        return summary
    
    def _generate_technical_summary(self, base_data: Dict[str, Any], options: SummaryOptions) -> Dict[str, Any]:
        """Generate detailed technical summary"""
        registry_summary = base_data["registry_summary"]
        
        summary = {
            "summary_type": "technical",
            "generated_at": base_data["collection_timestamp"],
            "technical_details": {
                "registry_statistics": registry_summary,
                "verification_engine_status": "operational",
                "download_protocols": ["https", "zenodo", "manual"],
                "supported_formats": ["ascii_table", "json", "fits"]
            }
        }
        
        if base_data["manifest_summary"]:
            summary["manifest_details"] = base_data["manifest_summary"]
        
        # Add detailed verification information
        summary["verification_details"] = self._get_detailed_verification_info()
        
        return summary
    
    def _calculate_dataset_statistics(self, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical information about datasets"""
        registry_summary = base_data["registry_summary"]
        
        stats = {
            "total_count": registry_summary["total_datasets"],
            "verification_rate": registry_summary["verified_datasets"] / registry_summary["total_datasets"] if registry_summary["total_datasets"] > 0 else 0,
            "source_distribution": {
                "downloaded": registry_summary["downloaded_datasets"],
                "manual": registry_summary["manual_datasets"]
            }
        }
        
        # Calculate size statistics if possible
        total_size = 0
        size_count = 0
        
        for dataset_name in self.registry_manager.list_datasets():
            try:
                entry = self.registry_manager.get_registry_entry(dataset_name)
                if entry and entry.verification.size_actual:
                    total_size += entry.verification.size_actual
                    size_count += 1
            except Exception:
                continue
        
        if size_count > 0:
            stats["storage_statistics"] = {
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "average_size_mb": round(total_size / size_count / 1024 / 1024, 2),
                "datasets_with_size_info": size_count
            }
        
        return stats
    
    def _analyze_compliance_status(self, integrity_results: Dict[str, Any], audit_trail: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze compliance status based on integrity and audit data"""
        compliance_score = 100
        
        # Deduct points for integrity issues
        if not integrity_results["valid"]:
            compliance_score -= 30
        
        if integrity_results["corrupted_entries"] > 0:
            compliance_score -= 20
        
        # Check audit trail completeness
        if len(audit_trail) == 0:
            compliance_score -= 10
        
        return {
            "overall_score": max(0, compliance_score),
            "status": "compliant" if compliance_score >= 90 else "non_compliant",
            "integrity_status": "pass" if integrity_results["valid"] else "fail",
            "audit_trail_status": "available" if len(audit_trail) > 0 else "missing"
        }
    
    def _generate_compliance_recommendations(self, integrity_results: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations based on integrity results"""
        recommendations = []
        
        if not integrity_results["valid"]:
            recommendations.append("Address registry integrity issues to ensure compliance")
        
        if integrity_results["corrupted_entries"] > 0:
            recommendations.append("Repair or remove corrupted registry entries")
        
        if integrity_results["warnings"]:
            recommendations.append("Review and address registry warnings")
        
        if not recommendations:
            recommendations.append("Registry is compliant - maintain current practices")
        
        return recommendations
    
    def _get_recent_activity(self, days: int = 7) -> Dict[str, Any]:
        """Get recent activity summary"""
        cutoff_date = datetime.now() - timedelta(days=days)
        audit_trail = self.registry_manager.get_audit_trail()
        
        recent_entries = [
            entry for entry in audit_trail
            if self._parse_timestamp(entry.get("timestamp")) >= cutoff_date
        ]
        
        # Count activity by type
        activity_counts = {}
        for entry in recent_entries:
            event_type = entry.get("event", "unknown")
            activity_counts[event_type] = activity_counts.get(event_type, 0) + 1
        
        return {
            "period_days": days,
            "total_activities": len(recent_entries),
            "activity_breakdown": activity_counts,
            "most_recent": recent_entries[0] if recent_entries else None
        }
    
    def _identify_operational_issues(self, registry_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify operational issues from registry summary"""
        issues = []
        
        if registry_summary["failed_datasets"] > 0:
            issues.append({
                "type": "verification_failures",
                "severity": "high" if registry_summary["failed_datasets"] > registry_summary["total_datasets"] * 0.1 else "medium",
                "description": f"{registry_summary['failed_datasets']} datasets have failed verification",
                "affected_datasets": registry_summary["failed_datasets"]
            })
        
        if registry_summary["total_datasets"] == 0:
            issues.append({
                "type": "no_datasets",
                "severity": "high",
                "description": "No datasets are registered in the system",
                "affected_datasets": 0
            })
        
        return issues
    
    def _analyze_activity_trends(self, audit_trail: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze activity trends from audit trail"""
        if not audit_trail:
            return {"trend": "no_data"}
        
        # Group by day
        daily_activity = {}
        for entry in audit_trail:
            timestamp = self._parse_timestamp(entry.get("timestamp"))
            if timestamp:
                date_key = timestamp.date().isoformat()
                daily_activity[date_key] = daily_activity.get(date_key, 0) + 1
        
        # Calculate trend
        dates = sorted(daily_activity.keys())
        if len(dates) >= 2:
            recent_avg = sum(daily_activity[date] for date in dates[-3:]) / min(3, len(dates))
            older_avg = sum(daily_activity[date] for date in dates[:-3]) / max(1, len(dates) - 3)
            
            if recent_avg > older_avg * 1.2:
                trend = "increasing"
            elif recent_avg < older_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "daily_activity": daily_activity,
            "total_days": len(dates),
            "average_daily_activity": sum(daily_activity.values()) / len(dates) if dates else 0
        }
    
    def _generate_executive_narrative(self, registry_summary: Dict[str, Any], success_rate: float) -> str:
        """Generate executive narrative summary"""
        total = registry_summary["total_datasets"]
        
        if total == 0:
            return "No datasets are currently managed by the registry system."
        
        narrative = f"The dataset registry currently manages {total} datasets with a {success_rate:.1f}% verification success rate. "
        
        if success_rate >= 95:
            narrative += "The system is operating at optimal performance with all datasets properly verified and accessible."
        elif success_rate >= 80:
            narrative += "The system is operating normally with minor verification issues that should be addressed."
        else:
            narrative += "The system requires attention due to significant verification failures affecting data integrity."
        
        if registry_summary["manual_datasets"] > 0:
            narrative += f" The registry includes {registry_summary['manual_datasets']} manually registered proprietary datasets alongside {registry_summary['downloaded_datasets']} datasets from public repositories."
        
        return narrative
    
    def _get_detailed_verification_info(self) -> Dict[str, Any]:
        """Get detailed verification information for technical summary"""
        verification_details = {
            "verification_methods": ["sha256_checksum", "file_size", "schema_validation"],
            "supported_schemas": ["ascii_table", "json", "fits"],
            "verification_statistics": {}
        }
        
        # Count verification results
        sha256_verified = 0
        size_verified = 0
        schema_verified = 0
        total_verified = 0
        
        for dataset_name in self.registry_manager.list_datasets():
            try:
                entry = self.registry_manager.get_registry_entry(dataset_name)
                if entry:
                    total_verified += 1
                    if entry.verification.sha256_verified:
                        sha256_verified += 1
                    if entry.verification.size_verified:
                        size_verified += 1
                    if entry.verification.schema_verified:
                        schema_verified += 1
            except Exception:
                continue
        
        if total_verified > 0:
            verification_details["verification_statistics"] = {
                "sha256_success_rate": sha256_verified / total_verified,
                "size_success_rate": size_verified / total_verified,
                "schema_success_rate": schema_verified / total_verified,
                "total_datasets_analyzed": total_verified
            }
        
        return verification_details
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime object"""
        if not timestamp_str:
            return None
        
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None