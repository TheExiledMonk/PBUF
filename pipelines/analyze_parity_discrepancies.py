#!/usr/bin/env python3
"""
Analyze parity test discrepancies and generate detailed diagnostic reports.

This script processes parity test results to identify patterns in discrepancies,
categorize issues, and provide recommendations for addressing them.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import json
import os
import glob
from typing import Dict, List, Any, Tuple
import numpy as np
from pathlib import Path
import argparse


def load_parity_reports(results_dir: str = "parity_results") -> List[Dict]:
    """Load all parity test reports from JSON files."""
    reports = []
    
    json_files = glob.glob(os.path.join(results_dir, "parity_report_*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                report = json.load(f)
                reports.append(report)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return reports


def analyze_discrepancies(reports: List[Dict]) -> Dict[str, Any]:
    """Analyze discrepancy patterns across all reports."""
    analysis = {
        "total_tests": len(reports),
        "total_comparisons": 0,
        "failed_comparisons": 0,
        "discrepancy_patterns": {},
        "metric_statistics": {},
        "model_performance": {"lcdm": {}, "pbuf": {}},
        "dataset_performance": {}
    }
    
    for report in reports:
        model = report.get("model", "unknown")
        datasets = report.get("datasets", [])
        dataset_key = "_".join(sorted(datasets))
        
        # Initialize counters
        if dataset_key not in analysis["dataset_performance"]:
            analysis["dataset_performance"][dataset_key] = {
                "total_tests": 0,
                "total_comparisons": 0,
                "failed_comparisons": 0,
                "common_failures": {}
            }
        
        analysis["dataset_performance"][dataset_key]["total_tests"] += 1
        
        # Process comparisons
        comparisons = report.get("comparisons", [])
        analysis["total_comparisons"] += len(comparisons)
        analysis["dataset_performance"][dataset_key]["total_comparisons"] += len(comparisons)
        
        for comp in comparisons:
            metric_name = comp.get("metric_name", "unknown")
            passes = comp.get("passes_tolerance", False)
            abs_diff = comp.get("absolute_diff", 0)
            rel_diff = comp.get("relative_diff", 0)
            
            if not passes:
                analysis["failed_comparisons"] += 1
                analysis["dataset_performance"][dataset_key]["failed_comparisons"] += 1
                
                # Track common failure patterns
                if metric_name not in analysis["dataset_performance"][dataset_key]["common_failures"]:
                    analysis["dataset_performance"][dataset_key]["common_failures"][metric_name] = 0
                analysis["dataset_performance"][dataset_key]["common_failures"][metric_name] += 1
                
                # Track discrepancy patterns
                category = metric_name.split('.')[0]
                if category not in analysis["discrepancy_patterns"]:
                    analysis["discrepancy_patterns"][category] = {
                        "count": 0,
                        "max_abs_diff": 0,
                        "max_rel_diff": 0,
                        "metrics": {}
                    }
                
                analysis["discrepancy_patterns"][category]["count"] += 1
                analysis["discrepancy_patterns"][category]["max_abs_diff"] = max(
                    analysis["discrepancy_patterns"][category]["max_abs_diff"], abs_diff
                )
                analysis["discrepancy_patterns"][category]["max_rel_diff"] = max(
                    analysis["discrepancy_patterns"][category]["max_rel_diff"], rel_diff
                )
                
                if metric_name not in analysis["discrepancy_patterns"][category]["metrics"]:
                    analysis["discrepancy_patterns"][category]["metrics"][metric_name] = {
                        "count": 0,
                        "total_abs_diff": 0,
                        "total_rel_diff": 0
                    }
                
                analysis["discrepancy_patterns"][category]["metrics"][metric_name]["count"] += 1
                analysis["discrepancy_patterns"][category]["metrics"][metric_name]["total_abs_diff"] += abs_diff
                analysis["discrepancy_patterns"][category]["metrics"][metric_name]["total_rel_diff"] += rel_diff
            
            # Track metric statistics
            if metric_name not in analysis["metric_statistics"]:
                analysis["metric_statistics"][metric_name] = {
                    "total_tests": 0,
                    "failures": 0,
                    "success_rate": 0,
                    "avg_abs_diff": 0,
                    "avg_rel_diff": 0
                }
            
            analysis["metric_statistics"][metric_name]["total_tests"] += 1
            if not passes:
                analysis["metric_statistics"][metric_name]["failures"] += 1
            analysis["metric_statistics"][metric_name]["avg_abs_diff"] += abs_diff
            analysis["metric_statistics"][metric_name]["avg_rel_diff"] += rel_diff
    
    # Calculate averages and success rates
    for metric_name, stats in analysis["metric_statistics"].items():
        total = stats["total_tests"]
        if total > 0:
            stats["success_rate"] = (total - stats["failures"]) / total * 100
            stats["avg_abs_diff"] /= total
            stats["avg_rel_diff"] /= total
    
    return analysis


def generate_discrepancy_report(analysis: Dict[str, Any]) -> str:
    """Generate a detailed discrepancy analysis report."""
    lines = []
    lines.append("=" * 80)
    lines.append("PARITY TEST DISCREPANCY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary statistics
    total_tests = analysis["total_tests"]
    total_comps = analysis["total_comparisons"]
    failed_comps = analysis["failed_comparisons"]
    success_rate = (total_comps - failed_comps) / total_comps * 100 if total_comps > 0 else 0
    
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total Tests Executed: {total_tests}")
    lines.append(f"Total Comparisons: {total_comps}")
    lines.append(f"Failed Comparisons: {failed_comps}")
    lines.append(f"Overall Success Rate: {success_rate:.1f}%")
    lines.append("")
    
    # Discrepancy patterns by category
    lines.append("DISCREPANCY PATTERNS BY CATEGORY")
    lines.append("-" * 40)
    for category, data in analysis["discrepancy_patterns"].items():
        lines.append(f"\n{category.upper()}:")
        lines.append(f"  Failed Comparisons: {data['count']}")
        lines.append(f"  Max Absolute Diff: {data['max_abs_diff']:.2e}")
        lines.append(f"  Max Relative Diff: {data['max_rel_diff']:.2e}")
        
        # Top failing metrics in this category
        metrics = data["metrics"]
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]["count"], reverse=True)
        lines.append(f"  Top Failing Metrics:")
        for metric, mdata in sorted_metrics[:3]:
            avg_abs = mdata["total_abs_diff"] / mdata["count"]
            avg_rel = mdata["total_rel_diff"] / mdata["count"]
            lines.append(f"    {metric}: {mdata['count']} failures, "
                        f"avg_abs={avg_abs:.2e}, avg_rel={avg_rel:.2e}")
    
    lines.append("")
    
    # Dataset performance analysis
    lines.append("DATASET PERFORMANCE ANALYSIS")
    lines.append("-" * 40)
    for dataset, data in analysis["dataset_performance"].items():
        total_comps = data["total_comparisons"]
        failed_comps = data["failed_comparisons"]
        success_rate = (total_comps - failed_comps) / total_comps * 100 if total_comps > 0 else 0
        
        lines.append(f"\n{dataset.upper()}:")
        lines.append(f"  Tests: {data['total_tests']}")
        lines.append(f"  Comparisons: {total_comps}")
        lines.append(f"  Success Rate: {success_rate:.1f}%")
        
        # Most common failures
        common_failures = data["common_failures"]
        if common_failures:
            sorted_failures = sorted(common_failures.items(), key=lambda x: x[1], reverse=True)
            lines.append(f"  Most Common Failures:")
            for metric, count in sorted_failures[:5]:
                lines.append(f"    {metric}: {count} failures")
    
    lines.append("")
    
    # Metric reliability analysis
    lines.append("METRIC RELIABILITY ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"{'Metric':<35} {'Tests':<6} {'Failures':<8} {'Success%':<8} {'Avg AbsDiff':<12}")
    lines.append("-" * 80)
    
    sorted_metrics = sorted(
        analysis["metric_statistics"].items(), 
        key=lambda x: x[1]["success_rate"]
    )
    
    for metric, stats in sorted_metrics:
        lines.append(f"{metric:<35} {stats['total_tests']:<6} {stats['failures']:<8} "
                    f"{stats['success_rate']:<8.1f} {stats['avg_abs_diff']:<12.2e}")
    
    lines.append("")
    
    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    
    # Identify high-priority issues
    high_failure_metrics = [
        (metric, stats) for metric, stats in analysis["metric_statistics"].items()
        if stats["success_rate"] < 50 and stats["total_tests"] >= 5
    ]
    
    if high_failure_metrics:
        lines.append("\n1. HIGH PRIORITY ISSUES:")
        for metric, stats in high_failure_metrics[:5]:
            lines.append(f"   - {metric}: {stats['success_rate']:.1f}% success rate")
            lines.append(f"     Action: Investigate systematic discrepancy")
    
    # Check for systematic patterns
    if "metrics" in analysis["discrepancy_patterns"]:
        metrics_failures = analysis["discrepancy_patterns"]["metrics"]["count"]
        total_metric_tests = sum(
            stats["total_tests"] for metric, stats in analysis["metric_statistics"].items()
            if metric.startswith("metrics.")
        )
        if metrics_failures / total_metric_tests > 0.8:
            lines.append("\n2. SYSTEMATIC METRIC DISCREPANCIES:")
            lines.append("   - Most statistical metrics are failing")
            lines.append("   - Action: Review mock legacy result generation")
            lines.append("   - Action: Validate unified system statistical calculations")
    
    # Performance recommendations
    lines.append("\n3. FRAMEWORK IMPROVEMENTS:")
    lines.append("   - Implement actual legacy system integration")
    lines.append("   - Create reference datasets with known good results")
    lines.append("   - Add convergence diagnostics to optimization")
    lines.append("   - Implement physics consistency cross-checks")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def identify_critical_discrepancies(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify critical discrepancies that require immediate attention."""
    critical_issues = []
    
    # Check for metrics with very high failure rates
    for metric, stats in analysis["metric_statistics"].items():
        if stats["success_rate"] < 10 and stats["total_tests"] >= 5:
            critical_issues.append({
                "type": "high_failure_rate",
                "metric": metric,
                "success_rate": stats["success_rate"],
                "total_tests": stats["total_tests"],
                "severity": "critical",
                "recommendation": f"Investigate systematic issue with {metric} calculation"
            })
    
    # Check for extremely large discrepancies
    for category, data in analysis["discrepancy_patterns"].items():
        if data["max_abs_diff"] > 1000:  # Very large absolute differences
            critical_issues.append({
                "type": "large_discrepancy",
                "category": category,
                "max_abs_diff": data["max_abs_diff"],
                "severity": "high",
                "recommendation": f"Review {category} calculation - extremely large discrepancies detected"
            })
    
    return critical_issues


def main():
    """Main entry point for discrepancy analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze parity test discrepancies and generate diagnostic reports"
    )
    parser.add_argument(
        "--results-dir", 
        default="parity_results",
        help="Directory containing parity test results"
    )
    parser.add_argument(
        "--output-file",
        default="parity_results/discrepancy_analysis.txt",
        help="Output file for analysis report"
    )
    parser.add_argument(
        "--json-output",
        default="parity_results/discrepancy_analysis.json",
        help="JSON output file for machine-readable analysis"
    )
    
    args = parser.parse_args()
    
    # Load parity reports
    print(f"Loading parity reports from {args.results_dir}...")
    reports = load_parity_reports(args.results_dir)
    
    if not reports:
        print("No parity reports found. Run parity tests first.")
        return 1
    
    print(f"Loaded {len(reports)} parity reports")
    
    # Analyze discrepancies
    print("Analyzing discrepancy patterns...")
    analysis = analyze_discrepancies(reports)
    
    # Generate report
    print("Generating discrepancy analysis report...")
    report_text = generate_discrepancy_report(analysis)
    
    # Save text report
    with open(args.output_file, 'w') as f:
        f.write(report_text)
    print(f"Analysis report saved to: {args.output_file}")
    
    # Save JSON analysis
    with open(args.json_output, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"JSON analysis saved to: {args.json_output}")
    
    # Identify critical issues
    critical_issues = identify_critical_discrepancies(analysis)
    if critical_issues:
        print(f"\n⚠️  {len(critical_issues)} critical issues identified:")
        for issue in critical_issues:
            print(f"   - {issue['type']}: {issue.get('metric', issue.get('category'))}")
            print(f"     {issue['recommendation']}")
    
    # Print summary
    total_comps = analysis["total_comparisons"]
    failed_comps = analysis["failed_comparisons"]
    success_rate = (total_comps - failed_comps) / total_comps * 100 if total_comps > 0 else 0
    
    print(f"\nSUMMARY:")
    print(f"  Total Tests: {analysis['total_tests']}")
    print(f"  Total Comparisons: {total_comps}")
    print(f"  Overall Success Rate: {success_rate:.1f}%")
    print(f"  Critical Issues: {len(critical_issues)}")
    
    return 0


if __name__ == "__main__":
    exit(main())