#!/usr/bin/env python3
"""Regenerate enhanced jackknife report for existing run."""

import json
from pathlib import Path
from datetime import datetime, timezone

def classify_robustness(stability_score: float) -> str:
    """Classify robustness based on stability score."""
    if stability_score > 0.8:
        return "Highly Robust"
    elif stability_score > 0.6:
        return "Robust"
    elif stability_score > 0.4:
        return "Moderately Robust"
    elif stability_score > 0.2:
        return "Marginally Robust"
    else:
        return "Not Robust"

def generate_enhanced_report(run_dir: Path):
    """Generate enhanced jackknife report."""
    
    # Load jackknife results
    level1_data = {}
    level2_data = {}
    combined_data = {}
    model_summaries = {}
    
    try:
        with open(run_dir / 'jackknife_level1_results.json') as f:
            level1_data = json.load(f)
        with open(run_dir / 'jackknife_level2_results.json') as f:
            level2_data = json.load(f)
        with open(run_dir / 'jackknife_combined_results.json') as f:
            combined_data = json.load(f)
        print('✅ Loaded jackknife results')
    except Exception as e:
        print(f'❌ Could not load jackknife results: {e}')
        return
    
    try:
        with open(run_dir / 'model_summaries.json') as f:
            model_summaries = json.load(f)
        print('✅ Loaded model summaries')
    except Exception as e:
        print(f'❌ Could not load model summaries: {e}')
    
    # Generate comprehensive report
    report_lines = [
        "# Enhanced Jackknife Analysis Report\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Run Directory: {run_dir.name}",
        "",
        "## Executive Summary\n",
        f"- **Level 1 (Data) Jackknife**: {level1_data.get('n_draws', 0)} draws completed",
        f"- **Level 2 (Fit) Jackknife**: {level2_data.get('n_draws', 0)} draws completed", 
        f"- **Total Jackknife Samples**: {level1_data.get('n_draws', 0) + level2_data.get('n_draws', 0)}",
        ""
    ]
    
    # Add baseline model comparison
    if model_summaries:
        report_lines.extend([
            "## Baseline Model Results\n",
            ""
        ])
        
        for model_name, model_data in model_summaries.items():
            best_fit = model_data.get("best", {})
            params = best_fit.get("parameters", {})
            chi2 = best_fit.get("chi_squared", 0)
            aic = best_fit.get("aic", 0)
            bic = best_fit.get("bic", 0)
            
            report_lines.extend([
                f"### {model_name.upper()} Model\n",
                f"- **χ²**: {chi2:.3f}",
                f"- **AIC**: {aic:.3f}",
                f"- **BIC**: {bic:.3f}",
                f"- **Best-fit Parameters**:",
            ])
            
            for param_name, param_value in params.items():
                report_lines.append(f"  - {param_name}: {param_value:.6f}")
            
            report_lines.extend(["", ""])
    
    # Level 1 detailed results
    if level1_data:
        report_lines.extend([
            "## Level 1 (Data) Jackknife: Detailed Results\n",
            f"**Total Draws**: {level1_data.get('n_draws', 0)}",
            f"**Strategies Analyzed**: {len(level1_data.get('chi2_changes', {}))}",
            "",
            "### Parameter Stability Analysis\n",
            ""
        ])
        
        for param_name, stability_data in level1_data.get('parameter_shifts', {}).items():
            mean_shift = stability_data.get("mean", 0)
            std_shift = stability_data.get("std", 0)
            min_shift = stability_data.get("min", 0)
            max_shift = stability_data.get("max", 0)
            
            report_lines.extend([
                f"#### {param_name}\n",
                f"- **Mean Shift**: {mean_shift:.6f}",
                f"- **Std Dev**: {std_shift:.6f}",
                f"- **Range**: [{min_shift:.6f}, {max_shift:.6f}]",
                f"- **Max Absolute Shift**: {max(abs(min_shift), abs(max_shift)):.6f}",
                ""
            ])
        
        # χ² changes by strategy
        report_lines.extend([
            "### χ² Changes by Removal Strategy\n",
            "| Strategy | χ² Change | % Change | Status |",
            "|----------|-----------|----------|--------|"
        ])
        
        baseline_chi2 = model_summaries.get("pbuf", {}).get("best", {}).get("chi_squared", 0)
        for strategy, chi2_change in level1_data.get('chi2_changes', {}).items():
            # Handle both direct values and dict structures
            if isinstance(chi2_change, dict):
                chi2_val = chi2_change.get('value', chi2_change.get('chi2_change', 0))
            else:
                chi2_val = chi2_change
            
            pct_change = (chi2_val / baseline_chi2 * 100) if baseline_chi2 > 0 else 0
            status = "Stable" if abs(pct_change) < 5 else "Moderate" if abs(pct_change) < 15 else "Significant"
            report_lines.append(f"| {strategy} | {chi2_val:.3f} | {pct_change:.2f}% | {status} |")
        
        report_lines.extend([
            "",
            "### Stability Metrics\n",
            f"- **Mean χ²**: {level1_data.get('stability_metrics', {}).get('mean_chi2', 0):.3f}",
            f"- **χ² Std Dev**: {level1_data.get('stability_metrics', {}).get('chi2_std', 0):.3f}",
            f"- **Stability Score**: {level1_data.get('stability_metrics', {}).get('stability_score', 0):.3f}",
            ""
        ])
    
    # Level 2 detailed results  
    if level2_data:
        report_lines.extend([
            "## Level 2 (Fit) Jackknife: Detailed Results\n",
            f"**Total Draws**: {level2_data.get('n_draws', 0)}",
            f"**Optimization Strategies**: {len(level2_data.get('chi2_changes', {}))}",
            "",
            "### Parameter Stability Analysis\n",
            ""
        ])
        
        for param_name, stability_data in level2_data.get('parameter_shifts', {}).items():
            mean_shift = stability_data.get("mean", 0)
            std_shift = stability_data.get("std", 0)
            min_shift = stability_data.get("min", 0)
            max_shift = stability_data.get("max", 0)
            
            report_lines.extend([
                f"#### {param_name}\n",
                f"- **Mean Shift**: {mean_shift:.6f}",
                f"- **Std Dev**: {std_shift:.6f}",
                f"- **Range**: [{min_shift:.6f}, {max_shift:.6f}]",
                f"- **Max Absolute Shift**: {max(abs(min_shift), abs(max_shift)):.6f}",
                ""
            ])
        
        # χ² changes by optimization strategy
        report_lines.extend([
            "### χ² Changes by Optimization Strategy\n",
            "| Strategy | χ² Change | % Change | Status |",
            "|----------|-----------|----------|--------|"
        ])
        
        baseline_chi2 = model_summaries.get("pbuf", {}).get("best", {}).get("chi_squared", 0)
        for strategy, chi2_change in level2_data.get('chi2_changes', {}).items():
            # Handle both direct values and dict structures
            if isinstance(chi2_change, dict):
                chi2_val = chi2_change.get('value', chi2_change.get('chi2_change', 0))
            else:
                chi2_val = chi2_change
            
            pct_change = (chi2_val / baseline_chi2 * 100) if baseline_chi2 > 0 else 0
            status = "Stable" if abs(pct_change) < 5 else "Moderate" if abs(pct_change) < 15 else "Significant"
            report_lines.append(f"| {strategy} | {chi2_val:.3f} | {pct_change:.2f}% | {status} |")
        
        report_lines.extend([
            "",
            "### Stability Metrics\n",
            f"- **Mean χ²**: {level2_data.get('stability_metrics', {}).get('mean_chi2', 0):.3f}",
            f"- **χ² Std Dev**: {level2_data.get('stability_metrics', {}).get('chi2_std', 0):.3f}",
            f"- **Stability Score**: {level2_data.get('stability_metrics', {}).get('stability_score', 0):.3f}",
            ""
        ])
    
    # Combined analysis
    if combined_data:
        report_lines.extend([
            "## Combined Jackknife Analysis\n",
            "",
            "### Overall Stability Assessment\n",
            f"- **Overall Stability Score**: {combined_data.get('stability_metrics', {}).get('overall_stability_score', 0):.3f}",
            f"- **Data Stability Score**: {combined_data.get('stability_metrics', {}).get('data_stability_score', 0):.3f}",
            f"- **Fit Stability Score**: {combined_data.get('stability_metrics', {}).get('fit_stability_score', 0):.3f}",
            "",
            "### Statistical Robustness\n",
            f"- **Total Validation Points**: {level1_data.get('n_draws', 0) + level2_data.get('n_draws', 0)}",
            f"- **Consistency Rate**: {combined_data.get('stability_metrics', {}).get('consistency_rate', 0):.2%}",
            f"- **Robustness Classification**: {classify_robustness(combined_data.get('stability_metrics', {}).get('overall_stability_score', 0))}",
            ""
        ])
    
    # Add file references
    report_lines.extend([
        "## Data Files\n",
        "",
        "### Main Results\n",
        f"- **Model Summaries**: `model_summaries.json` - Complete model fit results",
        f"- **Parameter Tables**: `tables/best_fit_parameters.csv` - Formatted parameter tables",
        f"- **Model Comparison**: `tables/model_comparison.csv` - Statistical comparison",
        "",
        "### Jackknife Results\n",
        f"- **Level 1 Results**: `jackknife_level1_results.json` - Detailed data jackknife results",
        f"- **Level 2 Results**: `jackknife_level2_results.json` - Detailed fit jackknife results", 
        f"- **Combined Results**: `jackknife_combined_results.json` - Combined analysis",
        "",
        "### Visualizations\n",
        f"- **Figures Directory**: `figures/` - Plots and diagrams (if available)",
        f"- **Enhanced Reports**: `enhanced_reports/` - HTML and additional reports",
        "",
        "## Conclusions\n",
        ""
    ])
    
    # Add conclusions based on stability
    if combined_data:
        stability_score = combined_data.get('stability_metrics', {}).get('overall_stability_score', 0)
        if stability_score > 0.8:
            conclusion = "Excellent stability - results are highly robust"
        elif stability_score > 0.6:
            conclusion = "Good stability - results are reasonably robust"
        elif stability_score > 0.4:
            conclusion = "Moderate stability - results show some sensitivity"
        else:
            conclusion = "Low stability - results require careful interpretation"
        
        report_lines.extend([
            f"**Stability Assessment**: {conclusion}",
            f"**Statistical Confidence**: {'High' if stability_score > 0.7 else 'Medium' if stability_score > 0.5 else 'Low'}",
            f"**Recommendation**: {'Results ready for publication' if stability_score > 0.6 else 'Additional validation recommended'}",
            ""
        ])
    
    report_lines.extend([
        "---\n",
        f"*Report generated by Cosmos2 Enhanced Jackknife System on {datetime.now(timezone.utc).isoformat()}*"
    ])
    
    # Write report
    report_path = run_dir / "jackknife_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    
    print(f"✅ Enhanced report generated: {report_path}")
    print(f"📄 Report length: {len(report_lines)} lines")

if __name__ == "__main__":
    run_dir = Path("data/science_runs/full_joint/2025-11-29T001013_enhanced_jackknife_config")
    generate_enhanced_report(run_dir)
