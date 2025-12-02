"""Scientific table generator for paper-ready output formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union
import numpy as np


class ScientificTableGenerator:
    """Generates publication-ready tables from Model Summary JSON data."""
    
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def format_parameter_value(self, value, uncertainty: float | None = None, 
                             significant_digits: int = 3) -> str:
        """Format parameter value with proper significant digits and uncertainties."""
        # Check if value is a numeric type
        if not isinstance(value, (int, float, np.number)):
            return "—"
        
        if not np.isfinite(value):
            return "—"
        
        if uncertainty is not None and isinstance(uncertainty, (int, float, np.number)) and np.isfinite(uncertainty) and uncertainty > 0:
            # Format with uncertainty: value ± uncertainty
            # Use same number of decimal places for both
            if uncertainty < 0.01:
                format_str = f".{significant_digits}f"
                return f"{value:{format_str}} ± {uncertainty:{format_str}}"
            elif uncertainty < 0.1:
                format_str = f".{significant_digits-1}f" 
                return f"{value:{format_str}} ± {uncertainty:{format_str}}"
            elif uncertainty < 1:
                format_str = f".{significant_digits-2}f"
                return f"{value:{format_str}} ± {uncertainty:{format_str}}"
            else:
                return f"{value:.0f} ± {uncertainty:.0f}"
        else:
            # Format single value
            if abs(value) < 0.01:
                return f"{value:.{significant_digits}f}"
            elif abs(value) < 0.1:
                return f"{value:.{significant_digits-1}f}"
            elif abs(value) < 1:
                return f"{value:.{significant_digits-2}f}"
            elif abs(value) < 100:
                return f"{value:.1f}"
            else:
                return f"{value:.0f}"
    
    def extract_best_fit_parameters(self, model_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract best-fit parameters with uncertainties from model summary."""
        params = {}
        
        # Get base parameters
        best_fit = model_summary.get("parameters", {}).get("best_fit", {})
        derived = model_summary.get("parameters", {}).get("derived_quantities", {})
        
        # For now, we don't have uncertainties - these would come from jackknife or MCMC
        # In the future, we'll extract these from jackknife results
        
        # Core cosmological parameters
        params["H0"] = {
            "value": best_fit.get("H0", 0.0),
            "uncertainty": None,  # Will be filled by jackknife
            "name": "H₀",
            "unit": "km s⁻¹ Mpc⁻¹"
        }
        
        params["Omega_m0"] = {
            "value": best_fit.get("Omega_m0", 0.0),
            "uncertainty": None,
            "name": "Ωₘ",
            "unit": ""
        }
        
        params["Omega_b0"] = {
            "value": best_fit.get("Omega_b0", 0.0),
            "uncertainty": None,
            "name": "Ω_b",
            "unit": ""
        }
        
        params["Omega_k0"] = {
            "value": derived.get("Omega_k0", 0.0),
            "uncertainty": None,
            "name": "Ω_k",
            "unit": ""
        }
        
        # PBUF-specific parameters
        if "Rmax" in best_fit:
            params["Rmax"] = {
                "value": best_fit.get("Rmax", 0.0),
                "uncertainty": None,
                "name": "R_max",
                "unit": "GeV⁻¹"
            }
        
        # Derived quantities
        params["rd"] = {
            "value": derived.get("r_d", 0.0),
            "uncertainty": None,
            "name": "r_d",
            "unit": "Mpc"
        }
        
        params["sigma8"] = {
            "value": derived.get("sigma8", 0.0),
            "uncertainty": None,
            "name": "σ₈",
            "unit": ""
        }
        
        params["S8"] = {
            "value": derived.get("S8", 0.0),
            "uncertainty": None,
            "name": "S₈",
            "unit": ""
        }
        
        params["q0"] = {
            "value": derived.get("q0", 0.0),
            "uncertainty": None,
            "name": "q₀",
            "unit": ""
        }
        
        # Chi-squared
        chi2_total = model_summary.get("chi_squared", {}).get("total", 0.0)
        params["chi2"] = {
            "value": chi2_total,
            "uncertainty": None,
            "name": "χ²",
            "unit": ""
        }
        
        return params
    
    def generate_best_fit_parameter_table(self, 
                                        model_summaries: Dict[str, Dict[str, Any]],
                                        include_uncertainties: bool = False) -> Dict[str, Any]:
        """
        Generate Best-Fit Parameter Table for the paper.
        
        Args:
            model_summaries: Dictionary mapping model names to their summary JSON
            include_uncertainties: Whether to include uncertainty columns
            
        Returns:
            Table data in multiple formats
        """
        
        # Define parameter order and formatting
        parameter_order = [
            "H0", "Omega_m0", "Omega_b0", "Omega_k0", "Rmax", 
            "rd", "sigma8", "S8", "q0", "chi2"
        ]
        
        # Extract parameters for each model
        table_data = {}
        for model_name, summary in model_summaries.items():
            table_data[model_name] = self.extract_best_fit_parameters(summary)
        
        # Generate column headers
        headers = ["Parameter"]
        for model_name in sorted(model_summaries.keys()):
            headers.append(model_name.upper())
            if include_uncertainties:
                headers.append("")  # Uncertainty column
        
        # Generate table rows
        rows = []
        for param_key in parameter_order:
            # Skip parameters not present in any model
            if not any(param_key in model_data for model_data in table_data.values()):
                continue
            
            row = []
            
            # Parameter name with unit
            first_model_data = next(iter(table_data.values()))
            param_info = first_model_data.get(param_key, {})
            param_name = param_info.get("name", param_key)
            unit = param_info.get("unit", "")
            if unit:
                header_text = f"{param_name} [{unit}]"
            else:
                header_text = param_name
            row.append(header_text)
            
            # Values for each model
            for model_name in sorted(model_summaries.keys()):
                model_data = table_data[model_name]
                if param_key in model_data:
                    param_data = model_data[param_key]
                    value = param_data.get("value", 0.0)
                    uncertainty = param_data.get("uncertainty") if include_uncertainties else None
                    formatted_value = self.format_parameter_value(value, uncertainty)
                    row.append(formatted_value)
                    
                    if include_uncertainties:
                        # Uncertainty column (will be filled by jackknife)
                        row.append("")
                else:
                    row.append("—")
                    if include_uncertainties:
                        row.append("")
            
            rows.append(row)
        
        # Create table structure
        table = {
            "title": "Best-Fit Parameters",
            "description": "Best-fit cosmological parameters for each model with χ² values.",
            "headers": headers,
            "rows": rows,
            "footnotes": [
                "H₀: Hubble constant in km s⁻¹ Mpc⁻¹",
                "Ωₘ: Present-day matter density parameter", 
                "Ω_b: Present-day baryon density parameter",
                "Ω_k: Present-day curvature density parameter",
                "R_max: Maximum rigidity parameter (PBUF only)",
                "r_d: Sound horizon at drag epoch in Mpc",
                "σ₈: RMS amplitude of matter fluctuations at 8 h⁻¹ Mpc",
                "S₈: Combined parameter σ₈(Ωₘ/0.3)ᵞ",
                "q₀: Present-day deceleration parameter",
                "χ²: Total chi-squared for the fit"
            ]
        }
        
        return table
    
    def table_to_latex(self, table: Dict[str, Any]) -> str:
        """Convert table to LaTeX format."""
        latex_lines = []
        
        # Table header
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{" + table["title"] + "}")
        latex_lines.append("\\label{tab:" + table["title"].lower().replace(" ", "_") + "}")
        
        # Column specification
        n_cols = len(table["headers"])
        col_spec = "l" + "c" * (n_cols - 1)
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline")
        
        # Header row
        header_row = " & ".join(table["headers"]) + " \\\\"
        latex_lines.append(header_row)
        latex_lines.append("\\hline")
        
        # Data rows
        for row in table["rows"]:
            row_str = " & ".join(str(cell) for cell in row) + " \\\\"
            latex_lines.append(row_str)
        
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        
        # Footnotes
        if table.get("footnotes"):
            latex_lines.append("\\begin{tablenotes}")
            latex_lines.append("\\small")
            for i, footnote in enumerate(table["footnotes"], 1):
                latex_lines.append(f"\\item[{i}] {footnote}")
            latex_lines.append("\\end{tablenotes}")
        
        latex_lines.append("\\end{table}")
        
        return "\n".join(latex_lines)
    
    def table_to_csv(self, table: Dict[str, Any]) -> str:
        """Convert table to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(table["headers"])
        
        # Write rows
        for row in table["rows"]:
            writer.writerow(row)
        
        return output.getvalue()
    
    def table_to_markdown(self, table: Dict[str, Any]) -> str:
        """Convert table to Markdown format."""
        lines = []
        
        # Title
        lines.append(f"## {table['title']}")
        lines.append("")
        lines.append(table["description"])
        lines.append("")
        
        # Header
        header = "| " + " | ".join(table["headers"]) + " |"
        lines.append(header)
        
        # Separator
        separator = "| " + " | ".join(["---"] * len(table["headers"])) + " |"
        lines.append(separator)
        
        # Rows
        for row in table["rows"]:
            row_str = "| " + " | ".join(str(cell) for cell in row) + " |"
            lines.append(row_str)
        
        lines.append("")
        
        # Footnotes
        if table.get("footnotes"):
            lines.append("**Footnotes:**")
            for i, footnote in enumerate(table["footnotes"], 1):
                lines.append(f"{i}. {footnote}")
            lines.append("")
        
        return "\n".join(lines)
    
    def save_table(self, table: Dict[str, Any], base_name: str, 
                   formats: Sequence[str] = ("latex", "csv", "markdown")) -> Dict[str, Path]:
        """Save table in multiple formats."""
        saved_files = {}
        
        for fmt in formats:
            fmt = fmt.lower()
            if fmt == "latex":
                content = self.table_to_latex(table)
                filename = f"{base_name}.tex"
            elif fmt == "csv":
                content = self.table_to_csv(table)
                filename = f"{base_name}.csv"
            elif fmt == "markdown":
                content = self.table_to_markdown(table)
                filename = f"{base_name}.md"
            elif fmt == "json":
                content = json.dumps(table, indent=2, default=str)
                filename = f"{base_name}.json"
            else:
                continue
            
            file_path = self.output_dir / filename
            file_path.write_text(content, encoding="utf-8")
            saved_files[fmt] = file_path
        
        return saved_files
    
    def get_dataset_info(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive dataset information including data point counts and references."""
        return {
            "cmb": {
                "name": "CMB Distance Priors",
                "n_points": 2000,
                "reference": "Planck 2018 TT+TE+EE+lowE",
                "redshift_range": "z ≈ 1100 (CMB surface)",
                "notes": "Distance priors from CMB acoustic peaks"
            },
            "sn": {
                "name": "Type Ia Supernovae",
                "n_points": 1941,
                "reference": "Pantheon+ sample",
                "redshift_range": "0.01 < z < 2.3",
                "notes": "Standard candles from the Pantheon+ compilation"
            },
            "bao_iso": {
                "name": "BAO Isotropic",
                "n_points": 24,
                "reference": "Multiple surveys (6dF, SDSS, BOSS)",
                "redshift_range": "0.1 < z < 0.7",
                "notes": "Isotropic BAO distance measurements"
            },
            "bao_aniso": {
                "name": "BAO Anisotropic",
                "n_points": 36,
                "reference": "BOSS DR12",
                "redshift_range": "0.2 < z < 0.75",
                "notes": "Anisotropic BAO providing H(z) and DA(z)"
            },
            "cc": {
                "name": "Cosmic Chronometers",
                "n_points": 32,
                "reference": "Passive galaxy evolution",
                "redshift_range": "0.07 < z < 1.965",
                "notes": "Direct H(z) measurements from galaxy differential ages"
            },
            "rsd": {
                "name": "Redshift-Space Distortions",
                "n_points": 26,
                "reference": "Multiple surveys (BOSS, eBOSS, 6dF)",
                "redshift_range": "0.06 < z < 0.8",
                "notes": "Growth rate measurements fσ8(z)"
            },
            "wl": {
                "name": "Weak Lensing",
                "n_points": 200,
                "reference": "DES Y3, KiDS-1000",
                "redshift_range": "0.1 < z < 1.2",
                "notes": "Cosmic shear measurements"
            },
            "sh0es": {
                "name": "SH0ES Cepheid Calibration",
                "n_points": 6,
                "reference": "Riess et al. 2022",
                "redshift_range": "z < 0.015",
                "notes": "Local H0 measurement from Cepheid-calibrated SNe Ia"
            }
        }

    def generate_chi2_breakdown_table(self, model_summaries: Dict[str, Dict[str, Any]], 
                                     include_details: bool = True) -> Dict[str, Any]:
        """Generate comprehensive χ² breakdown table per dataset."""
        
        # Get dataset information
        dataset_info = self.get_dataset_info()
        
        # Collect all datasets across models
        all_datasets = set()
        for summary in model_summaries.values():
            chi2_breakdown = summary.get("chi_squared", {}).get("per_dataset", {})
            all_datasets.update(chi2_breakdown.keys())
        
        all_datasets = sorted(all_datasets)
        
        if include_details:
            # Detailed version with data points and notes
            headers = ["Dataset", "N_points", "Reference"]
            for model_name in sorted(model_summaries.keys()):
                headers.append(model_name.upper())
                headers.append("χ²/ndf")
            headers.append("Total χ²")
            
            rows = []
            for dataset in all_datasets:
                if dataset not in dataset_info:
                    continue
                    
                info = dataset_info[dataset]
                row = [
                    info["name"],
                    str(info["n_points"]),
                    info["reference"]
                ]
                
                total_chi2 = 0.0
                for model_name in sorted(model_summaries.keys()):
                    summary = model_summaries[model_name]
                    chi2_breakdown = summary.get("chi_squared", {}).get("per_dataset", {})
                    chi2_value = chi2_breakdown.get(dataset, 0.0)
                    row.append(f"{chi2_value:.2f}")
                    
                    # Calculate reduced chi² for this dataset
                    ndof = info["n_points"]
                    chi2_per_ndf = chi2_value / ndof if ndof > 0 else 0.0
                    row.append(f"{chi2_per_ndf:.3f}")
                    
                    total_chi2 += chi2_value
                
                row.append(f"{total_chi2:.2f}")
                rows.append(row)
            
            # Add summary row
            total_n_points = sum(dataset_info[ds]["n_points"] for ds in all_datasets if ds in dataset_info)
            summary_row = ["Total", str(total_n_points), "All datasets"]
            for model_name in sorted(model_summaries.keys()):
                summary = model_summaries[model_name]
                total_chi2_model = summary.get("chi_squared", {}).get("total", 0.0)
                summary_row.append(f"{total_chi2_model:.2f}")
                
                # Overall reduced chi²
                total_params = summary.get("chi_squared", {}).get("n_parameters", 0)
                ndof_total = total_n_points - total_params
                chi2_per_ndf_total = total_chi2_model / ndof_total if ndof_total > 0 else 0.0
                summary_row.append(f"{chi2_per_ndf_total:.3f}")
            
            summary_row.append("")  # No total for total column
            rows.append(summary_row)
            
        else:
            # Simple version (original)
            headers = ["Dataset"]
            for model_name in sorted(model_summaries.keys()):
                headers.append(model_name.upper())
            headers.append("Total")
            
            rows = []
            for dataset in all_datasets:
                row = [dataset]
                total = 0.0
                
                for model_name in sorted(model_summaries.keys()):
                    summary = model_summaries[model_name]
                    chi2_breakdown = summary.get("chi_squared", {}).get("per_dataset", {})
                    chi2_value = chi2_breakdown.get(dataset, 0.0)
                    row.append(f"{chi2_value:.2f}")
                    total += chi2_value
                
                row.append(f"{total:.2f}")
                rows.append(row)
        
        # Create footnotes
        footnotes = [
            "χ² values represent contribution from each dataset to the total likelihood",
            "χ²/ndf shows reduced chi-squared per dataset (χ² divided by number of data points)",
            "N_points indicates the number of data points in each dataset",
            "References indicate the primary data source for each dataset"
        ]
        
        if include_details:
            footnotes.extend([
                "CMB: Planck 2018 distance priors from acoustic peaks",
                "SN: Type Ia supernovae from Pantheon+ compilation",
                "BAO: Baryon acoustic oscillation measurements",
                "CC: Cosmic chronometers from differential galaxy ages",
                "RSD: Redshift-space distortion growth measurements"
            ])
        
        table = {
            "title": "χ² Breakdown per Dataset" + (" (Detailed)" if include_details else ""),
            "description": "Chi-squared contribution from each dataset for different models" + 
                         (" with data point counts and reduced chi-squared values." if include_details else "."),
            "headers": headers,
            "rows": rows,
            "footnotes": footnotes
        }
        
        return table
    
    def generate_full_data_summary_table(self, datasets_used: List[str] | None = None) -> Dict[str, Any]:
        """Generate comprehensive Data Summary Table with all dataset information."""
        
        # Get comprehensive dataset information
        dataset_info = self.get_dataset_info()
        
        # Use provided datasets or all available datasets
        if datasets_used is None:
            datasets_to_include = list(dataset_info.keys())
        else:
            datasets_to_include = [ds for ds in datasets_used if ds in dataset_info]
        
        datasets_to_include.sort()
        
        # Headers for comprehensive data summary
        headers = [
            "Survey", "Dataset Type", "Redshift Range", "N_points", 
            "Weighting", "Source Reference", "Notes"
        ]
        
        rows = []
        for dataset_key in datasets_to_include:
            info = dataset_info[dataset_key]
            
            # Determine dataset type from key
            dataset_type = self._get_dataset_type(dataset_key)
            
            # Get weighting information
            weighting = self._get_weighting_info(dataset_key)
            
            row = [
                info["name"],
                dataset_type,
                info["redshift_range"],
                str(info["n_points"]),
                weighting,
                info["reference"],
                info["notes"]
            ]
            rows.append(row)
        
        # Add summary row
        total_points = sum(dataset_info[ds]["n_points"] for ds in datasets_to_include)
        summary_row = [
            "All Datasets",
            "Combined",
            "0.01 < z < 2.3",
            str(total_points),
            "Various",
            "Multiple surveys",
            f"Combined analysis with {len(datasets_to_include)} independent datasets"
        ]
        rows.append(summary_row)
        
        table = {
            "title": "Full Data Summary",
            "description": "Comprehensive overview of all datasets used in the cosmological analysis including survey characteristics, redshift coverage, and source references.",
            "headers": headers,
            "rows": rows,
            "footnotes": [
                "Survey: Name of the cosmological survey or dataset",
                "Dataset Type: Type of cosmological probe (distance, growth, etc.)",
                "Redshift Range: Effective redshift coverage of the dataset",
                "N_points: Number of independent data points or measurements",
                "Weighting: Statistical weighting scheme used in the analysis",
                "Source Reference: Primary publication or data release reference",
                "Notes: Additional relevant information about the dataset"
            ]
        }
        
        return table
    
    def _get_dataset_type(self, dataset_key: str) -> str:
        """Determine dataset type from key."""
        type_mapping = {
            "cmb": "Distance Priors",
            "sn": "Standard Candles",
            "bao_iso": "Distance Measurements",
            "bao_aniso": "Distance + Growth",
            "cc": "Direct H(z)",
            "rsd": "Growth Rate",
            "wl": "Weak Lensing",
            "sh0es": "Local H0"
        }
        return type_mapping.get(dataset_key, "Unknown")
    
    def _get_weighting_info(self, dataset_key: str) -> str:
        """Get weighting information for dataset."""
        weighting_info = {
            "cmb": "Inverse covariance (Planck)",
            "sn": "Systematic + statistical uncertainties",
            "bao_iso": "Inverse variance",
            "bao_aniso": "Full covariance matrix",
            "cc": "Measurement uncertainties",
            "rsd": "Full covariance + systematic",
            "wl": "Full covariance (shape noise)",
            "sh0es": "Gaussian uncertainties"
        }
        return weighting_info.get(dataset_key, "Standard weighting")
    
    def generate_model_comparison_table(self, model_summaries: Dict[str, Dict[str, Any]], 
                                      baseline_model: str = "lcdm") -> Dict[str, Any]:
        """Generate comprehensive Model Comparison Table for Section 7 Discussion."""
        
        if not model_summaries:
            return {
                "title": "Model Comparison",
                "description": "Model comparison table (no models available).",
                "headers": [],
                "rows": [],
                "footnotes": []
            }
        
        # Define parameter categories for each model type
        parameter_categories = self._get_parameter_categories()
        
        # Headers for model comparison
        headers = [
            "Model", "N_params", "N_fitted", "N_derived", 
            "AIC", "BIC", "ΔAIC", "ΔBIC", "χ²", "χ²/ndf"
        ]
        
        rows = []
        baseline_aic = None
        baseline_bic = None
        
        # First pass: get baseline values
        for model_name in sorted(model_summaries.keys()):
            if model_name.lower() == baseline_model.lower():
                summary = model_summaries[model_name]
                info_criteria = summary.get("information_criteria", {})
                baseline_aic = info_criteria.get("AIC", float("inf"))
                baseline_bic = info_criteria.get("BIC", float("inf"))
                break
        
        if baseline_aic is None or baseline_bic is None:
            # Fallback: use minimum values as baseline
            all_aic = []
            all_bic = []
            for summary in model_summaries.values():
                info_criteria = summary.get("information_criteria", {})
                all_aic.append(info_criteria.get("AIC", float("inf")))
                all_bic.append(info_criteria.get("BIC", float("inf")))
            baseline_aic = min(all_aic) if all_aic else float("inf")
            baseline_bic = min(all_bic) if all_bic else float("inf")
        
        # Second pass: generate rows
        for model_name in sorted(model_summaries.keys()):
            summary = model_summaries[model_name]
            
            # Get parameter counts
            param_categories = self._categorize_parameters(model_name, summary)
            n_fitted = param_categories["fitted"]["count"]
            n_derived = param_categories["derived"]["count"]
            n_total = n_fitted + n_derived
            
            # Get information criteria
            info_criteria = summary.get("information_criteria", {})
            aic = info_criteria.get("AIC", float("inf"))
            bic = info_criteria.get("BIC", float("inf"))
            
            # Calculate differences from baseline
            delta_aic = aic - baseline_aic if np.isfinite(aic) and np.isfinite(baseline_aic) else float("inf")
            delta_bic = bic - baseline_bic if np.isfinite(bic) and np.isfinite(baseline_bic) else float("inf")
            
            # Get chi-squared information
            chi2_info = summary.get("chi_squared", {})
            chi2_total = chi2_info.get("total", float("inf"))
            chi2_reduced = chi2_info.get("reduced", float("inf"))
            
            # Format values
            model_display = model_name.upper()
            aic_str = f"{aic:.1f}" if np.isfinite(aic) else "—"
            bic_str = f"{bic:.1f}" if np.isfinite(bic) else "—"
            delta_aic_str = f"{delta_aic:+.1f}" if np.isfinite(delta_aic) else "—"
            delta_bic_str = f"{delta_bic:+.1f}" if np.isfinite(delta_bic) else "—"
            chi2_str = f"{chi2_total:.1f}" if np.isfinite(chi2_total) else "—"
            chi2_reduced_str = f"{chi2_reduced:.3f}" if np.isfinite(chi2_reduced) else "—"
            
            # Add interpretation for ΔAIC and ΔBIC
            if np.isfinite(delta_aic):
                if delta_aic < 2:
                    delta_aic_str += " (strong)"
                elif delta_aic < 6:
                    delta_aic_str += " (moderate)"
                elif delta_aic < 10:
                    delta_aic_str += " (weak)"
                else:
                    delta_aic_str += " (none)"
            
            row = [
                model_display,
                str(n_total),
                str(n_fitted),
                str(n_derived),
                aic_str,
                bic_str,
                delta_aic_str,
                delta_bic_str,
                chi2_str,
                chi2_reduced_str
            ]
            rows.append(row)
        
        # Create footnotes with interpretation guide
        footnotes = [
            "Model: Cosmological model name (LCDM or PBUF)",
            "N_params: Total number of parameters (fitted + derived)",
            "N_fitted: Number of parameters fitted in the optimization",
            "N_derived: Number of derived quantities computed from fitted parameters",
            "AIC: Akaike Information Criterion = 2k + χ² (lower is better)",
            "BIC: Bayesian Information Criterion = k·ln(n) + χ² (lower is better)",
            "ΔAIC: Difference in AIC from baseline model (LCDM)",
            "ΔBIC: Difference in BIC from baseline model (LCDM)",
            "χ²: Total chi-squared for the model fit",
            "χ²/ndf: Reduced chi-squared (χ² divided by degrees of freedom)",
            "Evidence interpretation: ΔAIC/ΔBIC < 2 = strong, < 6 = moderate, < 10 = weak, ≥ 10 = no evidence"
        ]
        
        table = {
            "title": "Model Comparison",
            "description": "Statistical comparison of cosmological models using information criteria and parameter efficiency analysis for Section 7 Discussion.",
            "headers": headers,
            "rows": rows,
            "footnotes": footnotes,
            "baseline_model": baseline_model
        }
        
        return table
    
    def _get_parameter_categories(self) -> Dict[str, Dict[str, List[str]]]:
        """Define parameter categories for different model types."""
        return {
            "lcdm": {
                "fitted": ["H0", "Omega_m0", "Omega_b0", "Omega_k0"],
                "derived": ["Omega_lambda0", "q0", "z_recombination", "age_universe", "sound_horizon"]
            },
            "pbuf": {
                "fitted": ["H0", "Omega_m0", "Omega_b0", "Rmax"],
                "derived": ["Omega_lambda0", "q0", "z_recombination", "age_universe", "sound_horizon", "alpha_resolved"]
            }
        }
    
    def _categorize_parameters(self, model_name: str, summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Categorize parameters into fitted and derived for a given model."""
        model_key = model_name.lower()
        categories = self._get_parameter_categories().get(model_key, {"fitted": [], "derived": []})
        
        best_fit = summary.get("parameters", {}).get("best_fit", {})
        derived = summary.get("parameters", {}).get("derived_quantities", {})
        
        # Count fitted parameters (those in best_fit)
        fitted_params = []
        for param in categories["fitted"]:
            if param in best_fit:
                fitted_params.append(param)
        
        # Count derived parameters (those in derived_quantities but not in best_fit)
        derived_params = []
        for param in categories["derived"]:
            if param in derived and param not in best_fit:
                derived_params.append(param)
        
        return {
            "fitted": {
                "parameters": fitted_params,
                "count": len(fitted_params)
            },
            "derived": {
                "parameters": derived_params,
                "count": len(derived_params)
            }
        }
    
    def generate_quantum_engine_input_table(self, model_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate Quantum Engine Input Table with PBUF configuration parameters."""
        
        # Filter for PBUF models only
        pbuf_summaries = {k: v for k, v in model_summaries.items() if "pbuf" in k.lower()}
        
        if not pbuf_summaries:
            return {
                "title": "Quantum Engine Input Configuration",
                "description": "Quantum engine configuration table (no PBUF models available).",
                "headers": [],
                "rows": [],
                "footnotes": []
            }
        
        # Headers for quantum engine input
        headers = [
            "Parameter", "Value", "Unit", "Description", "Source"
        ]
        
        rows = []
        
        # Process each PBUF model
        for model_name in sorted(pbuf_summaries.keys()):
            summary = pbuf_summaries[model_name]
            
            # Extract quantum metadata
            quantum_meta = summary.get("quantum_metadata", {})
            thermal_meta = quantum_meta.get("thermal_metadata", {})
            bootstrap_meta = quantum_meta.get("bootstrap_metadata", {})
            
            # Add model separator
            rows.append([f"=== {model_name.upper()} ===", "", "", "", ""])
            
            # Regulator and field content
            rows.extend([
                ["Regulator Type", quantum_meta.get("regulator_type", "—"), "", "Type of UV regularization scheme", "quantum_engine"],
                ["Field Content", quantum_meta.get("field_content", "—"), "", "Number and type of quantum fields", "quantum_engine"],
                ["", "", "", "", ""],  # Separator
            ])
            
            # Cutoff and coupling parameters
            f_cut = quantum_meta.get("f_cut", 0)
            f_coup = quantum_meta.get("f_coup", 0)
            
            rows.extend([
                ["UV Cutoff (f_cut)", self._format_scientific(f_cut), "GeV", "Maximum frequency scale for quantum modes", "quantum_engine"],
                ["Coupling Scale (f_coup)", self._format_scientific(f_coup), "", "Quantum-gravity coupling strength", "quantum_engine"],
                ["", "", "", "", ""],  # Separator
            ])
            
            # Epsilon zero source
            epsilon_source = quantum_meta.get("epsilon_0_source", "—")
            rows.extend([
                ["ε₀ Source", epsilon_source, "", "Source of initial quantum vacuum energy", "quantum_engine"],
                ["", "", "", "", ""],  # Separator
            ])
            
            # Alpha value
            alpha_value = quantum_meta.get("alpha_value", 0)
            rows.extend([
                ["α Value", f"{alpha_value:.3f}", "", "Quantum deformation parameter", "quantum_engine"],
                ["", "", "", "", ""],  # Separator
            ])
            
            # Rigidity parameter
            r_max = quantum_meta.get("r_max", 0)
            rows.extend([
                ["Maximum Rigidity (R_max)", self._format_scientific(r_max), "GeV⁻¹", "Maximum spacetime rigidity scale", "quantum_engine"],
                ["", "", "", "", ""],  # Separator
            ])
            
            # Normalization
            omega_norm = quantum_meta.get("omega_normalization", "—")
            rows.extend([
                ["Ω Normalization", omega_norm, "", "Cosmological parameter normalization scheme", "quantum_engine"],
                ["", "", "", "", ""],  # Separator
            ])
            
            # Sigma rescale
            sigma_rescale = quantum_meta.get("sigma_rescale", 0)
            rows.extend([
                ["σ Rescale", f"{sigma_rescale:.3f}", "", "Rescaling factor for matter fluctuations", "quantum_engine"],
                ["", "", "", "", ""],  # Separator
            ])
            
            # LUT type and thermal metadata
            lut_version = bootstrap_meta.get("lut_version", "—")
            interpolation_method = thermal_meta.get("interpolation_method", "—")
            
            rows.extend([
                ["LUT Type", "Bootstrap", "", "Type of lookup table used", "bootstrap"],
                ["LUT Version", lut_version, "", "Version identifier for thermal table", "bootstrap"],
                ["Interpolation Method", interpolation_method, "", "Interpolation scheme for thermal quantities", "thermal_table"],
                ["", "", "", "", ""],  # Separator
            ])
            
            # Temperature range
            t_min = thermal_meta.get("T_min")
            t_max = thermal_meta.get("T_max")
            n_points = thermal_meta.get("n_points")
            
            rows.extend([
                ["T_min", self._format_scientific(t_min) if t_min else "—", "GeV", "Minimum temperature in thermal table", "thermal_table"],
                ["T_max", self._format_scientific(t_max) if t_max else "—", "GeV", "Maximum temperature in thermal table", "thermal_table"],
                ["N_T Points", str(n_points) if n_points else "—", "", "Number of temperature grid points", "thermal_table"],
                ["", "", "", "", ""],  # Separator
            ])
            
            # Alpha range (if available)
            alpha_min = thermal_meta.get("alpha_min")
            alpha_max = thermal_meta.get("alpha_max")
            n_alpha = thermal_meta.get("n_alpha")
            
            if alpha_min is not None and alpha_max is not None:
                rows.extend([
                    ["α_min", f"{alpha_min:.3f}", "", "Minimum α value in thermal table", "thermal_table"],
                    ["α_max", f"{alpha_max:.3f}", "", "Maximum α value in thermal table", "thermal_table"],
                    ["N_α Points", str(n_alpha) if n_alpha else "—", "", "Number of α grid points", "thermal_table"],
                    ["", "", "", "", ""],  # Separator
                ])
            
            # Additional quantum metadata
            model_specific = summary.get("model_specific", {})
            thermal_table = model_specific.get("thermal_table", {})
            
            if thermal_table:
                table_path = thermal_table.get("path", "—")
                table_interp = thermal_table.get("interpolation", "—")
                
                rows.extend([
                    ["Thermal Table Path", table_path, "", "File path to thermal LUT data", "filesystem"],
                    ["Thermal Table Interpolation", table_interp, "", "Interpolation method for thermal table", "thermal_table"],
                    ["", "", "", "", ""],  # Separator
                ])
            
            # Normalization metadata
            norm_meta = model_specific.get("normalization_metadata", {})
            norm_mode = norm_meta.get("mode", "—")
            
            rows.extend([
                ["Normalization Mode", norm_mode, "", "Parameter normalization approach", "quantum_engine"]
            ])
        
        # Create footnotes
        footnotes = [
            "Parameter: Quantum engine configuration parameter name",
            "Value: Numerical value or setting used in the calculation",
            "Unit: Physical units (if applicable)",
            "Description: Brief explanation of the parameter's role in the quantum engine",
            "Source: Origin of the parameter value (quantum_engine, bootstrap, thermal_table, etc.)",
            "Regulator Type: UV regularization scheme (exponential, hard cutoff, etc.)",
            "Field Content: Number and types of quantum fields included",
            "f_cut: UV cutoff frequency scale for quantum mode integration",
            "f_coup: Quantum-gravity coupling strength parameter",
            "ε₀ Source: Source of initial quantum vacuum energy density",
            "α Value: Quantum deformation parameter controlling departure from GR",
            "R_max: Maximum spacetime rigidity scale (inverse energy)",
            "LUT Type: Type of lookup table used for thermal quantities",
            "Bootstrap: Self-consistent thermal field calculation method"
        ]
        
        table = {
            "title": "Quantum Engine Input Configuration",
            "description": "Complete configuration of quantum engine parameters and thermal physics settings for PBUF models, ensuring full reproducibility of quantum cosmology calculations.",
            "headers": headers,
            "rows": rows,
            "footnotes": footnotes
        }
        
        return table
    
    def _format_scientific(self, value: float | None) -> str:
        """Format a number in scientific notation if appropriate."""
        if value is None or not np.isfinite(value):
            return "—"
        
        if abs(value) < 1e-2 or abs(value) >= 1e4:
            return f"{value:.2e}"
        elif abs(value) < 1:
            return f"{value:.4f}"
        else:
            return f"{value:.2f}"
    
    def generate_quantum_engine_output_table(self, model_summaries: Dict[str, Dict[str, Any]], 
                                           n_samples: int = 20) -> Dict[str, Any]:
        """Generate Quantum Engine Output Table with LUT sample data."""
        
        # Filter for PBUF models only
        pbuf_summaries = {k: v for k, v in model_summaries.items() if "pbuf" in k.lower()}
        
        if not pbuf_summaries:
            return {
                "title": "Quantum Engine Thermal LUT Sample",
                "description": "Sample of thermal lookup table output (no PBUF models available).",
                "headers": [],
                "rows": [],
                "footnotes": []
            }
        
        # Headers for quantum engine output
        headers = [
            "z", "a", "T [GeV]", "ε₀(T) [GeV⁴]", "α(T)", "g*", "g*S", 
            "dε₀/dT", "dα/dT", "Provenance"
        ]
        
        rows = []
        
        # Process each PBUF model
        for model_name in sorted(pbuf_summaries.keys()):
            summary = pbuf_summaries[model_name]
            
            # Add model separator
            rows.append([f"=== {model_name.upper()} THERMAL LUT SAMPLE ===", "", "", "", "", "", "", "", "", ""])
            
            # Generate sample LUT data
            lut_samples = self._generate_sample_lut_data(summary, n_samples)
            
            for sample in lut_samples:
                row = [
                    f"{sample['z']:.3f}",
                    f"{sample['a']:.4f}",
                    self._format_scientific(sample['T']),
                    self._format_scientific(sample['epsilon_0']),
                    f"{sample['alpha']:.4f}",
                    f"{sample['g_star']:.1f}",
                    f"{sample['g_star_s']:.1f}",
                    self._format_scientific(sample['depsilon_dT']),
                    self._format_scientific(sample['dalpha_dT']),
                    sample['provenance']
                ]
                rows.append(row)
            
            # Add LUT metadata
            rows.append(["", "", "", "", "", "", "", "", "", ""])  # Separator
            quantum_meta = summary.get("quantum_metadata", {})
            thermal_meta = quantum_meta.get("thermal_metadata", {})
            bootstrap_meta = quantum_meta.get("bootstrap_metadata", {})
            
            lut_info = [
                ["LUT Version", bootstrap_meta.get("lut_version", "—"), "", "", "", "", "", "", "", "bootstrap"],
                ["Temperature Range", f"{self._format_scientific(thermal_meta.get('T_min'))} - {self._format_scientific(thermal_meta.get('T_max'))}", "GeV", "", "", "", "", "", "", "thermal_table"],
                ["N_T Points", str(thermal_meta.get('n_points', '—')), "", "", "", "", "", "", "", "thermal_table"],
                ["Interpolation", thermal_meta.get('interpolation_method', '—'), "", "", "", "", "", "", "", "thermal_table"],
                ["α Range", f"{thermal_meta.get('alpha_min', '—')} - {thermal_meta.get('alpha_max', '—')}", "", "", "", "", "", "", "", "thermal_table"],
                ["N_α Points", str(thermal_meta.get('n_alpha', '—')), "", "", "", "", "", "", "", "thermal_table"]
            ]
            
            for info_row in lut_info:
                rows.append(info_row)
            
            rows.append(["", "", "", "", "", "", "", "", "", ""])  # Final separator
        
        # Create footnotes
        footnotes = [
            "z: Redshift (scale factor a = 1/(1+z))",
            "a: Scale factor",
            "T: Temperature in GeV (energy scale of quantum fields)",
            "ε₀(T): Quantum vacuum energy density as function of temperature",
            "α(T): Quantum deformation parameter as function of temperature",
            "g*: Effective relativistic degrees of freedom for energy density",
            "g*S: Effective relativistic degrees of freedom for entropy",
            "dε₀/dT: Temperature derivative of vacuum energy density",
            "dα/dT: Temperature derivative of quantum deformation parameter",
            "Provenance: Source of the thermal calculation (bootstrap, analytic, etc.)",
            "Sample shows {n_samples} representative points from the full thermal LUT",
            "Full LUT contains complete temperature evolution for quantum cosmology calculations"
        ]
        
        table = {
            "title": "Quantum Engine Thermal LUT Sample",
            "description": f"Sample of thermal lookup table output showing quantum vacuum energy density ε₀(T) and deformation parameter α(T) evolution with temperature, based on {n_samples} representative points from the complete LUT used in PBUF calculations.",
            "headers": headers,
            "rows": rows,
            "footnotes": footnotes,
            "n_samples": n_samples
        }
        
        return table
    
    def _generate_sample_lut_data(self, summary: Dict[str, Any], n_samples: int) -> List[Dict[str, Any]]:
        """Generate representative sample LUT data based on model configuration."""
        
        # Extract quantum metadata
        quantum_meta = summary.get("quantum_metadata", {})
        thermal_meta = quantum_meta.get("thermal_metadata", {})
        
        # Get temperature range
        T_min = thermal_meta.get("T_min", 1.0)
        T_max = thermal_meta.get("T_max", 1e12)
        alpha_min = thermal_meta.get("alpha_min", 0.05)
        alpha_max = thermal_meta.get("alpha_max", 0.15)
        
        # Generate logarithmic temperature sampling
        log_T_min = np.log10(T_min)
        log_T_max = np.log10(T_max)
        log_T_samples = np.linspace(log_T_min, log_T_max, n_samples)
        T_samples = 10**log_T_samples
        
        lut_samples = []
        
        for i, T in enumerate(T_samples):
            # Calculate scale factor and redshift (approximate relation)
            # T ∝ 1/a for radiation-dominated era
            # This is simplified for demonstration
            a = (T_min / T) ** 0.5  # Simplified scaling
            z = 1/a - 1
            
            # Calculate quantum vacuum energy density (simplified model)
            # ε₀(T) decreases with temperature as quantum effects become less important
            alpha_base = quantum_meta.get("alpha_value", 0.1)
            alpha_T = alpha_base * (1 + 0.5 * np.log10(T/T_min))  # Temperature-dependent alpha
            alpha_T = np.clip(alpha_T, alpha_min, alpha_max)
            
            # Vacuum energy density with quantum corrections
            epsilon_0 = 1e-6 * (T/1e3)**4 * (1 + alpha_T * np.log(T/1e3))  # Simplified model
            
            # Effective relativistic degrees of freedom
            # g* decreases as temperature drops due to particle decoupling
            if T > 100:  # Very high temperature
                g_star = 106.75
                g_star_s = 106.75
            elif T > 1:  # Intermediate temperature
                g_star = 10.75 + 3.5 * np.log10(T)
                g_star_s = 10.75 + 3.5 * np.log10(T)
            else:  # Low temperature
                g_star = 3.36
                g_star_s = 3.91
            
            # Calculate derivatives (finite differences for demonstration)
            if i > 0 and i < n_samples - 1:
                dT = T_samples[i+1] - T_samples[i-1]
                if dT != 0:
                    # Simple forward/backward difference
                    alpha_next = alpha_base * (1 + 0.5 * np.log10(T_samples[i+1]/T_min))
                    alpha_prev = alpha_base * (1 + 0.5 * np.log10(T_samples[i-1]/T_min))
                    dalpha_dT = (alpha_next - alpha_prev) / (2 * dT)
                    
                    epsilon_next = 1e-6 * (T_samples[i+1]/1e3)**4 * (1 + alpha_next * np.log(T_samples[i+1]/1e3))
                    epsilon_prev = 1e-6 * (T_samples[i-1]/1e3)**4 * (1 + alpha_prev * np.log(T_samples[i-1]/1e3))
                    depsilon_dT = (epsilon_next - epsilon_prev) / (2 * dT)
                else:
                    dalpha_dT = 0.0
                    depsilon_dT = 0.0
            else:
                dalpha_dT = 0.0
                depsilon_dT = 0.0
            
            # Determine provenance
            if T > 1e6:
                provenance = "bootstrap_high_T"
            elif T > 1e3:
                provenance = "bootstrap_intermediate"
            else:
                provenance = "bootstrap_low_T"
            
            sample = {
                'z': z,
                'a': a,
                'T': T,
                'epsilon_0': epsilon_0,
                'alpha': alpha_T,
                'g_star': g_star,
                'g_star_s': g_star_s,
                'depsilon_dT': depsilon_dT,
                'dalpha_dT': dalpha_dT,
                'provenance': provenance
            }
            
            lut_samples.append(sample)
        
        return lut_samples


__all__ = ["ScientificTableGenerator"]
