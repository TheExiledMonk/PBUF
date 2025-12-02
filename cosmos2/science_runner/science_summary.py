"""Science summary generator for cosmos2 science runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .naming_conventions import NamingConventions


class ScienceSummaryGenerator:
    """Generates comprehensive science summary text chunks."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        
    def _get_first_summary(self, summaries: List[Dict[str, Any]] | Dict[str, Any]) -> Dict[str, Any]:
        """Safely get the first summary, handling both list and dict formats."""
        if isinstance(summaries, list):
            return summaries[0] if summaries else {}
        return summaries if summaries else {}
        
    def generate_science_summary(self, 
                               model_summaries: Dict[str, List[Dict[str, Any]]],
                               jackknife_summary: Dict[str, Any] | None = None,
                               quantum_state: Dict[str, Any] | None = None) -> Path:
        """Generate comprehensive science summary text chunk."""
        output_path = self.run_dir / "science_summary.txt"
        
        summary_lines = []
        summary_lines.append("COSMOLOGICAL ANALYSIS SCIENCE SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        
        # 1. Executive Summary
        summary_lines.extend(self._generate_executive_summary(model_summaries))
        summary_lines.append("")
        
        # 2. Parameter Results
        summary_lines.extend(self._generate_parameter_results(model_summaries, jackknife_summary))
        summary_lines.append("")
        
        # 3. Dataset Statistics
        summary_lines.extend(self._generate_dataset_statistics(model_summaries))
        summary_lines.append("")
        
        # 4. Chi² Statistics
        summary_lines.extend(self._generate_chi2_statistics(model_summaries, jackknife_summary))
        summary_lines.append("")
        
        # 5. Derived Quantities
        summary_lines.extend(self._generate_derived_quantities(model_summaries, jackknife_summary))
        summary_lines.append("")
        
        # 6. r_d Comparison
        summary_lines.extend(self._generate_rd_comparison(model_summaries))
        summary_lines.append("")
        
        # 7. H₀ and S₈ Results
        summary_lines.extend(self._generate_h0_s8_results(model_summaries, jackknife_summary))
        summary_lines.append("")
        
        # 8. Predictions Summary
        summary_lines.extend(self._generate_predictions_summary(model_summaries))
        summary_lines.append("")
        
        # 9. Jackknife Stability Highlights
        if jackknife_summary:
            summary_lines.extend(self._generate_jackknife_highlights(jackknife_summary))
            summary_lines.append("")
        
        # 10. Quantum Results (if available)
        if quantum_state:
            summary_lines.extend(self._generate_quantum_results(quantum_state))
            summary_lines.append("")
        
        # 11. Model Comparison
        if len(model_summaries) > 1:
            summary_lines.extend(self._generate_model_comparison(model_summaries))
            summary_lines.append("")
        
        # 12. Quality Assessment
        summary_lines.extend(self._generate_quality_assessment(model_summaries, jackknife_summary))
        summary_lines.append("")
        
        # 13. Recommendations
        summary_lines.extend(self._generate_recommendations(model_summaries, jackknife_summary))
        summary_lines.append("")
        
        # 14. Metadata
        summary_lines.extend(self._generate_metadata_summary())
        
        output_path.write_text("\n".join(summary_lines))
        return output_path
    
    def _generate_executive_summary(self, model_summaries: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate executive summary."""
        lines = []
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 20)
        lines.append("")
        
        total_models = len(model_summaries)
        lines.append(f"Analysis completed for {total_models} cosmological model(s):")
        
        for model_name, summaries in model_summaries.items():
            if summaries and len(summaries) > 0:
                model_display = NamingConventions.format_model_name(model_name)
                best_summary = self._get_first_summary(summaries)
                chi2 = best_summary.get("best_chi2", 0.0)
                dof = best_summary.get("total_dof", 0)
                chi2_per_dof = chi2 / dof if dof > 0 else 0.0
                
                lines.append(f"  • {model_display}: χ²/dof = {chi2_per_dof:.2f}")
        
        lines.append("")
        lines.append("Key findings:")
        
        # Find best model by chi2/dof
        best_model = None
        best_chi2_dof = float('inf')
        
        for model_name, summaries in model_summaries.items():
            if summaries:
                best_summary = self._get_first_summary(summaries)
                chi2 = best_summary.get("best_chi2", 0.0)
                dof = best_summary.get("total_dof", 0)
                chi2_per_dof = chi2 / dof if dof > 0 else float('inf')
                
                if chi2_per_dof < best_chi2_dof:
                    best_chi2_dof = chi2_per_dof
                    best_model = model_name
        
        if best_model:
            best_display = NamingConventions.format_model_name(best_model)
            lines.append(f"  • Best fit: {best_display} (χ²/dof = {best_chi2_dof:.2f})")
        
        # Check for H0 values
        h0_values = []
        for model_name, summaries in model_summaries.items():
            if summaries:
                params = self._get_first_summary(summaries).get("best_params", {})
                h0 = params.get("H0")
                if h0 is not None:
                    h0_values.append((model_name, h0))
        
        if h0_values:
            lines.append(f"  • Hubble constant values:")
            for model_name, h0 in h0_values:
                model_display = NamingConventions.format_model_name(model_name)
                lines.append(f"    - {model_display}: H₀ = {h0:.1f} km s⁻¹ Mpc⁻¹")
        
        lines.append("")
        return lines
    
    def _generate_parameter_results(self, 
                                   model_summaries: Dict[str, List[Dict[str, Any]]],
                                   jackknife_summary: Dict[str, Any] | None = None) -> List[str]:
        """Generate parameter results section."""
        lines = []
        lines.append("PARAMETER RESULTS")
        lines.append("-" * 20)
        lines.append("")
        
        # Collect all parameters
        all_params = set()
        for summaries in model_summaries.values():
            if summaries:
                all_params.update(self._get_first_summary(summaries).get("best_params", {}).keys())
        
        # Sort parameters in standard order
        param_order = ["H0", "Omega_m0", "Omega_b0", "Omega_k0", "Rmax", "alpha", "w", "wa"]
        sorted_params = [p for p in param_order if p in all_params] + sorted([p for p in all_params if p not in param_order])
        
        for param_name in sorted_params:
            param_display = NamingConventions.format_parameter_name(param_name)
            units = NamingConventions.get_parameter_unit(param_name)
            unit_str = f" {units}" if units else ""
            
            lines.append(f"{param_display}{unit_str}:")
            
            for model_name, summaries in model_summaries.items():
                if summaries:
                    params = self._get_first_summary(summaries).get("best_params", {})
                    value = params.get(param_name)
                    
                    if value is not None:
                        model_display = NamingConventions.format_model_name(model_name)
                        
                        # Add uncertainty if available
                        uncertainty = None
                        if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                            jackknife_params = jackknife_summary["models"][model_name].get("parameters", {})
                            if param_name in jackknife_params:
                                uncertainty = jackknife_params[param_name].get("std")
                        
                        if uncertainty is not None:
                            lines.append(f"  • {model_display}: {value:.3f} ± {uncertainty:.3f}")
                        else:
                            lines.append(f"  • {model_display}: {value:.3f}")
            
            lines.append("")
        
        return lines
    
    def _generate_dataset_statistics(self, model_summaries: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate dataset statistics section."""
        lines = []
        lines.append("DATASET STATISTICS")
        lines.append("-" * 20)
        lines.append("")
        
        # Count datasets across all models
        all_datasets = set()
        for summaries in model_summaries.values():
            if summaries:
                all_datasets.update(self._get_first_summary(summaries).get("chi2_breakdown", {}).keys())
        
        lines.append(f"Total datasets used: {len(all_datasets)}")
        lines.append("")
        
        for dataset_name in sorted(all_datasets):
            dataset_display = NamingConventions.format_dataset_name(dataset_name)
            lines.append(f"{dataset_display}:")
            
            for model_name, summaries in model_summaries.items():
                if summaries:
                    chi2_breakdown = self._get_first_summary(summaries).get("chi2_breakdown", {})
                    if dataset_name in chi2_breakdown:
                        chi2_info = chi2_breakdown[dataset_name]
                        chi2 = chi2_info.get("chi2", 0.0)
                        dof = chi2_info.get("dof", 0)
                        chi2_per_dof = chi2 / dof if dof > 0 else 0.0
                        
                        model_display = NamingConventions.format_model_name(model_name)
                        lines.append(f"  • {model_display}: χ²/dof = {chi2_per_dof:.2f} (χ² = {chi2:.1f}, dof = {dof})")
            
            lines.append("")
        
        return lines
    
    def _generate_chi2_statistics(self, 
                                model_summaries: Dict[str, List[Dict[str, Any]]],
                                jackknife_summary: Dict[str, Any] | None = None) -> List[str]:
        """Generate chi² statistics section."""
        lines = []
        lines.append("CHI² STATISTICS")
        lines.append("-" * 20)
        lines.append("")
        
        for model_name, summaries in model_summaries.items():
            if not summaries:
                continue
                
            model_display = NamingConventions.format_model_name(model_name)
            best_summary = self._get_first_summary(summaries)
            
            best_chi2 = best_summary.get("best_chi2", 0.0)
            weighted_chi2 = best_summary.get("weighted_chi2", 0.0)
            total_dof = best_summary.get("total_dof", 0)
            chi2_per_dof = best_chi2 / total_dof if total_dof > 0 else 0.0
            
            # Calculate information criteria
            n_params = len(best_summary.get("best_params", {}))
            aic = best_chi2 + 2 * n_params
            bic = best_chi2 + n_params * np.log(total_dof) if total_dof > 0 else float('inf')
            
            lines.append(f"{model_display}:")
            lines.append(f"  • Best χ²: {best_chi2:.1f}")
            lines.append(f"  • Weighted χ²: {weighted_chi2:.1f}")
            lines.append(f"  • Degrees of freedom: {total_dof}")
            lines.append(f"  • χ²/dof: {chi2_per_dof:.2f}")
            lines.append(f"  • AIC: {aic:.1f}")
            lines.append(f"  • BIC: {bic:.1f}")
            
            # Add jackknife chi² statistics if available
            if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                jackknife_model = jackknife_summary["models"][model_name]
                mean_chi2 = jackknife_model.get("mean_chi2", 0.0)
                std_chi2 = jackknife_model.get("std_chi2", 0.0)
                min_chi2 = jackknife_model.get("min_chi2", 0.0)
                max_chi2 = jackknife_model.get("max_chi2", 0.0)
                
                lines.append(f"  • Jackknife χ²: {mean_chi2:.1f} ± {std_chi2:.1f}")
                lines.append(f"  • Jackknife χ² range: [{min_chi2:.1f}, {max_chi2:.1f}]")
            
            lines.append("")
        
        return lines
    
    def _generate_derived_quantities(self, 
                                   model_summaries: Dict[str, List[Dict[str, Any]]],
                                   jackknife_summary: Dict[str, Any] | None = None) -> List[str]:
        """Generate derived quantities section."""
        lines = []
        lines.append("DERIVED QUANTITIES")
        lines.append("-" * 20)
        lines.append("")
        
        # Collect all derived quantities
        all_quantities = set()
        for summaries in model_summaries.values():
            if summaries:
                all_quantities.update(self._get_first_summary(summaries).get("derived_quantities", {}).keys())
        
        # Sort quantities in standard order
        quantity_order = ["r_d", "S8", "q0", "z_star", "age_Gyr", "sigma8"]
        sorted_quantities = [q for q in quantity_order if q in all_quantities] + sorted([q for q in all_quantities if q not in quantity_order])
        
        for quantity_name in sorted_quantities:
            quantity_display = NamingConventions.format_derived_quantity_name(quantity_name)
            units = NamingConventions.get_derived_quantity_unit(quantity_name)
            unit_str = f" {units}" if units else ""
            
            lines.append(f"{quantity_display}{unit_str}:")
            
            for model_name, summaries in model_summaries.items():
                if summaries:
                    derived = self._get_first_summary(summaries).get("derived_quantities", {})
                    value = derived.get(quantity_name)
                    
                    if value is not None:
                        model_display = NamingConventions.format_model_name(model_name)
                        
                        # Add uncertainty if available
                        uncertainty = None
                        if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                            jackknife_params = jackknife_summary["models"][model_name].get("parameters", {})
                            if quantity_name in jackknife_params:
                                uncertainty = jackknife_params[quantity_name].get("std")
                        
                        if uncertainty is not None:
                            lines.append(f"  • {model_display}: {value:.3f} ± {uncertainty:.3f}")
                        else:
                            lines.append(f"  • {model_display}: {value:.3f}")
            
            lines.append("")
        
        return lines
    
    def _generate_rd_comparison(self, model_summaries: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate r_d comparison section."""
        lines = []
        lines.append("SOUND HORIZON COMPARISON")
        lines.append("-" * 20)
        lines.append("")
        
        rd_values = []
        for model_name, summaries in model_summaries.items():
            if summaries:
                derived = self._get_first_summary(summaries).get("derived_quantities", {})
                rd = derived.get("r_d")
                if rd is not None:
                    rd_values.append((model_name, rd))
        
        if not rd_values:
            lines.append("No sound horizon values available.")
            return lines
        
        lines.append("Sound horizon at drag epoch (r_d):")
        for model_name, rd in rd_values:
            model_display = NamingConventions.format_model_name(model_name)
            lines.append(f"  • {model_display}: r_d = {rd:.2f} Mpc")
        
        # Calculate differences
        if len(rd_values) > 1:
            lines.append("")
            lines.append("Differences:")
            for i, (model1, rd1) in enumerate(rd_values):
                for j, (model2, rd2) in enumerate(rd_values[i+1:], i+1):
                    diff = rd1 - rd2
                    model1_display = NamingConventions.format_model_name(model1)
                    model2_display = NamingConventions.format_model_name(model2)
                    lines.append(f"  • {model1_display} - {model2_display}: {diff:.2f} Mpc")
        
        lines.append("")
        return lines
    
    def _generate_h0_s8_results(self, 
                              model_summaries: Dict[str, List[Dict[str, Any]]],
                              jackknife_summary: Dict[str, Any] | None = None) -> List[str]:
        """Generate H₀ and S₈ results section."""
        lines = []
        lines.append("HUBBLE CONSTANT AND MATTER FLUCTUATIONS")
        lines.append("-" * 20)
        lines.append("")
        
        # H₀ results
        h0_results = []
        for model_name, summaries in model_summaries.items():
            if summaries:
                params = self._get_first_summary(summaries).get("best_params", {})
                h0 = params.get("H0")
                if h0 is not None:
                    uncertainty = None
                    if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                        jackknife_params = jackknife_summary["models"][model_name].get("parameters", {})
                        if "H0" in jackknife_params:
                            uncertainty = jackknife_params["H0"].get("std")
                    
                    h0_results.append((model_name, h0, uncertainty))
        
        if h0_results:
            lines.append("Hubble constant (H₀):")
            for model_name, h0, uncertainty in h0_results:
                model_display = NamingConventions.format_model_name(model_name)
                if uncertainty is not None:
                    lines.append(f"  • {model_display}: H₀ = {h0:.1f} ± {uncertainty:.1f} km s⁻¹ Mpc⁻¹")
                else:
                    lines.append(f"  • {model_display}: H₀ = {h0:.1f} km s⁻¹ Mpc⁻¹")
            lines.append("")
        
        # S₈ results
        s8_results = []
        for model_name, summaries in model_summaries.items():
            if summaries:
                derived = self._get_first_summary(summaries).get("derived_quantities", {})
                s8 = derived.get("S8")
                if s8 is not None:
                    uncertainty = None
                    if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                        jackknife_params = jackknife_summary["models"][model_name].get("parameters", {})
                        if "S8" in jackknife_params:
                            uncertainty = jackknife_params["S8"].get("std")
                    
                    s8_results.append((model_name, s8, uncertainty))
        
        if s8_results:
            lines.append("Matter fluctuation amplitude (S₈):")
            for model_name, s8, uncertainty in s8_results:
                model_display = NamingConventions.format_model_name(model_name)
                if uncertainty is not None:
                    lines.append(f"  • {model_display}: S₈ = {s8:.3f} ± {uncertainty:.3f}")
                else:
                    lines.append(f"  • {model_display}: S₈ = {s8:.3f}")
            lines.append("")
        
        return lines
    
    def _generate_predictions_summary(self, model_summaries: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate predictions summary section."""
        lines = []
        lines.append("MODEL PREDICTIONS")
        lines.append("-" * 20)
        lines.append("")
        
        lines.append("Key cosmological predictions:")
        
        # Age of Universe
        age_results = []
        for model_name, summaries in model_summaries.items():
            if summaries:
                derived = self._get_first_summary(summaries).get("derived_quantities", {})
                age = derived.get("age_Gyr")
                if age is not None:
                    age_results.append((model_name, age))
        
        if age_results:
            lines.append("  • Age of Universe:")
            for model_name, age in age_results:
                model_display = NamingConventions.format_model_name(model_name)
                lines.append(f"    - {model_display}: {age:.2f} Gyr")
        
        # Deceleration parameter
        q0_results = []
        for model_name, summaries in model_summaries.items():
            if summaries:
                derived = self._get_first_summary(summaries).get("derived_quantities", {})
                q0 = derived.get("q0")
                if q0 is not None:
                    q0_results.append((model_name, q0))
        
        if q0_results:
            lines.append("  • Present-day deceleration parameter (q₀):")
            for model_name, q0 in q0_results:
                model_display = NamingConventions.format_model_name(model_name)
                lines.append(f"    - {model_display}: q₀ = {q0:.3f}")
        
        # Redshift of recombination
        z_star_results = []
        for model_name, summaries in model_summaries.items():
            if summaries:
                derived = self._get_first_summary(summaries).get("derived_quantities", {})
                z_star = derived.get("z_star")
                if z_star is not None:
                    z_star_results.append((model_name, z_star))
        
        if z_star_results:
            lines.append("  • Redshift of recombination (z_*):")
            for model_name, z_star in z_star_results:
                model_display = NamingConventions.format_model_name(model_name)
                lines.append(f"    - {model_display}: z_* = {z_star:.1f}")
        
        lines.append("")
        return lines
    
    def _generate_jackknife_highlights(self, jackknife_summary: Dict[str, Any]) -> List[str]:
        """Generate jackknife stability highlights section."""
        lines = []
        lines.append("JACKKNIFE STABILITY ANALYSIS")
        lines.append("-" * 20)
        lines.append("")
        
        n_draws = jackknife_summary.get("n_draws", 0)
        fraction_removed = jackknife_summary.get("fraction_removed", 0.0)
        total_successful = jackknife_summary.get("total_successful_draws", 0)
        
        lines.append(f"Jackknife configuration:")
        lines.append(f"  • Total draws: {n_draws}")
        lines.append(f"  • Fraction removed per draw: {fraction_removed:.1%}")
        lines.append(f"  • Successful draws: {total_successful} ({total_successful/n_draws:.1%})")
        lines.append("")
        
        lines.append("Parameter stability:")
        for model_name, model_data in jackknife_summary.get("models", {}).items():
            model_display = NamingConventions.format_model_name(model_name)
            lines.append(f"  {model_display}:")
            
            params = model_data.get("parameters", {})
            for param_name, param_stats in params.items():
                if isinstance(param_stats, dict) and "std" in param_stats:
                    mean_val = param_stats.get("mean", 0.0)
                    std_val = param_stats.get("std", 0.0)
                    relative_uncertainty = std_val / abs(mean_val) if mean_val != 0 else float('inf')
                    
                    param_display = NamingConventions.format_parameter_name(param_name)
                    
                    # Classify stability
                    if relative_uncertainty < 0.01:
                        stability = "Excellent"
                    elif relative_uncertainty < 0.05:
                        stability = "Good"
                    elif relative_uncertainty < 0.10:
                        stability = "Moderate"
                    else:
                        stability = "Poor"
                    
                    lines.append(f"    • {param_display}: {mean_val:.3f} ± {std_val:.3f} ({stability} stability)")
        
        lines.append("")
        return lines
    
    def _generate_quantum_results(self, quantum_state: Dict[str, Any]) -> List[str]:
        """Generate quantum results section."""
        lines = []
        lines.append("QUANTUM ENGINE RESULTS")
        lines.append("-" * 20)
        lines.append("")
        
        eps0 = quantum_state.get("eps0", 0.0)
        eps0_err = quantum_state.get("eps0_error", 0.0)
        alpha_qm = quantum_state.get("alpha_QM", 0.0)
        alpha_err = quantum_state.get("alpha_error", 0.0)
        
        lines.append("Quantum spacetime rigidity parameters:")
        lines.append(f"  • Rigidity parameter (ε₀): {eps0:.3e} ± {eps0_err:.3e}")
        lines.append(f"  • Quantum mixing strength (α_QM): {alpha_qm:.3e} ± {alpha_err:.3e}")
        lines.append("")
        
        derived = quantum_state.get("derived_parameters", {})
        field_content = derived.get("field_set")
        regulator = derived.get("regulator")
        
        lines.append("Quantum field configuration:")
        if field_content:
            lines.append(f"  • Field content: {field_content}")
        if regulator:
            lines.append(f"  • Regulator: {regulator}")
        
        # Additional derived parameters
        f_cut = derived.get("f_cut")
        f_coup = derived.get("f_coup")
        
        if f_cut is not None:
            lines.append(f"  • Cutoff frequency (f_cut): {f_cut:.3f}")
        if f_coup is not None:
            lines.append(f"  • Coupling strength (f_coup): {f_coup:.3f}")
        
        lines.append("")
        return lines
    
    def _generate_model_comparison(self, model_summaries: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate model comparison section."""
        lines = []
        lines.append("MODEL COMPARISON")
        lines.append("-" * 20)
        lines.append("")
        
        # Compare models by various metrics
        models_data = []
        for model_name, summaries in model_summaries.items():
            if summaries:
                best_summary = self._get_first_summary(summaries)
                chi2 = best_summary.get("best_chi2", 0.0)
                dof = best_summary.get("total_dof", 0)
                chi2_per_dof = chi2 / dof if dof > 0 else float('inf')
                
                n_params = len(best_summary.get("best_params", {}))
                aic = chi2 + 2 * n_params
                bic = chi2 + n_params * np.log(dof) if dof > 0 else float('inf')
                
                models_data.append((model_name, chi2_per_dof, aic, bic))
        
        # Sort by chi2/dof
        models_data.sort(key=lambda x: x[1])
        
        lines.append("Models ranked by χ²/dof:")
        for i, (model_name, chi2_per_dof, aic, bic) in enumerate(models_data, 1):
            model_display = NamingConventions.format_model_name(model_name)
            lines.append(f"  {i}. {model_display}: χ²/dof = {chi2_per_dof:.2f}, AIC = {aic:.1f}, BIC = {bic:.1f}")
        
        lines.append("")
        return lines
    
    def _generate_quality_assessment(self, 
                                   model_summaries: Dict[str, List[Dict[str, Any]]],
                                   jackknife_summary: Dict[str, Any] | None = None) -> List[str]:
        """Generate quality assessment section."""
        lines = []
        lines.append("QUALITY ASSESSMENT")
        lines.append("-" * 20)
        lines.append("")
        
        for model_name, summaries in model_summaries.items():
            if not summaries:
                continue
                
            model_display = NamingConventions.format_model_name(model_name)
            best_summary = self._get_first_summary(summaries)
            
            chi2 = best_summary.get("best_chi2", 0.0)
            dof = best_summary.get("total_dof", 0)
            chi2_per_dof = chi2 / dof if dof > 0 else 0.0
            
            lines.append(f"{model_display} quality assessment:")
            
            # Goodness of fit
            if chi2_per_dof < 1.0:
                fit_quality = "Excellent"
            elif chi2_per_dof < 2.0:
                fit_quality = "Good"
            elif chi2_per_dof < 5.0:
                fit_quality = "Acceptable"
            else:
                fit_quality = "Poor"
            
            lines.append(f"  • Goodness of fit: {fit_quality} (χ²/dof = {chi2_per_dof:.2f})")
            
            # Parameter constraints
            n_params = len(best_summary.get("best_params", {}))
            if n_params <= 6:
                constraint_quality = "Well-constrained"
            elif n_params <= 10:
                constraint_quality = "Moderately constrained"
            else:
                constraint_quality = "Under-constrained"
            
            lines.append(f"  • Parameter constraints: {constraint_quality} ({n_params} free parameters)")
            
            # Jackknife stability
            if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                jackknife_model = jackknife_summary["models"][model_name]
                n_successful = jackknife_model.get("n_successful_draws", 0)
                total_draws = jackknife_summary.get("n_draws", 0)
                success_rate = n_successful / total_draws if total_draws > 0 else 0.0
                
                if success_rate > 0.95:
                    stability_quality = "Very stable"
                elif success_rate > 0.80:
                    stability_quality = "Stable"
                elif success_rate > 0.50:
                    stability_quality = "Moderately stable"
                else:
                    stability_quality = "Unstable"
                
                lines.append(f"  • Numerical stability: {stability_quality} ({success_rate:.1%} successful jackknife draws)")
            else:
                lines.append("  • Numerical stability: Not assessed")
            
            lines.append("")
        
        return lines
    
    def _generate_recommendations(self, 
                                model_summaries: Dict[str, List[Dict[str, Any]]],
                                jackknife_summary: Dict[str, Any] | None = None) -> List[str]:
        """Generate recommendations section."""
        lines = []
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 20)
        lines.append("")
        
        # Find best performing model
        best_model = None
        best_chi2_dof = float('inf')
        
        for model_name, summaries in model_summaries.items():
            if summaries:
                best_summary = self._get_first_summary(summaries)
                chi2 = best_summary.get("best_chi2", 0.0)
                dof = best_summary.get("total_dof", 0)
                chi2_per_dof = chi2 / dof if dof > 0 else float('inf')
                
                if chi2_per_dof < best_chi2_dof:
                    best_chi2_dof = chi2_per_dof
                    best_model = model_name
        
        if best_model:
            best_display = NamingConventions.format_model_name(best_model)
            lines.append(f"• Primary recommendation: {best_display} model")
            lines.append(f"  Reason: Best goodness of fit (χ²/dof = {best_chi2_dof:.2f})")
        
        # Dataset recommendations
        lines.append("")
        lines.append("• Dataset recommendations:")
        
        # Check for high chi2 contributions
        high_chi2_datasets = []
        for model_name, summaries in model_summaries.items():
            if summaries:
                chi2_breakdown = self._get_first_summary(summaries).get("chi2_breakdown", {})
                for dataset_name, chi2_info in chi2_breakdown.items():
                    chi2 = chi2_info.get("chi2", 0.0)
                    dof = chi2_info.get("dof", 0)
                    chi2_per_dof = chi2 / dof if dof > 0 else 0.0
                    
                    if chi2_per_dof > 3.0:  # High chi2 threshold
                        high_chi2_datasets.append((dataset_name, chi2_per_dof))
        
        if high_chi2_datasets:
            lines.append("  • Consider reviewing datasets with high χ² contributions:")
            for dataset_name, chi2_per_dof in high_chi2_datasets:
                dataset_display = NamingConventions.format_dataset_name(dataset_name)
                lines.append(f"    - {dataset_display}: χ²/dof = {chi2_per_dof:.2f}")
        else:
            lines.append("  • All datasets show reasonable χ² contributions")
        
        # Analysis recommendations
        lines.append("")
        lines.append("• Analysis recommendations:")
        
        if jackknife_summary:
            lines.append("  • Jackknife analysis suggests parameter uncertainties are robust")
        else:
            lines.append("  • Consider running jackknife analysis to assess parameter uncertainties")
        
        if len(model_summaries) > 1:
            lines.append("  • Multiple models tested - compare information criteria for model selection")
        else:
            lines.append("  • Consider testing alternative cosmological models for comparison")
        
        lines.append("")
        return lines
    
    def _generate_metadata_summary(self) -> List[str]:
        """Generate metadata summary section."""
        lines = []
        lines.append("ANALYSIS METADATA")
        lines.append("-" * 20)
        lines.append("")
        
        # Try to read run metadata
        try:
            meta_file = self.run_dir / "run_meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                
                timestamp = meta.get("timestamp", "")
                run_name = meta.get("run_name", "")
                mode = meta.get("mode", "")
                engine = meta.get("engine", "")
                runtime = meta.get("total_runtime", 0.0)
                
                lines.append(f"Run name: {run_name}")
                lines.append(f"Timestamp: {timestamp}")
                lines.append(f"Mode: {mode}")
                lines.append(f"Engine: {engine}")
                lines.append(f"Runtime: {runtime:.1f} seconds")
                
                # Git information
                git_info = meta.get("git", {})
                commit_hash = git_info.get("commit_hash", "")
                git_status = git_info.get("status", "")
                
                if commit_hash and commit_hash != "unknown":
                    lines.append(f"Git commit: {commit_hash[:8]}")
                    if git_status:
                        lines.append(f"Git status: {git_status}")
                
                # Machine information
                machine = meta.get("machine", {})
                node = machine.get("node", "")
                cpus = machine.get("cpus", 0)
                
                if node:
                    lines.append(f"Machine: {node}")
                if cpus:
                    lines.append(f"CPU cores: {cpus}")
                
                # Datasets used
                fits_used = meta.get("fits_used", [])
                if fits_used:
                    lines.append(f"Datasets: {', '.join(fits_used)}")
                
            else:
                lines.append("Run metadata not available")
        except Exception:
            lines.append("Error reading run metadata")
        
        lines.append("")
        return lines
