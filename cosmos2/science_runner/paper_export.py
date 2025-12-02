"""Paper export utilities for cosmos2 science runner - machine-readable formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from cosmos2.science_runner.naming_conventions import NamingConventions
from cosmos2.science_runner.table_generator import ScientificTableGenerator


def _get_best_summary(summaries):
    """Helper function to get the best summary from either list or dict format."""
    if isinstance(summaries, dict):
        return summaries
    elif isinstance(summaries, list) and len(summaries) > 0:
        return summaries[0]
    else:
        return None


class PaperExportGenerator:
    """Generates machine-readable exports for paper writing."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        
    def generate_all_paper_exports(self, 
                                 model_summaries: Dict[str, List[Dict[str, Any]]],
                                 jackknife_summary: Dict[str, Any] | None = None,
                                 quantum_state: Dict[str, Any] | None = None) -> Dict[str, Path]:
        """Generate all machine-readable paper exports."""
        exports = {}
        
        # 1. JSON parameter results table
        exports["parameters"] = self._generate_parameter_table(model_summaries, jackknife_summary)
        
        # 2. LaTeX parameter table
        exports["parameters_latex"] = self._generate_parameter_table_latex(model_summaries, jackknife_summary)
        
        # 3. LibreOffice parameter table
        exports["parameters_libreoffice"] = self._generate_parameter_table_libreoffice(model_summaries, jackknife_summary)
        
        # 4. Dataset statistics table
        exports["datasets"] = self._generate_dataset_table(model_summaries)
        
        # 5. LaTeX dataset table
        exports["datasets_latex"] = self._generate_dataset_table_latex(model_summaries)
        
        # 6. Chi² statistics table
        exports["chi2"] = self._generate_chi2_table(model_summaries, jackknife_summary)
        
        # 7. LaTeX chi² table
        exports["chi2_latex"] = self._generate_chi2_table_latex(model_summaries, jackknife_summary)
        
        # 8. Derived quantities table
        exports["derived"] = self._generate_derived_table(model_summaries, jackknife_summary)
        
        # 9. LaTeX derived quantities table
        exports["derived_latex"] = self._generate_derived_table_latex(model_summaries, jackknife_summary)
        
        # 10. Jackknife results (if available)
        if jackknife_summary:
            exports["jackknife"] = self._generate_jackknife_table(jackknife_summary)
            exports["jackknife_latex"] = self._generate_jackknife_table_latex(jackknife_summary)
        
        # 11. Quantum parameters (if available)
        if quantum_state:
            exports["quantum"] = self._generate_quantum_table(quantum_state)
            exports["quantum_latex"] = self._generate_quantum_table_latex(quantum_state)
        
        # 12. Complete machine-readable summary
        exports["summary"] = self._generate_complete_summary(
            model_summaries, jackknife_summary, quantum_state
        )
        
        return exports
    
    def _generate_parameter_table(self, 
                                model_summaries: Dict[str, List[Dict[str, Any]]],
                                jackknife_summary: Dict[str, Any] | None = None) -> Path:
        """Generate machine-readable parameter results table."""
        output_path = self.run_dir / "paper_exports" / "parameters.json"
        output_path.parent.mkdir(exist_ok=True)
        
        results = {}
        
        for model_name, summaries in model_summaries.items():
            if not summaries:
                continue
                
            # Handle both list and dict formats for summaries
            best_summary = _get_best_summary(summaries)
            if best_summary is None:
                continue
                
            params = best_summary.get("best_params", {})
            
            model_results = {
                "parameters": {},
                "uncertainties": {},
                "units": {},
                "descriptions": {}
            }
            
            # Add parameters with uncertainties
            for param_name, param_value in params.items():
                model_results["parameters"][param_name] = float(param_value)
                
                # Add jackknife uncertainties if available
                if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                    jackknife_model = jackknife_summary["models"][model_name]
                    jackknife_params = jackknife_model.get("parameters", {})
                    if param_name in jackknife_params:
                        model_results["uncertainties"][param_name] = float(jackknife_params[param_name]["std"])
                    else:
                        model_results["uncertainties"][param_name] = None
                else:
                    model_results["uncertainties"][param_name] = None
                
                # Add units and descriptions
                model_results["units"][param_name] = self._get_parameter_units(param_name)
                model_results["descriptions"][param_name] = self._get_parameter_description(param_name)
            
            results[model_name] = model_results
        
        # Add metadata
        export_data = {
            "title": "Cosmological Parameter Results",
            "description": "Best-fit parameters with uncertainties from cosmological model fitting",
            "models": results,
            "metadata": {
                "format": "json",
                "machine_readable": True,
                "paper_ready": True
            }
        }
        
        output_path.write_text(json.dumps(export_data, indent=2, default=str))
        return output_path
    
    def _generate_dataset_table(self, model_summaries: Dict[str, List[Dict[str, Any]]]) -> Path:
        """Generate machine-readable dataset statistics table."""
        output_path = self.run_dir / "paper_exports" / "datasets.json"
        output_path.parent.mkdir(exist_ok=True)
        
        results = {}
        
        for model_name, summaries in model_summaries.items():
            if not summaries:
                continue
                
            best_summary = _get_best_summary(summaries)
            if best_summary is None:
                continue
                
            chi2_breakdown = best_summary.get("chi2_breakdown", {})
            
            model_results = {
                "datasets": {},
                "chi2_contributions": {},
                "data_points": {},
                "chi2_per_dof": {}
            }
            
            total_chi2 = 0.0
            total_dof = 0
            
            for dataset_name, chi2_info in chi2_breakdown.items():
                chi2_value = float(chi2_info.get("chi2", 0.0))
                dof = int(chi2_info.get("dof", 0))
                
                model_results["chi2_contributions"][dataset_name] = chi2_value
                model_results["data_points"][dataset_name] = dof
                
                if dof > 0:
                    model_results["chi2_per_dof"][dataset_name] = chi2_value / dof
                else:
                    model_results["chi2_per_dof"][dataset_name] = None
                
                total_chi2 += chi2_value
                total_dof += dof
                
                # Add dataset metadata
                model_results["datasets"][dataset_name] = {
                    "type": self._get_dataset_type(dataset_name),
                    "description": self._get_dataset_description(dataset_name),
                    "reference": self._get_dataset_reference(dataset_name)
                }
            
            # Add totals
            model_results["total_chi2"] = total_chi2
            model_results["total_dof"] = total_dof
            if total_dof > 0:
                model_results["total_chi2_per_dof"] = total_chi2 / total_dof
            else:
                model_results["total_chi2_per_dof"] = None
            
            results[model_name] = model_results
        
        export_data = {
            "title": "Dataset Statistics",
            "description": "Chi² contributions and data point counts for each dataset",
            "models": results,
            "metadata": {
                "format": "json",
                "machine_readable": True,
                "paper_ready": True
            }
        }
        
        output_path.write_text(json.dumps(export_data, indent=2, default=str))
        return output_path
    
    def _generate_chi2_table(self, 
                           model_summaries: Dict[str, List[Dict[str, Any]]],
                           jackknife_summary: Dict[str, Any] | None = None) -> Path:
        """Generate machine-readable chi² statistics table."""
        output_path = self.run_dir / "paper_exports" / "chi2_statistics.json"
        output_path.parent.mkdir(exist_ok=True)
        
        results = {}
        
        for model_name, summaries in model_summaries.items():
            if not summaries:
                continue
                
            best_summary = _get_best_summary(summaries)
            if best_summary is None:
                continue
            
            model_results = {
                "best_chi2": float(best_summary.get("best_chi2", 0.0)),
                "weighted_chi2": float(best_summary.get("weighted_chi2", 0.0)),
                "total_dof": int(best_summary.get("total_dof", 0)),
                "chi2_per_dof": None,
                "aic": None,
                "bic": None
            }
            
            # Calculate chi² per dof
            total_dof = model_results["total_dof"]
            if total_dof > 0:
                model_results["chi2_per_dof"] = model_results["best_chi2"] / total_dof
                
                # Calculate information criteria
                n_params = len(best_summary.get("best_params", {}))
                model_results["aic"] = model_results["best_chi2"] + 2 * n_params
                model_results["bic"] = model_results["best_chi2"] + n_params * np.log(total_dof)
            
            # Add jackknife chi² statistics if available
            if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                jackknife_model = jackknife_summary["models"][model_name]
                model_results["jackknife"] = {
                    "mean_chi2": float(jackknife_model.get("mean_chi2", 0.0)),
                    "std_chi2": float(jackknife_model.get("std_chi2", 0.0)),
                    "min_chi2": float(jackknife_model.get("min_chi2", 0.0)),
                    "max_chi2": float(jackknife_model.get("max_chi2", 0.0)),
                    "n_successful_draws": int(jackknife_model.get("n_successful_draws", 0))
                }
            
            results[model_name] = model_results
        
        export_data = {
            "title": "Chi² Statistics",
            "description": "Goodness-of-fit statistics for cosmological models",
            "models": results,
            "metadata": {
                "format": "json",
                "machine_readable": True,
                "paper_ready": True
            }
        }
        
        output_path.write_text(json.dumps(export_data, indent=2, default=str))
        return output_path
    
    def _generate_derived_table(self, 
                              model_summaries: Dict[str, List[Dict[str, Any]]],
                              jackknife_summary: Dict[str, Any] | None = None) -> Path:
        """Generate machine-readable derived quantities table."""
        output_path = self.run_dir / "paper_exports" / "derived_quantities.json"
        output_path.parent.mkdir(exist_ok=True)
        
        results = {}
        
        for model_name, summaries in model_summaries.items():
            if not summaries:
                continue
                
            best_summary = _get_best_summary(summaries)
            if best_summary is None:
                continue
                
            derived = best_summary.get("derived_quantities", {})
            
            model_results = {
                "quantities": {},
                "uncertainties": {},
                "units": {},
                "descriptions": {}
            }
            
            # Add derived quantities
            for quantity_name, quantity_value in derived.items():
                model_results["quantities"][quantity_name] = float(quantity_value)
                
                # Add jackknife uncertainties if available
                if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                    jackknife_model = jackknife_summary["models"][model_name]
                    jackknife_params = jackknife_model.get("parameters", {})
                    if quantity_name in jackknife_params:
                        model_results["uncertainties"][quantity_name] = float(jackknife_params[quantity_name]["std"])
                    else:
                        model_results["uncertainties"][quantity_name] = None
                else:
                    model_results["uncertainties"][quantity_name] = None
                
                # Add units and descriptions
                model_results["units"][quantity_name] = self._get_derived_units(quantity_name)
                model_results["descriptions"][quantity_name] = self._get_derived_description(quantity_name)
            
            results[model_name] = model_results
        
        export_data = {
            "title": "Derived Cosmological Quantities",
            "description": "Derived quantities from best-fit cosmological parameters",
            "models": results,
            "metadata": {
                "format": "json",
                "machine_readable": True,
                "paper_ready": True
            }
        }
        
        output_path.write_text(json.dumps(export_data, indent=2, default=str))
        return output_path
    
    def _generate_jackknife_table(self, jackknife_summary: Dict[str, Any]) -> Path:
        """Generate machine-readable jackknife results table."""
        output_path = self.run_dir / "paper_exports" / "jackknife_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        results = {
            "summary": {
                "n_draws": int(jackknife_summary.get("n_draws", 0)),
                "fraction_removed": float(jackknife_summary.get("fraction_removed", 0.0)),
                "random_seed": jackknife_summary.get("random_seed"),
                "total_successful_draws": int(jackknife_summary.get("total_successful_draws", 0))
            },
            "models": {}
        }
        
        for model_name, model_data in jackknife_summary.get("models", {}).items():
            results["models"][model_name] = {
                "parameters": model_data.get("parameters", {}),
                "chi2_statistics": {
                    "mean_chi2": float(model_data.get("mean_chi2", 0.0)),
                    "std_chi2": float(model_data.get("std_chi2", 0.0)),
                    "min_chi2": float(model_data.get("min_chi2", 0.0)),
                    "max_chi2": float(model_data.get("max_chi2", 0.0))
                },
                "n_successful_draws": int(model_data.get("n_successful_draws", 0))
            }
        
        export_data = {
            "title": "Jackknife Resampling Results",
            "description": "Statistical uncertainty estimation through jackknife resampling",
            "results": results,
            "metadata": {
                "format": "json",
                "machine_readable": True,
                "paper_ready": True
            }
        }
        
        output_path.write_text(json.dumps(export_data, indent=2, default=str))
        return output_path
    
    def _generate_quantum_table(self, quantum_state: Dict[str, Any]) -> Path:
        """Generate machine-readable quantum parameters table."""
        output_path = self.run_dir / "paper_exports" / "quantum_parameters.json"
        output_path.parent.mkdir(exist_ok=True)
        
        results = {
            "quantum_parameters": {
                "eps0": {
                    "value": float(quantum_state.get("eps0", 0.0)),
                    "uncertainty": float(quantum_state.get("eps0_error", 0.0)),
                    "unit": "",
                    "description": "Spacetime rigidity parameter"
                },
                "alpha_QM": {
                    "value": float(quantum_state.get("alpha_QM", 0.0)),
                    "uncertainty": float(quantum_state.get("alpha_error", 0.0)),
                    "unit": "",
                    "description": "Quantum mixing strength"
                }
            },
            "derived_parameters": {},
            "run_metadata": {
                "runtime_seconds": quantum_state.get("run_metadata", {}).get("stats", {}).get("runtime_seconds"),
                "events_processed": quantum_state.get("run_metadata", {}).get("stats", {}).get("events", {}).get("n_valid"),
                "field_content": quantum_state.get("derived_parameters", {}).get("field_set"),
                "regulator": quantum_state.get("derived_parameters", {}).get("regulator")
            }
        }
        
        # Add derived parameters
        derived = quantum_state.get("derived_parameters", {})
        for param_name, param_value in derived.items():
            if isinstance(param_value, (int, float)):
                results["derived_parameters"][param_name] = {
                    "value": float(param_value),
                    "unit": self._get_quantum_param_units(param_name),
                    "description": self._get_quantum_param_description(param_name)
                }
        
        export_data = {
            "title": "Quantum Engine Parameters",
            "description": "Parameters from quantum spacetime rigidity analysis",
            "results": results,
            "metadata": {
                "format": "json",
                "machine_readable": True,
                "paper_ready": True
            }
        }
        
        output_path.write_text(json.dumps(export_data, indent=2, default=str))
        return output_path
    
    def _generate_complete_summary(self, 
                                  model_summaries: Dict[str, List[Dict[str, Any]]],
                                  jackknife_summary: Dict[str, Any] | None = None,
                                  quantum_state: Dict[str, Any] | None = None) -> Path:
        """Generate complete machine-readable summary for paper writing."""
        output_path = self.run_dir / "paper_exports" / "complete_summary.json"
        output_path.parent.mkdir(exist_ok=True)
        
        summary = {
            "title": "Complete Cosmological Analysis Results",
            "description": "Machine-readable summary of all cosmological model fitting results",
            "analysis_info": {
                "models": list(model_summaries.keys()),
                "has_jackknife": jackknife_summary is not None,
                "has_quantum": quantum_state is not None,
                "timestamp": self._get_run_timestamp()
            },
            "key_results": {}
        }
        
        # Extract key results for easy access
        for model_name, summaries in model_summaries.items():
            if not summaries:
                continue
                
            best_summary = _get_best_summary(summaries)
            if best_summary is None:
                continue
                
            params = best_summary.get("best_params", {})
            derived = best_summary.get("derived_quantities", {})
            
            summary["key_results"][model_name] = {
                "H0": params.get("H0"),
                "Omega_m0": params.get("Omega_m0"),
                "r_d": derived.get("r_d"),
                "S8": derived.get("S8"),
                "q0": derived.get("q0"),
                "best_chi2": best_summary.get("best_chi2"),
                "chi2_per_dof": best_summary.get("best_chi2", 0.0) / best_summary.get("total_dof", 1)
            }
            
            # Add jackknife uncertainties if available
            if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                jackknife_params = jackknife_summary["models"][model_name].get("parameters", {})
                summary["key_results"][model_name]["jackknife_uncertainties"] = {
                    "H0": jackknife_params.get("H0", {}).get("std"),
                    "Omega_m0": jackknife_params.get("Omega_m0", {}).get("std"),
                    "r_d": jackknife_params.get("r_d", {}).get("std"),
                    "S8": jackknife_params.get("S8", {}).get("std")
                }
        
        # Add quantum results if available
        if quantum_state:
            summary["quantum_results"] = {
                "eps0": quantum_state.get("eps0"),
                "alpha_QM": quantum_state.get("alpha_QM"),
                "field_content": quantum_state.get("derived_parameters", {}).get("field_set"),
                "regulator": quantum_state.get("derived_parameters", {}).get("regulator")
            }
        
        export_data = {
            "summary": summary,
            "metadata": {
                "format": "json",
                "machine_readable": True,
                "paper_ready": True,
                "comprehensive": True
            }
        }
        
        output_path.write_text(json.dumps(export_data, indent=2, default=str))
        return output_path
    
    # Helper methods for metadata
    def _get_parameter_units(self, param_name: str) -> str:
        """Get units for a parameter."""
        units_map = {
            "H0": "km/s/Mpc",
            "Omega_m0": "",
            "Omega_b0": "",
            "Omega_k0": "",
            "Rmax": "GeV",
            "alpha": ""
        }
        return units_map.get(param_name, "")
    
    def _get_parameter_description(self, param_name: str) -> str:
        """Get description for a parameter."""
        desc_map = {
            "H0": "Hubble constant",
            "Omega_m0": "Matter density parameter",
            "Omega_b0": "Baryon density parameter", 
            "Omega_k0": "Curvature density parameter",
            "Rmax": "Maximum rigidity scale",
            "alpha": "PBUF model parameter"
        }
        return desc_map.get(param_name, "")
    
    def _get_dataset_type(self, dataset_name: str) -> str:
        """Get type for a dataset."""
        type_map = {
            "cmb": "Cosmic Microwave Background",
            "sn": "Type Ia Supernovae",
            "bao_iso": "Isotropic Baryon Acoustic Oscillations",
            "bao_aniso": "Anisotropic Baryon Acoustic Oscillations",
            "cc": "Cosmic Chronometers",
            "rsd": "Redshift-Space Distortions"
        }
        return type_map.get(dataset_name, "")
    
    def _get_dataset_description(self, dataset_name: str) -> str:
        """Get description for a dataset."""
        desc_map = {
            "cmb": "Planck CMB temperature and polarization anisotropies",
            "sn": "Pantheon supernova distance modulus compilation",
            "bao_iso": "Isotropic BAO distance measurements from various surveys",
            "bao_aniso": "Anisotropic BAO measurements providing angular diameter distance and H(z)",
            "cc": "Passively evolving galaxy chronometers providing H(z) measurements",
            "rsd": "Growth rate measurements from redshift-space distortions"
        }
        return desc_map.get(dataset_name, "")
    
    def _get_dataset_reference(self, dataset_name: str) -> str:
        """Get reference for a dataset."""
        ref_map = {
            "cmb": "Planck Collaboration 2018",
            "sn": "Scolnic et al. 2018, Pantheon sample",
            "bao_iso": "Various surveys (BOSS, eBOSS, 6dF, SDSS, etc.)",
            "bao_aniso": "BOSS DR12, eBOSS DR16, etc.",
            "cc": "Moresco et al. 2016 compilation",
            "rsd": "Various spectroscopic surveys (BOSS, eBOSS, VIPERS, etc.)"
        }
        return ref_map.get(dataset_name, "")
    
    def _get_derived_units(self, quantity_name: str) -> str:
        """Get units for a derived quantity."""
        units_map = {
            "r_d": "Mpc",
            "S8": "",
            "q0": "",
            "z_star": "",
            "age_Gyr": "Gyr",
            "sigma8": ""
        }
        return units_map.get(quantity_name, "")
    
    def _get_derived_description(self, quantity_name: str) -> str:
        """Get description for a derived quantity."""
        desc_map = {
            "r_d": "Sound horizon at drag epoch",
            "S8": "Amplitude of matter fluctuations",
            "q0": "Present-day deceleration parameter",
            "z_star": "Redshift of recombination",
            "age_Gyr": "Age of the Universe",
            "sigma8": "RMS amplitude of matter fluctuations at 8 Mpc/h"
        }
        return desc_map.get(quantity_name, "")
    
    def _get_quantum_param_units(self, param_name: str) -> str:
        """Get units for a quantum parameter."""
        units_map = {
            "f_cut": "",
            "f_coup": "",
            "mixing_strength": ""
        }
        return units_map.get(param_name, "")
    
    def _get_quantum_param_description(self, param_name: str) -> str:
        """Get description for a quantum parameter."""
        desc_map = {
            "f_cut": "Cutoff frequency",
            "f_coup": "Coupling strength",
            "mixing_strength": "Mixing parameter",
            "field_set": "Effective field content",
            "regulator": "Loop regularization scheme"
        }
        return desc_map.get(param_name, "")
    
    def _get_run_timestamp(self) -> str:
        """Get run timestamp from metadata."""
        try:
            meta_file = self.run_dir / "run_meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                return meta.get("timestamp", "")
        except Exception:
            pass
        return ""

    def _generate_parameter_table_latex(self, 
                                      model_summaries: Dict[str, List[Dict[str, Any]]],
                                      jackknife_summary: Dict[str, Any] | None = None) -> Path:
        """Generate LaTeX parameter results table."""
        output_path = self.run_dir / "paper_exports" / "parameters.tex"
        output_path.parent.mkdir(exist_ok=True)
        
        # Collect all parameters across models
        all_params = set()
        for summaries in model_summaries.values():
            if summaries:
                best_summary = _get_best_summary(summaries)
                if best_summary:
                    all_params.update(best_summary.get("best_params", {}).keys())
        
        # Sort parameters for consistent ordering
        param_order = ["H0", "Omega_m0", "Omega_b0", "Omega_k0", "Rmax", "alpha"]
        sorted_params = [p for p in param_order if p in all_params] + sorted([p for p in all_params if p not in param_order])
        
        latex_content = []
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{" + NamingConventions.get_table_caption('parameters') + "}")
        latex_content.append("\\label{" + NamingConventions.get_table_id('parameters') + "}")
        
        # Build table header
        header = ["Parameter"]
        for model_name in model_summaries.keys():
            header.append(NamingConventions.format_model_name(model_name))
        latex_content.append("\\begin{tabular}{" + "c" * len(header) + "}")
        latex_content.append(" \\\\ \\hline")
        latex_content.append(" & ".join(header) + " \\\\ \\hline")
        
        # Add parameter rows
        for param_name in sorted_params:
            row = [NamingConventions.format_parameter_name(param_name)]
            
            for model_name, summaries in model_summaries.items():
                if summaries:
                    best_summary = _get_best_summary(summaries)
                    if best_summary:
                        params = best_summary.get("best_params", {})
                        value = params.get(param_name, None)
                    
                    if value is not None:
                        # Add uncertainty if available from jackknife
                        uncertainty = None
                        if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                            jackknife_params = jackknife_summary["models"][model_name].get("parameters", {})
                            if param_name in jackknife_params:
                                uncertainty = jackknife_params[param_name].get("std")
                        
                        if uncertainty is not None:
                            formatted_value = f"${value:.3f} \\pm {uncertainty:.3f}$"
                        else:
                            formatted_value = f"${value:.3f}$"
                        
                        # Add units
                        units = NamingConventions.get_parameter_unit(param_name)
                        if units:
                            formatted_value += f"\\,\\mathrm{{{units}}}"
                        
                        row.append(formatted_value)
                    else:
                        row.append("--")
                else:
                    row.append("--")
            
            latex_content.append(" & ".join(row) + " \\\\")
        
        latex_content.append("\\hline")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        output_path.write_text("\n".join(latex_content))
        return output_path
    
    def _generate_parameter_table_libreoffice(self, 
                                           model_summaries: Dict[str, List[Dict[str, Any]]],
                                           jackknife_summary: Dict[str, Any] | None = None) -> Path:
        """Generate LibreOffice-compatible CSV parameter table."""
        output_path = self.run_dir / "paper_exports" / "parameters_libreoffice.csv"
        output_path.parent.mkdir(exist_ok=True)
        
        import csv
        
        # Collect all parameters across models
        all_params = set()
        for summaries in model_summaries.values():
            if summaries:
                best_summary = _get_best_summary(summaries)
                if best_summary:
                    all_params.update(best_summary.get("best_params", {}).keys())
        
        # Sort parameters for consistent ordering
        param_order = ["H0", "Omega_m0", "Omega_b0", "Omega_k0", "Rmax", "alpha"]
        sorted_params = [p for p in param_order if p in all_params] + sorted([p for p in all_params if p not in param_order])
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ["Parameter"] + list(model_summaries.keys())
            writer.writerow(header)
            
            # Write parameter rows
            for param_name in sorted_params:
                row = [param_name]
                
                for model_name, summaries in model_summaries.items():
                    if summaries:
                        best_summary = _get_best_summary(summaries)
                        if best_summary:
                            params = best_summary.get("best_params", {})
                            value = params.get(param_name, None)
                            
                            if value is not None:
                                # Add uncertainty if available from jackknife
                                uncertainty = None
                                if jackknife_summary and model_name in jackknife_summary.get("models", {}):
                                    jackknife_params = jackknife_summary["models"][model_name].get("parameters", {})
                                    if param_name in jackknife_params:
                                        uncertainty = jackknife_params[param_name].get("std")
                                
                                if uncertainty is not None:
                                    cell_value = f"{value:.6f} ± {uncertainty:.6f}"
                                else:
                                    cell_value = f"{value:.6f}"
                                
                                # Add units
                                units = self._get_parameter_units(param_name)
                                if units:
                                    cell_value += f" {units}"
                                
                                row.append(cell_value)
                            else:
                                row.append("")
                        else:
                            row.append("")
                    else:
                        row.append("")
                
                writer.writerow(row)
        
        return output_path

    def _format_parameter_latex(self, param_name: str) -> str:
        """Format parameter name for LaTeX."""
        latex_map = {
            "H0": "$H_0$",
            "Omega_m0": "$\\Omega_{m,0}$",
            "Omega_b0": "$\\Omega_{b,0}$",
            "Omega_k0": "$\\Omega_{k,0}$",
            "Rmax": "$R_{{\\rm max}}$",
            "alpha": "$\\alpha$"
        }
        return latex_map.get(param_name, param_name.replace("_", "\\_"))

    # Stub methods for remaining LaTeX exports (to be implemented)
    def _generate_dataset_table_latex(self, model_summaries: Dict[str, List[Dict[str, Any]]]) -> Path:
        """Generate LaTeX dataset statistics table (placeholder)."""
        output_path = self.run_dir / "paper_exports" / "datasets.tex"
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text("% LaTeX dataset table - to be implemented")
        return output_path
    
    def _generate_chi2_table_latex(self, model_summaries: Dict[str, List[Dict[str, Any]]], jackknife_summary: Dict[str, Any] | None = None) -> Path:
        """Generate LaTeX chi² statistics table (placeholder)."""
        output_path = self.run_dir / "paper_exports" / "chi2_statistics.tex"
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text("% LaTeX chi² table - to be implemented")
        return output_path
    
    def _generate_derived_table_latex(self, model_summaries: Dict[str, List[Dict[str, Any]]], jackknife_summary: Dict[str, Any] | None = None) -> Path:
        """Generate LaTeX derived quantities table (placeholder)."""
        output_path = self.run_dir / "paper_exports" / "derived_quantities.tex"
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text("% LaTeX derived quantities table - to be implemented")
        return output_path
    
    def _generate_jackknife_table_latex(self, jackknife_summary: Dict[str, Any]) -> Path:
        """Generate LaTeX jackknife results table (placeholder)."""
        output_path = self.run_dir / "paper_exports" / "jackknife_results.tex"
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text("% LaTeX jackknife table - to be implemented")
        return output_path
    
    def _generate_quantum_table_latex(self, quantum_state: Dict[str, Any]) -> Path:
        """Generate LaTeX quantum parameters table (placeholder)."""
        output_path = self.run_dir / "paper_exports" / "quantum_parameters.tex"
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text("% LaTeX quantum parameters table - to be implemented")
        return output_path
