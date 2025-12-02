"""
Proper data loader that handles model-specific jackknife data correctly
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from cosmos2.api.engine import _make_joint_evaluator
from cosmos2.data import registry
from cosmos2.plots.temperature_evolution import (
    create_temperature_evolution,
    create_thermal_fields_plot,
)
from cosmos2.science_runner.config import ScienceRunConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for cosmos2 science runs with proper model-specific data handling."""
    
    def __init__(self, run_directory: Path):
        """Initialize data loader for a specific run directory."""
        self.run_dir = Path(run_directory)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {self.run_dir}")

        self.logger = logging.getLogger(f"{__name__}.DataLoader")
        self.logger.info(f"Initialized DataLoader for: {self.run_dir}")
        self._science_config: ScienceRunConfig | None = None
        self._joint_config_path: Path | None = None
        self._model_evaluators: Dict[str, Callable[[Dict[str, float]], float]] = {}

    def __del__(self):
        if self._science_config:
            try:
                self._science_config.cleanup()
            except Exception:
                pass

    def _sanitize_filename_component(self, value: str | None, fallback: str) -> str:
        """Return a filesystem-safe component for plot filenames."""
        if not value:
            return fallback
        sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized or fallback

    def _figure_filename(self, group: str, plot_name: str, model_name: str | None = None) -> str:
        """Build a standardized filename as <model>_<group>_<plot>."""
        model_part = self._sanitize_filename_component(model_name, "run")
        group_part = self._sanitize_filename_component(group, "general")
        plot_part = self._sanitize_filename_component(plot_name, "plot")
        return f"{model_part}_{group_part}_{plot_part}"

    def _figure_path(self, group: str, plot_name: str, model_name: str | None = None) -> Path:
        """Get the path for a standardized plot and ensure the directory exists."""
        figures_dir = self.run_dir / "figures"
        figures_dir.mkdir(exist_ok=True, parents=True)
        filename = f"{self._figure_filename(group, plot_name, model_name)}.png"
        return figures_dir / filename
    
    def get_available_models(self) -> List[str]:
        """Get list of available models - DYNAMIC from config or scan directory."""
        # First try to get models from config
        config_files = [self.run_dir / "run_config.json", self.run_dir / "config_used.json"]
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    models = config.get("models", [])
                    if models:
                        self.logger.info(f"Found {len(models)} models from {config_file.name}: {models}")
                        return models
                except Exception as e:
                    self.logger.warning(f"Failed to load models from {config_file.name}: {e}")
        
        # Fallback 1: Check model_summaries.json for actual models run
        model_summaries_path = self.run_dir / "model_summaries.json"
        if model_summaries_path.exists():
            try:
                with open(model_summaries_path, 'r') as f:
                    model_summaries = json.load(f)
                models = list(model_summaries.keys())
                if models:
                    self.logger.info(f"Found {len(models)} models from model_summaries.json: {models}")
                    return sorted(models)
            except Exception as e:
                self.logger.warning(f"Failed to load models from model_summaries.json: {e}")

        # Fallback 2: scan for model directories (DYNAMIC)
        models = []
        for item in self.run_dir.iterdir():
            if item.is_dir() and item.name not in ['tables', 'figures', 'logs', '__pycache__']:
                # Check if this looks like a model directory (has results files)
                model_files = list(item.glob("*.json")) + list(item.glob("*.csv"))
                if model_files:
                    models.append(item.name)

        if models:
            self.logger.info(f"Found {len(models)} models by scanning directories: {models}")
            return sorted(models)
        
        # Final fallback: check model_summaries.json for model names
        summaries_file = self.run_dir / "model_summaries.json"
        if summaries_file.exists():
            try:
                with open(summaries_file, 'r') as f:
                    summaries = json.load(f)
                models = list(summaries.keys())
                if models:
                    self.logger.info(f"Found {len(models)} models from summaries: {models}")
                    return sorted(models)
            except Exception as e:
                self.logger.warning(f"Failed to load models from summaries: {e}")
        
        self.logger.warning("No models found - using empty list")
        return []
    
    def load_model_data(self, model_name: str) -> Dict[str, Any]:
        """Load model-specific data including proper jackknife results."""
        # First try to load from model_summaries.json (new format)
        model_summaries_file = self.run_dir / "model_summaries.json"
        if model_summaries_file.exists():
            try:
                with open(model_summaries_file, 'r') as f:
                    summaries = json.load(f)

                if model_name in summaries:
                    model_info = summaries[model_name]
                    self.logger.info(f"Loading {model_name} from model_summaries.json")

                    # Convert new format to expected format
                    best_fit = dict(model_info.get('best', {}))
                    parameters = best_fit.get('parameters', {})
                    chi2 = best_fit.get('chi_squared', best_fit.get('chi2', 0))
                    if 'chi_squared' not in best_fit:
                        best_fit['chi_squared'] = chi2
                    n_params = len(parameters)
                    n_data_points = self._count_data_points_from_fit(best_fit)

                    # Calculate AIC if not present (AIC = χ² + 2k)
                    if 'aic' not in best_fit and chi2 > 0:
                        best_fit['aic'] = chi2 + 2 * n_params
                    if 'bic' not in best_fit and chi2 > 0 and n_data_points > 0:
                        best_fit['bic'] = chi2 + n_params * math.log(n_data_points)

                    model_data = {
                        'name': model_name,
                        'best_fit': best_fit,
                        'chi2_breakdown': self._extract_chi2_breakdown(best_fit.get('fit_results', {})),
                        'parameters': parameters,
                        'jackknife_data': self._load_model_jackknife_new_format(model_name),  # Load jackknife from new format
                    }
                    model_predictions = self._extract_model_predictions(best_fit)
                    model_data['predictions'] = model_predictions
                    summary_dir = self.run_dir / model_name
                    self._attach_parameter_snapshot(summary_dir, model_data, best_fit)
                    self._enrich_model_data(model_name, model_data)
                    self._attach_dataset_prediction_plots(model_name, model_predictions)
                    return model_data
            except Exception as e:
                self.logger.warning(f"Failed to load from model_summaries.json: {e}")

        # Fallback to old format with individual model directories
        model_dir = self.run_dir / model_name
        if not model_dir.exists():
            self.logger.warning(f"No model directory found for {model_name}")
            return {}

        best_fit = self._load_best_fit(model_dir)
        if 'chi_squared' not in best_fit:
            best_fit['chi_squared'] = best_fit.get('chi2', float('inf'))
        parameters = best_fit.get('parameters', {})
        chi2 = best_fit.get('chi_squared', 0)
        n_params = len(parameters)
        n_data_points = self._count_data_points_from_fit(best_fit)

        # Calculate AIC if not present (AIC = χ² + 2k)
        if 'aic' not in best_fit and chi2 > 0:
            best_fit['aic'] = chi2 + 2 * n_params

        if 'bic' not in best_fit and chi2 > 0 and n_data_points > 0:
            best_fit['bic'] = chi2 + n_params * math.log(n_data_points)

        model_data = {
            'name': model_name,
            'best_fit': best_fit,
            'chi2_breakdown': self._load_chi2_breakdown(model_dir),
            'parameters': self._load_parameters(model_dir),
            'jackknife_data': self._load_model_jackknife(model_dir),
            'fits_data': self._load_fits_data(model_dir)
        }
        predictions = self._extract_model_predictions(best_fit)
        model_data['predictions'] = predictions
        self._attach_dataset_prediction_plots(model_name, predictions)
        dataset_points = self._collect_residual_dataset_points(best_fit)
        model_data['residual_plot'] = self._ensure_model_residual_plot(model_name, dataset_points)
        dataset_points = self._collect_residual_dataset_points(best_fit)
        model_data['residual_plot'] = self._ensure_model_residual_plot(model_name, dataset_points)
        self._attach_parameter_snapshot(model_dir, model_data, best_fit)
        self._enrich_model_data(model_name, model_data)
        
        return model_data
    
    def _load_best_fit(self, model_dir: Path) -> Dict[str, Any]:
        """Load best fit parameters."""
        best_fit_file = model_dir / "best_fit.json"
        if best_fit_file.exists():
            with open(best_fit_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_chi2_breakdown(self, model_dir: Path) -> Dict[str, Any]:
        """Load chi² breakdown by dataset."""
        chi2_file = model_dir / "chi2_breakdown.json"
        if chi2_file.exists():
            with open(chi2_file, 'r') as f:
                return json.load(f)
        return {}

    def _load_model_chi2_contributions(self, model_name: str) -> Dict[str, float]:
        """Retrieve dataset chi² contributions for any supported run layout."""
        summaries_file = self.run_dir / "model_summaries.json"
        if summaries_file.exists():
            try:
                with open(summaries_file, 'r') as f:
                    summaries = json.load(f)
                model_info = summaries.get(model_name)
                if isinstance(model_info, dict):
                    best_fit = model_info.get('best', {})
                    fit_results = best_fit.get('fit_results', {})
                    return self._extract_chi2_breakdown(fit_results)
            except Exception as e:
                self.logger.warning(f"Failed to load chi² contributions for {model_name}: {e}")

        model_dir = self.run_dir / model_name
        if model_dir.exists():
            breakdown = self._load_chi2_breakdown(model_dir)
            fits = breakdown.get('fits')
            if isinstance(fits, dict):
                return {ds: float(val) for ds, val in fits.items() if isinstance(val, (int, float))}
            return {ds: float(val) for ds, val in breakdown.items() if isinstance(val, (int, float))}
        return {}
    
    def _load_parameters(self, model_dir: Path) -> Dict[str, Any]:
        """Load model parameters."""
        params_file = model_dir / "parameters.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_model_jackknife(self, model_dir: Path) -> Dict[str, Any]:
        """Load model-specific jackknife data - handles new simple jackknife format."""
        jackknife_data = {
            'jackknife_available': False,
            'level1_available': False,
            'level2_available': False,
            'parameter_stability': {},
            'chi2_stability': {},
            'recommendations': []
        }

        # Look for jackknife files in run directory
        jackknife_files = [
            self.run_dir / "jackknife_results.json",  # New format
            self.run_dir / "jackknife_level1_results.json",  # Old format fallback
            self.run_dir / "jackknife_combined_results.json"  # Old format fallback
        ]

        for jk_file in jackknife_files:
            if jk_file.exists():
                try:
                    with open(jk_file, 'r') as f:
                        data = json.load(f)

                    # Check if this is the new multi-model format
                    if self._is_new_multi_model_jackknife_format(data):
                        model_jackknife = self._extract_new_multi_model_jackknife(data, model_dir.name)
                        if model_jackknife:
                            jackknife_data.update(model_jackknife)
                            jackknife_data['jackknife_available'] = True
                            self.logger.info(
                                f"Loaded new multi-model jackknife format for {model_dir.name} from {jk_file.name}"
                            )
                            break
                    elif self._is_new_jackknife_format(data):
                        model_jackknife = self._extract_new_jackknife_format(data)
                        if model_jackknife:
                            jackknife_data.update(model_jackknife)
                            jackknife_data['jackknife_available'] = True
                            self.logger.info(
                                f"Loaded new jackknife format for {model_dir.name} from {jk_file.name}"
                            )
                            break
                    else:
                        model_name = model_dir.name
                        model_jackknife = self._extract_model_jackknife(data, model_name)

                        if model_jackknife:
                            jackknife_data['level1_available'] = True
                            jackknife_data['jackknife_available'] = True
                            jackknife_data['parameter_stability'] = model_jackknife.get('parameter_shifts', {})
                            jackknife_data['chi2_stability'] = model_jackknife.get('chi2_changes', {})
                            jackknife_data['recommendations'] = self._generate_recommendations(model_jackknife)

                            stability_metrics = data.get('stability_metrics', {})
                            jackknife_data['stability_score'] = stability_metrics.get('overall_stability_score', 0.0)
                            jackknife_data['success_rate'] = data.get('success_rate', 0.0)
                            jackknife_data['n_draws'] = stability_metrics.get('n_draws_total', 0)
                            jackknife_data['dataset_impact'] = stability_metrics.get('dataset_impact', {})

                            self.logger.info(f"Loaded old jackknife format for {model_name} from {jk_file.name}")
                            break

                except Exception as e:
                    self.logger.error(f"Error loading jackknife data from {jk_file}: {e}")

        return jackknife_data

    def _count_data_points_from_fit(self, best_fit: Dict[str, Any]) -> int:
        """Estimate total number of observed points from fit outputs."""
        total = 0
        fit_outputs = best_fit.get('fit_outputs') or best_fit.get('fit_results') or {}
        if not isinstance(fit_outputs, dict):
            return 0
        for payload in fit_outputs.values():
            extras = payload.get('extras', {})
            observed = None
            if isinstance(extras, dict):
                observed = extras.get('observed')
            if observed is None:
                observed = payload.get('observed')
            total += self._sequence_length(observed)
        return total

    def _sequence_length(self, seq: Any) -> int:
        """Return length of a list-like object, safely."""
        if seq is None:
            return 0
        if hasattr(seq, "__len__"):
            try:
                return len(seq)
            except TypeError:
                return 0
        return 0

    def _is_new_jackknife_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is in the new simple jackknife format."""
        # New format has top-level keys like 'success_rate', 'stability_metrics', etc.
        new_format_keys = ['success_rate', 'stability_metrics', 'parameter_shifts', 'chi2_changes', 'draws']
        return any(key in data for key in new_format_keys)

    def _is_new_multi_model_jackknife_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is in the new multi-model jackknife format."""
        analysis_section = data.get('analysis', data)
        return (
            isinstance(analysis_section, dict) and
            'model_analyses' in analysis_section and
            'stability_metrics' in analysis_section
        )

    def _extract_new_jackknife_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract jackknife data from new simple format."""
        try:
            stability_metrics = data.get('stability_metrics', {})
            parameter_shifts = data.get('parameter_shifts', {})
            chi2_changes = data.get('chi2_changes', {})

            # Extract parameter stability
            parameter_stability = {}
            if 'parameter_stats' in parameter_shifts:
                for param, stats in parameter_shifts['parameter_stats'].items():
                    mean_val = stats.get('mean', 0)
                    std_val = stats.get('std', 0)
                    cv = stats.get('coefficient_of_variation', 0)
                    min_val = stats.get('min', 0)
                    max_val = stats.get('max', 0)

                    # Get confidence interval if available
                    ci = stats.get('confidence_interval', {})
                    ci_lower = ci.get('lower', mean_val)
                    ci_upper = ci.get('upper', mean_val)

                    # Determine stability level
                    if cv < 0.01:
                        stability_level = 'stable'
                    elif cv < 0.05:
                        stability_level = 'moderate'
                    else:
                        stability_level = 'unstable'

                    parameter_stability[param] = {
                        'mean': mean_val,
                        'std': std_val,
                        'cv': cv,
                        'stability': stability_level,
                        'range': max_val - min_val,
                        'confidence_interval': {
                            'lower': ci_lower,
                            'upper': ci_upper,
                            'level': ci.get('level', 0.95)
                        }
                    }

            # Extract chi² stability
            chi2_stability = {}
            if 'chi2_stats' in chi2_changes:
                chi2_stats = chi2_changes['chi2_stats']
                chi2_stability['overall'] = {
                    'mean_chi2': chi2_stats.get('mean', 0),
                    'std_chi2': chi2_stats.get('std', 0),
                    'range_chi2': chi2_stats.get('range', 0)
                }

            # Generate recommendations
            recommendations = self._generate_new_format_recommendations(data)

            return {
                'jackknife_available': True,
                'level1_available': True,
                'level2_available': False,  # New format is simplified
                'parameter_stability': parameter_stability,
                'chi2_stability': chi2_stability,
                'recommendations': recommendations,
                'stability_score': stability_metrics.get('overall_stability_score', 0.0),
                'success_rate': data.get('success_rate', 0.0),
                'n_draws': stability_metrics.get('n_draws_total', 0)
            }

        except Exception as e:
            self.logger.error(f"Error extracting new jackknife format: {e}")
            return {}

    def _extract_new_multi_model_jackknife(self, data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Extract jackknife data for a specific model from multi-model format."""
        try:
            analysis_section = data.get('analysis', data)
            stability_metrics = analysis_section.get('stability_metrics', {})
            model_analyses = analysis_section.get('model_analyses', {})
            model_comparison = stability_metrics.get('model_comparison', {})

            if model_name not in model_analyses:
                self.logger.warning(f"Model {model_name} not found in jackknife model analyses")
                return {}

            model_analysis = model_analyses[model_name]

            # Extract parameter stability
            parameter_stability = {}
            if 'parameter_stability' in model_analysis and 'parameter_stats' in model_analysis['parameter_stability']:
                for param, stats in model_analysis['parameter_stability']['parameter_stats'].items():
                    mean_val = stats.get('mean', 0)
                    std_val = stats.get('std', 0)
                    cv = stats.get('coefficient_of_variation', 0)
                    min_val = stats.get('min', 0)
                    max_val = stats.get('max', 0)

                    # Get confidence interval if available
                    ci = stats.get('confidence_interval', {})
                    ci_lower = ci.get('lower', mean_val)
                    ci_upper = ci.get('upper', mean_val)

                    # Determine stability level
                    if cv < 0.01:
                        stability_level = 'stable'
                    elif cv < 0.05:
                        stability_level = 'moderate'
                    else:
                        stability_level = 'unstable'

                    parameter_stability[param] = {
                        'mean': mean_val,
                        'std': std_val,
                        'cv': cv,
                        'stability': stability_level,
                        'range': max_val - min_val,
                        'confidence_interval': {
                            'lower': ci_lower,
                            'upper': ci_upper,
                            'level': ci.get('level', 0.95)
                        }
                    }

            # Extract chi² stability
            chi2_stability = {}
            if 'chi2_stability' in model_analysis:
                chi2_stats = model_analysis['chi2_stability']
                chi2_stability['overall'] = {
                    'mean_chi2': chi2_stats.get('mean', 0),
                    'std_chi2': chi2_stats.get('std', 0),
                    'range_chi2': chi2_stats.get('range', 0)
                }

            # Generate recommendations
            recommendations = self._generate_new_multi_model_recommendations({
                'stability_metrics': stability_metrics,
                'model_comparison': model_comparison,
                'model_analyses': model_analyses,
                'success_rate': analysis_section.get('success_rate', data.get('success_rate', 0.0)),
                'n_draws': analysis_section.get('n_draws_total', data.get('n_draws_total', 0))
            }, model_name)

            return {
                'jackknife_available': True,
                'level1_available': True,
                'level2_available': False,
                'parameter_stability': parameter_stability,
                'chi2_stability': chi2_stability,
                'recommendations': recommendations,
                'stability_score': model_analysis.get('overall_stability_score', 0.0),
                'success_rate': analysis_section.get('success_rate', data.get('success_rate', 0.0)),
                'n_draws': analysis_section.get('n_draws_total', data.get('n_draws_total', 0)),
                'model_comparison': model_comparison,
                'dataset_impact': stability_metrics.get('dataset_impact', {}),
                'stability_metrics': stability_metrics,
                'model_analyses': model_analyses
            }

        except Exception as e:
            self.logger.error(f"Error extracting multi-model jackknife format for {model_name}: {e}")
            return {}
    
    def _extract_model_jackknife(self, jackknife_data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Extract model-specific jackknife data from shared results."""
        # Check for model-specific results section
        if 'model_specific_results' in jackknife_data:
            return jackknife_data['model_specific_results'].get(model_name, {})
        
        # Check for results array with model data
        if 'results' in jackknife_data:
            # Extract parameter shifts for this model
            model_shifts = {}
            model_chi2 = {}
            
            for result in jackknife_data['results']:
                model_results = result.get('model_results', {})
                if model_name in model_results:
                    model_data = model_results[model_name]
                    params = model_data.get('parameters', {})
                    chi2 = model_data.get('chi_squared')
                    
                    if params and isinstance(params, dict):
                        for param, value in params.items():
                            if param not in model_shifts:
                                model_shifts[param] = []
                            model_shifts[param].append(value)
                    
                    if chi2 is not None and chi2 != float('inf'):
                        if 'chi2_values' not in model_chi2:
                            model_chi2['chi2_values'] = []
                        model_chi2['chi2_values'].append(chi2)
            
            # Calculate statistics
            parameter_shifts = {}
            for param, values in model_shifts.items():
                if values:
                    import numpy as np
                    valid_values = [v for v in values if v is not None and v != float('inf')]
                    if valid_values:
                        parameter_shifts[param] = {
                            'mean': np.mean(valid_values),
                            'std': np.std(valid_values),
                            'min': np.min(valid_values),
                            'max': np.max(valid_values)
                        }
            
            chi2_changes = {}
            if 'chi2_values' in model_chi2:
                values = model_chi2['chi2_values']
                if values:
                    import numpy as np
                    chi2_changes['overall'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'range': np.max(values) - np.min(values)
                    }
            
            return {
                'parameter_shifts': parameter_shifts,
                'chi2_changes': chi2_changes
            }
        
        return {}
    
    def _process_parameter_stability(self, jackknife_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process parameter stability from jackknife data."""
        stability = {}
        
        # Handle different possible data structures
        if 'parameter_shifts' in jackknife_data:
            param_data = jackknife_data['parameter_shifts']
        elif 'parameters' in jackknife_data:
            param_data = jackknife_data['parameters']
        else:
            return stability
        
        for param, stats in param_data.items():
            if isinstance(stats, dict):
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                cv = std_val / mean_val if mean_val != 0 else float('inf')
                
                # Determine stability
                if cv < 0.01:
                    stability_level = 'stable'
                elif cv < 0.05:
                    stability_level = 'moderate'
                else:
                    stability_level = 'unstable'
                
                stability[param] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv,
                    'stability': stability_level,
                    'range': stats.get('max', 0) - stats.get('min', 0)
                }
        
        return stability
    
    def _process_chi2_stability(self, jackknife_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process chi² stability from jackknife data."""
        stability = {}
        
        if 'chi2_changes' in jackknife_data:
            chi2_data = jackknife_data['chi2_changes']
            for dataset, stats in chi2_data.items():
                if isinstance(stats, dict):
                    stability[dataset] = {
                        'mean_chi2': stats.get('mean', 0),
                        'std_chi2': stats.get('std', 0),
                        'range_chi2': stats.get('range', 0)
                    }
        
        return stability
    
    def _generate_recommendations(self, jackknife_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on jackknife analysis (old format)."""
        recommendations = []

        n_draws = jackknife_data.get('n_draws', 0)
        if n_draws > 0:
            recommendations.append(f"Jackknife analysis based on {n_draws} data subsets")

        # Check parameter stability
        if 'parameter_shifts' in jackknife_data:
            stable_params = 0
            total_params = len(jackknife_data['parameter_shifts'])

            for param, stats in jackknife_data['parameter_shifts'].items():
                if isinstance(stats, dict):
                    mean_val = stats.get('mean', 0)
                    std_val = stats.get('std', 0)
                    cv = std_val / mean_val if mean_val != 0 else float('inf')
                    if cv < 0.01:
                        stable_params += 1

            if total_params > 0:
                stability_ratio = stable_params / total_params
                if stability_ratio >= 0.8:
                    recommendations.append("Parameters show good stability across jackknife samples")
                elif stability_ratio >= 0.5:
                    recommendations.append("Parameters show moderate stability - consider larger datasets")
                else:
                    recommendations.append("Parameters show poor stability - review data quality")

        return recommendations

    def _generate_new_format_recommendations(self, jackknife_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on new jackknife format."""
        recommendations = []
        stability_metrics = jackknife_data.get('stability_metrics', {})

        # Add basic info
        n_draws = stability_metrics.get('n_draws_total', 0)
        success_rate = jackknife_data.get('success_rate', 0.0)
        stability_score = stability_metrics.get('overall_stability_score', 0.0)

        if n_draws > 0:
            recommendations.append(f"Simple jackknife analysis: {n_draws} draws, {success_rate*100:.1f}% success rate")

        # Stability assessment
        if stability_score >= 0.9:
            recommendations.append("✅ Excellent stability - results are highly robust")
        elif stability_score >= 0.8:
            recommendations.append("👍 Very good stability - results are well-validated")
        elif stability_score >= 0.7:
            recommendations.append("🟡 Good stability - results are reasonably robust")
        elif stability_score >= 0.6:
            recommendations.append("⚠️ Moderate stability - consider increasing dataset size")
        elif stability_score >= 0.5:
            recommendations.append("⚠️ Fair stability - review data quality and model assumptions")
        else:
            recommendations.append("❌ Poor stability - significant concerns about result reliability")

        # Parameter-specific recommendations
        parameter_analysis = jackknife_data.get('parameter_shifts', {})
        if 'parameter_stats' in parameter_analysis:
            param_stats = parameter_analysis['parameter_stats']
            unstable_params = []

            for param, stats in param_stats.items():
                cv = stats.get('coefficient_of_variation', 0)
                if cv > 0.05:  # High variability
                    unstable_params.append(f"{param} (CV: {cv:.3f})")

            if unstable_params:
                recommendations.append(f"High variability in parameters: {', '.join(unstable_params[:3])}")

        # χ² stability
        chi2_analysis = jackknife_data.get('chi2_changes', {})
        if 'chi2_stats' in chi2_analysis:
            chi2_stats = chi2_analysis['chi2_stats']
            relative_std = chi2_stats.get('relative_std', 0)
            if relative_std > 0.1:
                recommendations.append(f"χ² shows high variability (σ/μ = {relative_std:.3f}) - investigate fit quality")

        return recommendations

    def _generate_new_multi_model_recommendations(self, jackknife_data: Dict[str, Any], model_name: str) -> List[str]:
        """Generate recommendations for multi-model jackknife analysis."""
        recommendations = []
        stability_metrics = jackknife_data.get('stability_metrics', {})
        model_comparison = jackknife_data.get('model_comparison', {})
        model_analyses = jackknife_data.get('model_analyses', {})

        # Basic info
        success_rate = jackknife_data.get('success_rate', 0.0)
        n_draws = stability_metrics.get('n_draws_total', 0)

        if n_draws > 0:
            recommendations.append(f"Multi-model jackknife analysis: {n_draws} draws, {success_rate*100:.1f}% success rate")

        # Model-specific stability
        if model_name in model_analyses:
            model_analysis = model_analyses[model_name]
            stability_score = model_analysis.get('overall_stability_score', 0.0)

            if stability_score >= 0.9:
                recommendations.append(f"✅ Model {model_name}: Excellent stability")
            elif stability_score >= 0.8:
                recommendations.append(f"👍 Model {model_name}: Very good stability")
            elif stability_score >= 0.7:
                recommendations.append(f"🟡 Model {model_name}: Good stability")
            elif stability_score >= 0.6:
                recommendations.append(f"⚠️ Model {model_name}: Moderate stability - consider larger datasets")
            else:
                recommendations.append(f"❌ Model {model_name}: Poor stability - significant concerns")

        # Model comparison stability
        preference_stability = model_comparison.get('preference_stability', 0.0)
        if preference_stability < 0.8:
            recommendations.append(f"⚠️ Model preference changes in {model_comparison.get('preference_changes', 0)}/{model_comparison.get('total_draws', 0)} draws")

        dominant_model = model_comparison.get('dominant_model')
        if dominant_model:
            freq = model_comparison.get('dominant_model_frequency', 0.0)
            recommendations.append(f"Dominant model: {dominant_model} ({freq*100:.1f}% of draws)")

        # Dataset impact
        dataset_impact = stability_metrics.get('dataset_impact', {})
        if dataset_impact:
            high_impact_datasets = [name for name, impact in dataset_impact.items()
                                   if impact.get('impact_frequency', 0) > 0.5]
            if high_impact_datasets:
                recommendations.append(f"Datasets frequently affected: {', '.join(high_impact_datasets)}")

        return recommendations

    def _load_fits_data(self, model_dir: Path) -> Dict[str, Any]:
        """Load fits data directory contents."""
        fits_dir = model_dir / "fits"
        if fits_dir.exists():
            fits_data = {}
            for fits_file in fits_dir.glob("*.json"):
                try:
                    with open(fits_file, 'r') as f:
                        fits_data[fits_file.stem] = json.load(f)
                except Exception as e:
                    self.logger.error(f"Error loading fits file {fits_file}: {e}")
            return fits_data
        return {}
    
    def load_run_metadata(self) -> Dict[str, Any]:
        """Load run-level metadata."""
        metadata = {}
        
        # Load run metadata
        run_meta_file = self.run_dir / "run_meta.json"
        if run_meta_file.exists():
            with open(run_meta_file, 'r') as f:
                metadata['run_meta'] = json.load(f)
        
        # Load configuration
        config_file = self.run_dir / "config_used.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                metadata['config'] = json.load(f)
        
        return metadata
    
    def get_figures(self) -> List[Dict[str, Any]]:
        """Get list of available figures."""
        figures = []
        self._ensure_jackknife_plot()
        self._ensure_model_jackknife_plots()
        self._ensure_h0_convergence_plot()
        self._ensure_dataset_chi2_contributions_plot()
        self._ensure_residuals_by_redshift_plot()
        self._ensure_temperature_evolution_plot()
        self._ensure_thermal_fields_plot()

        figures_dir = self.run_dir / "figures"
        if figures_dir.exists():
            for fig_file in figures_dir.glob("*.png"):
                stem = fig_file.stem
                if stem.endswith("_datasets_residuals_by_redshift"):
                    continue
                if stem.endswith("_datasets_residuals"):
                    continue
                figures.append({
                    'name': fig_file.stem,
                    'file_path': str(fig_file),
                    'type': 'image'
                })

        return figures

    def _ensure_jackknife_plot(self) -> None:
        """Create a jackknife chi² plot if jackknife results are available."""
        jk_file = self.run_dir / "jackknife_results.json"
        if not jk_file.exists():
            return
        plot_path = self._figure_path("jackknife", "chi2")
        if plot_path.exists():
            try:
                plot_path.unlink()
            except Exception:
                pass

        try:
            with open(jk_file, 'r') as f:
                data = json.load(f)
        except Exception:
            return

        draws = data.get('draws', [])
        if not draws:
            return

        models = list(draws[0].get('jackknife_models', {}).keys())
        draw_indices = list(range(1, len(draws) + 1))
        baseline_chi2 = {}
        for model in models:
            best_fit_file = self.run_dir / model / "best_fit.json"
            if best_fit_file.exists():
                try:
                    with open(best_fit_file, 'r') as bf:
                        best = json.load(bf)
                        baseline_chi2[model] = best.get('chi_squared', best.get('chi2'))
                        continue
                except Exception:
                    pass
            # fallback to original_models first draw
            baseline_chi2[model] = draws[0].get('original_models', {}).get(model, {}).get('chi_squared')

        draw_chi2 = {model: [] for model in models}
        for draw in draws:
            for model in models:
                chi2_val = draw.get('jackknife_models', {}).get(model, {}).get('chi_squared')
                if chi2_val is None:
                    chi2_val = draw.get('original_models', {}).get(model, {}).get('chi_squared')
                if chi2_val is not None:
                    draw_chi2[model].append(chi2_val)

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, model in enumerate(models):
            color = colors[idx % len(colors)]
            baseline = baseline_chi2.get(model)
            if baseline is not None:
                ax.hlines(baseline, draw_indices[0], draw_indices[-1], colors=color, linestyles="--", label=f"{model.upper()} baseline")
            scatter_y = draw_chi2.get(model, [])
            if scatter_y:
                ax.scatter(draw_indices[:len(scatter_y)], scatter_y, color=color, alpha=0.6, label=f"{model.upper()} draws")

        ax.set_title("Jackknife χ² comparison")
        ax.set_xlabel("Draw index")
        ax.set_ylabel("χ²")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.7)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)

    def _ensure_model_jackknife_plots(self) -> None:
        """Create jackknife chi² plots for each model individually."""
        jk_file = self.run_dir / "jackknife_results.json"
        if not jk_file.exists():
            return

        try:
            with open(jk_file, 'r') as f:
                data = json.load(f)
        except Exception:
            return

        draws = data.get('draws', [])
        if not draws:
            return

        models = list(draws[0].get('jackknife_models', {}).keys())
        for model in models:
            plot_path = self._figure_path("jackknife", "chi2", model)
            if plot_path.exists():
                try:
                    plot_path.unlink()
                except Exception:
                    pass

            draw_indices = list(range(1, len(draws) + 1))
            chi2_values = []
            for draw in draws:
                chi2_val = draw.get('jackknife_models', {}).get(model, {}).get('chi_squared')
                if chi2_val is None:
                    chi2_val = draw.get('original_models', {}).get(model, {}).get('chi_squared')
                chi2_values.append(chi2_val if chi2_val is not None else np.nan)

            baseline = self._load_model_baseline_chi2(model)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                draw_indices,
                chi2_values,
                marker="o",
                linestyle="-",
                color="tab:blue",
                label=f"{model.upper()} draws",
            )
            if baseline is not None:
                ax.hlines(
                    baseline,
                    draw_indices[0],
                    draw_indices[-1],
                    colors="tab:orange",
                    linestyles="--",
                    label=f"{model.upper()} best-fit",
                )

            ax.set_title(f"{model.upper()} jackknife χ²")
            ax.set_xlabel("Draw index")
            ax.set_ylabel("χ²")
            ax.grid(True, linestyle=":", alpha=0.7)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)

    def _load_model_baseline_chi2(self, model: str) -> float | None:
        """Load the baseline χ² for the given model."""
        best_fit_file = self.run_dir / model / "best_fit.json"
        if best_fit_file.exists():
            try:
                with open(best_fit_file, 'r') as f:
                    data = json.load(f)
                    return data.get("chi_squared", data.get("chi2"))
            except Exception:
                pass
        return None

    def _ensure_h0_convergence_plot(self) -> None:
        """Plot H₀ values from each jackknife draw to monitor convergence."""
        jk_file = self.run_dir / "jackknife_results.json"
        if not jk_file.exists():
            return
        plot_path = self._figure_path("jackknife", "h0_convergence")

        try:
            with open(jk_file, 'r') as f:
                data = json.load(f)
        except Exception:
            return

        draws = data.get('draws', [])
        if not draws:
            return

        models = list(draws[0].get('jackknife_models', {}).keys())
        if not models:
            return

        baseline_models = data.get('baseline_models', {})
        draw_indices = [draw.get('draw_index', idx) for idx, draw in enumerate(draws)]

        draw_points = {model: {'x': [], 'y': []} for model in models}
        for idx, draw in enumerate(draws):
            draw_idx = draw.get('draw_index', idx)
            jack_models = draw.get('jackknife_models', {})
            for model in models:
                params = jack_models.get(model, {}).get('parameters', {})
                h0 = params.get('H0')
                if h0 is None:
                    continue
                draw_points[model]['x'].append(draw_idx)
                draw_points[model]['y'].append(h0)

        if not any(draw_points[model]['x'] for model in models):
            return

        x_min = min(draw_indices)
        x_max = max(draw_indices)
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, model in enumerate(models):
            color = colors[idx % len(colors)]
            baseline = baseline_models.get(model, {}).get('parameters', {}).get('H0')
            if baseline is not None:
                ax.hlines(
                    baseline,
                    x_min,
                    x_max,
                    colors=color,
                    linestyles="--",
                    linewidth=1.2,
                    label=f"{model.upper()} baseline"
                )
            xs = draw_points[model]['x']
            ys = draw_points[model]['y']
            if xs:
                ax.scatter(xs, ys, color=color, alpha=0.75, label=f"{model.upper()} draws", edgecolors='none')

        ax.set_title("H₀ convergence across jackknife draws")
        ax.set_xlabel("Draw index")
        ax.set_ylabel("H₀ (km/s/Mpc)")
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)

    def _ensure_dataset_chi2_contributions_plot(self) -> None:
        """Create a bar chart showing each dataset's χ² share for every model."""
        models = self.get_available_models()
        if not models:
            return

        plot_path = self._figure_path("datasets", "chi2_contributions")

        model_contributions: Dict[str, Dict[str, float]] = {}
        dataset_totals: Dict[str, float] = {}
        for model in models:
            breakdown = self._load_model_chi2_contributions(model)
            if not breakdown:
                continue
            model_contributions[model] = breakdown
            for dataset, value in breakdown.items():
                dataset_totals[dataset] = dataset_totals.get(dataset, 0.0) + value

        if not model_contributions:
            return

        dataset_order = sorted(dataset_totals.keys(), key=lambda name: -dataset_totals[name])
        if not dataset_order:
            return

        n_models = len(model_contributions)
        x = np.arange(len(dataset_order))
        bar_width = 0.8 / max(n_models, 1)
        fig, ax = plt.subplots(figsize=(max(9, len(dataset_order) * 0.7), 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for idx, model in enumerate(model_contributions):
            values = [float(model_contributions[model].get(ds, 0.0)) for ds in dataset_order]
            offsets = x + (idx - (n_models - 1) / 2) * bar_width
            ax.bar(offsets, values, width=bar_width * 0.9, color=colors[idx % len(colors)], label=model.upper())

        ax.set_xticks(x)
        ax.set_xticklabels(dataset_order, rotation=45, ha="right")
        ax.set_ylabel("χ² contribution")
        ax.set_title("Dataset χ² contributions")
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", alpha=0.7)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)

    def _ensure_temperature_evolution_plot(self) -> None:
        """Render a LUT-based temperature evolution figure if missing."""
        plot_path = self._figure_path("temperature", "evolution", "pbuf")
        if plot_path.exists():
            try:
                plot_path.unlink()
            except Exception:
                pass
        try:
            create_temperature_evolution(output_path=plot_path, dpi=150)
        except Exception as exc:
            self.logger.warning("Unable to render temperature evolution plot: %s", exc)

    def _ensure_thermal_fields_plot(self) -> None:
        """Render ε(T) and α(T) plots derived from the thermal table."""
        plot_path = self._figure_path("temperature", "thermal_fields", "pbuf")
        if plot_path.exists():
            try:
                plot_path.unlink()
            except Exception:
                pass
        try:
            create_thermal_fields_plot(output_path=plot_path, dpi=150)
        except Exception as exc:
            self.logger.warning("Unable to render thermal fields plot: %s", exc)

    def _ensure_residuals_by_redshift_plot(self) -> None:
        """Plot residuals versus redshift for the best-fit model."""
        plot_path = self._figure_path("datasets", "residuals_by_redshift")

        models = self.get_available_models()
        if not models:
            return

        best_model = None
        best_chi2 = float("inf")
        best_fit_data: Dict[str, Any] = {}
        for model in models:
            model_data = self.load_model_data(model)
            best_fit = model_data.get('best_fit', {})
            chi2_value = best_fit.get('chi_squared') or best_fit.get('chi2')
            if chi2_value is None:
                continue
            if chi2_value < best_chi2:
                best_chi2 = chi2_value
                best_model = model
                best_fit_data = best_fit

        if not best_fit_data:
            return

        dataset_points = self._collect_residual_dataset_points(best_fit_data)
        if not dataset_points:
            return

        title = f"Residuals by redshift ({best_model.upper() if best_model else 'best model'})"
        self._render_residuals_figure(dataset_points, title, plot_path)

    def _collect_residual_dataset_points(self, best_fit: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather dataset metadata needed for residual plotting."""
        dataset_sources: Dict[str, Dict[str, Any]] = {}
        fit_results = best_fit.get('fit_results', {})
        fit_outputs = best_fit.get('fit_outputs', {})
        for dataset, payload in fit_results.items():
            dataset_sources[dataset] = payload
        for dataset, payload in fit_outputs.items():
            if dataset not in dataset_sources:
                dataset_sources[dataset] = payload

        dataset_points: List[Dict[str, Any]] = []
        for dataset_name, payload in dataset_sources.items():
            extras = payload.get('extras', {})
            if not extras:
                continue
            residuals = extras.get('residuals')
            if not residuals:
                continue
            try:
                dataset_payload = registry.get_dataset(dataset_name)
            except Exception:
                continue

            z_values = np.asarray(dataset_payload.get('z', []), dtype=float)
            residual_array = np.asarray(residuals, dtype=float)
            if z_values.size != residual_array.size or z_values.size == 0:
                continue

            cov = dataset_payload.get('cov')
            if isinstance(cov, np.ndarray) and cov.ndim == 2:
                errors = np.sqrt(np.clip(np.diag(cov), 0, None))
            else:
                errors = np.ones_like(residual_array)

            if errors.shape != residual_array.shape:
                errors = np.ones_like(residual_array)

            with np.errstate(divide="ignore", invalid="ignore"):
                normalized = np.divide(residual_array, errors, out=np.zeros_like(residual_array), where=errors > 0)

            label = extras.get('dataset', {}).get('name') or dataset_name.upper()
            dataset_points.append({
                "label": label,
                "z": z_values,
                "residuals": residual_array,
                "normalized": normalized
            })

        return dataset_points

    def _render_residuals_figure(self, dataset_points: List[Dict[str, Any]], title: str, plot_path: Path) -> bool:
        """Render a residual plot with log(z) and dataset panels."""
        if not dataset_points:
            return False

        figures_dir = plot_path.parent
        figures_dir.mkdir(exist_ok=True, parents=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 2]})
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        positive_z_arrays = [dp["z"][dp["z"] > 0] for dp in dataset_points if np.any(dp["z"] > 0)]
        if positive_z_arrays:
            positive_z = np.concatenate(positive_z_arrays)
            min_positive_z = max(1e-3, float(np.min(positive_z)))
        else:
            min_positive_z = 1e-3
        max_z = float(max(np.max(dp["z"]) for dp in dataset_points))
        if max_z <= 0:
            max_z = min_positive_z * 10

        for idx, dp in enumerate(dataset_points):
            label = dp["label"]
            color = color_cycle[idx % len(color_cycle)]
            ax1.scatter(dp["z"], dp["normalized"], s=15, alpha=0.65, color=color, label=label, edgecolors="none")

        ax1.set_xscale("log")
        ax1.set_xlim(min_positive_z, max_z * 1.05)
        ax1.set_ylim(-5.5, 5.5)
        ax1.axhline(0, color="black", linestyle="--", linewidth=1)
        ax1.set_xlabel("Redshift (z)")
        ax1.set_ylabel("Residual / σ")
        ax1.set_title(title)
        ax1.legend(loc="upper right", fontsize="small")
        ax1.grid(True, linestyle=":", alpha=0.6)

        positions = np.arange(len(dataset_points))
        for idx, dp in enumerate(dataset_points):
            values = dp["normalized"]
            if values.size == 0:
                continue
            base = idx
            jitter = ((np.arange(len(values)) % 5) - 2) * 0.02
            x_vals = base + jitter
            color = color_cycle[idx % len(color_cycle)]
            ax2.scatter(x_vals, values, s=12, color=color, alpha=0.7, label=dp["label"], edgecolors="none")

        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_xticks(positions)
        ax2.set_xticklabels([dp["label"] for dp in dataset_points], rotation=45, ha="right")
        ax2.set_ylabel("Residual / σ")
        ax2.set_title("Residuals per dataset")
        ax2.set_ylim(-5.5, 5.5)
        ax2.grid(True, linestyle=":", alpha=0.6)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        return True

    def _ensure_model_residual_plot(self, model_name: str, dataset_points: List[Dict[str, Any]]) -> Optional[str]:
        """Generate the residual plot for a single model and return its path."""
        if not dataset_points:
            return None

        plot_path = self._figure_path("datasets", "residuals", model_name)
        title = f"Residuals by redshift ({model_name.upper()})"
        self._render_residuals_figure(dataset_points, title, plot_path)
        return str(plot_path)

    def _extract_model_predictions(self, best_fit: Dict[str, Any]) -> Dict[str, Any]:
        """Extract prediction metadata for each dataset in the fit."""
        predictions: Dict[str, Any] = {}
        fit_outputs = best_fit.get('fit_outputs', {})
        for dataset_key, payload in fit_outputs.items():
            extras = payload.get('extras', {})
            dataset_info = extras.get('dataset', {})
            dataset_name = dataset_info.get('name') if dataset_info.get('name') else dataset_key.upper()

            observed = extras.get('observed', [])
            preds = extras.get('predictions', [])
            residuals = extras.get('residuals', [])

            predictions[dataset_key] = {
                'dataset_name': dataset_name,
                'dataset_meta': dataset_info.get('meta', {}),
                'observed_summary': self._array_summary(observed),
                'prediction_summary': self._array_summary(preds),
                'residual_summary': self._array_summary(residuals),
                'observed': observed,
                'predictions': preds,
                'residuals': residuals,
                'dv_over_rd': extras.get('DV_over_rd_model', []),
                'rd': extras.get('rd'),
                'extras': extras,
                'prediction_plot': None,
            }

        return predictions

    def _extract_derived_parameters(self, best_fit: Dict[str, Any]) -> Dict[str, float]:
        """Extract scalar derived parameters from best-fit predictions."""
        derived: Dict[str, float] = {}
        raw_predictions = best_fit.get('predictions', {})
        if isinstance(raw_predictions, dict):
            for key, value in raw_predictions.items():
                if isinstance(value, (int, float)):
                    derived[key] = float(value)
        return derived

    def _array_summary(self, values: Any, limit: int = 30) -> Dict[str, Any]:
        """Return a small preview of array-like values."""
        if values is None:
            values = []
        arr = np.asarray(values, dtype=float) if len(values) > 0 else np.array([])
        total = int(arr.size)
        preview_list = arr.tolist()[:limit]
        if total > limit:
            preview_list.append("…")
        return {
            'count': total,
            'preview': preview_list
        }

    def _attach_dataset_prediction_plots(self, model_name: str, predictions: Dict[str, Any]) -> None:
        """Generate and attach prediction comparison plots for each dataset."""
        if not predictions:
            return

        for dataset_key, info in predictions.items():
            plot_path = self._figure_path("datasets", str(dataset_key), model_name)
            try:
                rendered = self._render_prediction_comparison_figure(model_name, dataset_key, info, plot_path)
            except Exception as exc:
                self.logger.warning(f"Prediction plot failed for {model_name}/{dataset_key}: {exc}")
                rendered = False
            info["prediction_plot"] = str(plot_path) if rendered else None

    def _render_prediction_comparison_figure(
        self,
        model_name: str,
        dataset_key: str,
        dataset_info: Dict[str, Any],
        plot_path: Path,
    ) -> bool:
        """Render a small figure comparing predictions and observations for a dataset."""
        observed = np.asarray(dataset_info.get("observed", []) or [], dtype=float)
        predictions = np.asarray(dataset_info.get("predictions", []) or [], dtype=float)
        residuals = np.asarray(dataset_info.get("residuals", []) or [], dtype=float)
        dv_model = np.asarray(dataset_info.get("dv_over_rd", []) or [], dtype=float)

        if not any(arr.size for arr in (observed, predictions, dv_model)):
            return False

        n_points = max(observed.size, predictions.size, dv_model.size, residuals.size)
        n_points = max(int(n_points), 1)

        axis = self._determine_prediction_axis(dataset_info.get("extras", {}), n_points)

        figures_dir = plot_path.parent
        figures_dir.mkdir(exist_ok=True, parents=True)

        fig, (ax_main, ax_resid) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle(f"{dataset_info.get('dataset_name', dataset_key)} predictions ({model_name.upper()})")

        def plot_series(ax, values, label, **kwargs):
            if values.size == 0:
                return
            ax.plot(axis[: values.size], values, label=label, **kwargs)

        plot_series(ax_main, observed, "Observed", linestyle="-", marker="o")
        plot_series(ax_main, predictions, "Predictions", linestyle="--", marker="s")
        plot_series(ax_main, dv_model, "DV/rd model", linestyle=":", marker="d")

        ax_main.set_ylabel("Value")
        ax_main.grid(True, linestyle=":", alpha=0.6)
        ax_main.legend(loc="upper right", fontsize="small")

        if residuals.size > 0:
            ax_resid.plot(axis[: residuals.size], residuals, "o-", label="Residuals")
            ax_resid.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax_resid.set_ylabel("Residual")
            ax_resid.set_xlabel("Point index")
            ax_resid.grid(True, linestyle=":", alpha=0.6)
        else:
            ax_resid.set_visible(False)

        rd_value = dataset_info.get("rd")
        if rd_value not in (None, ""):
            rd_text = f"rd {float(rd_value):.3f}" if isinstance(rd_value, (int, float)) else f"rd {rd_value}"
            fig.text(0.02, 0.02, rd_text, fontsize=8, color="#475569")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        return True

    def _determine_prediction_axis(self, extras: Dict[str, Any], length: int) -> np.ndarray:
        """Pick the most sensible x-axis values for dataset plots."""
        if length <= 0:
            return np.arange(1)

        axis_keys = ["z", "redshift", "redshifts", "z_values", "x_values"]
        for key in axis_keys:
            values = extras.get(key)
            if isinstance(values, (list, tuple)):
                arr = np.asarray(values, dtype=float)
                if arr.size == length:
                    return arr

        return np.arange(1, length + 1)

    def _load_model_jackknife_new_format(self, model_name: str) -> Dict[str, Any]:
        """Load jackknife data from the new multi-model format."""
        jackknife_data = {
            'jackknife_available': False,
            'level1_available': False,
            'parameter_stability': {},
            'chi2_stability': {},
            'recommendations': [],
            'stability_score': 0.0,
            'success_rate': 0.0,
            'n_draws': 0,
            'dataset_impact': {}
        }

        jackknife_file = self.run_dir / "jackknife_results.json"
        if jackknife_file.exists():
            try:
                with open(jackknife_file, 'r') as f:
                    data = json.load(f)

                # Extract overall metrics
                model_analysis = data.get('model_analyses', {}).get(model_name, {})
                jackknife_data.update({
                    'jackknife_available': True,
                    'level1_available': True,
                    'stability_score': model_analysis.get('overall_stability_score', 0.0),
                    'success_rate': data.get('success_rate', 0.0),
                    'n_draws': data.get('stability_metrics', {}).get('n_draws_total', 0),
                    'dataset_impact': data.get('stability_metrics', {}).get('dataset_impact', {})
                })

                # Extract model-specific data
                model_analyses = data.get('model_analyses', {})
                if model_name in model_analyses:
                    model_analysis = model_analyses[model_name]
                    param_stability = {}

                    # Convert parameter stability format
                    param_stats = model_analysis.get('parameter_stability', {}).get('parameter_stats', {})
                    for param_name, stats in param_stats.items():
                        cv = stats.get('coefficient_of_variation', 0)
                        param_stability[param_name] = {
                            'mean': stats.get('mean', 0),
                            'std': stats.get('std', 0),
                            'cv': cv,
                            'stability': 'stable' if cv < 0.1 else 'unstable'
                        }

                    jackknife_data['parameter_stability'] = param_stability

                    # Add chi2 stability if available
                    chi2_stats = model_analysis.get('chi2_stability', {})
                    if chi2_stats:
                        jackknife_data['chi2_stability'] = chi2_stats

                # Generate recommendations
                jackknife_data['recommendations'] = self._generate_new_format_recommendations(jackknife_data)

                self.logger.info(f"Loaded jackknife data for {model_name} from {jackknife_file.name}")

            except Exception as e:
                self.logger.error(f"Error loading jackknife data: {e}")

        return jackknife_data

    def _generate_new_format_recommendations(self, jackknife_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on new jackknife format data."""
        recommendations = []

        if not jackknife_data.get('jackknife_available'):
            return recommendations

        stability_score = jackknife_data.get('stability_score', 0)
        success_rate = jackknife_data.get('success_rate', 0)
        n_draws = jackknife_data.get('n_draws', 0)

        if stability_score > 0.8:
            recommendations.append("✅ Excellent parameter stability across jackknife draws")
        elif stability_score > 0.6:
            recommendations.append("⚠️ Moderate parameter stability - consider additional data")
        else:
            recommendations.append("❌ Poor parameter stability - investigate data quality or model issues")

        if success_rate < 0.8:
            recommendations.append(f"⚠️ Low success rate ({success_rate:.1%}) - {n_draws} draws completed")

        # Check parameter stability
        param_stability = jackknife_data.get('parameter_stability', {})
        unstable_params = []
        for param, stats in param_stability.items():
            if stats.get('stability') == 'unstable':
                unstable_params.append(param)

        if unstable_params:
            recommendations.append(f"⚠️ Unstable parameters: {', '.join(unstable_params)}")

        return recommendations

    def _extract_chi2_breakdown(self, fit_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract chi2 breakdown from fit_results."""
        chi2_breakdown = {}
        for dataset, results in fit_results.items():
            if isinstance(results, dict):
                # Use weighted_chi2 if available, otherwise chi2
                chi2_val = results.get('weighted_chi2') or results.get('chi2')
                if chi2_val is not None:
                    chi2_breakdown[dataset] = float(chi2_val)
        return chi2_breakdown

    def _attach_parameter_snapshot(
        self,
        model_dir: Path,
        model_data: Dict[str, Any],
        best_fit: Dict[str, Any],
    ) -> None:
        """Ensure the model data exposes the parameter snapshot (if present)."""
        snapshot = best_fit.get("parameter_snapshot")
        if not snapshot:
            params_payload = model_data.get("parameters") or {}
            if isinstance(params_payload, dict):
                snapshot = params_payload.get("parameter_snapshot")

        if isinstance(snapshot, dict):
            model_data["parameter_snapshot"] = snapshot
            derived = snapshot.get("derived")
            if isinstance(derived, dict):
                model_data["derived_parameters"] = derived
                return

        model_data["derived_parameters"] = self._extract_derived_parameters(best_fit)

    def _enrich_model_data(self, model_name: str, model_data: Dict[str, Any]) -> None:
        best_fit = model_data.get("best_fit", {})
        engine_result = best_fit.get("engine_result") or {}
        performance_meta = engine_result.get("performance") or {}
        iterations = 0
        results = engine_result.get("results")
        if isinstance(results, list):
            iterations = len(results)

        gradient_norm, hessian_diag = self._approximate_gradient_and_hessian(model_name, best_fit.get("parameters", {}))

        stop_reason = performance_meta.get("stop_reason")
        if not stop_reason:
            batches = performance_meta.get("batch_iterations")
            stop_reason = f"Completed {batches} batches" if batches else "Iteration budget reached"

        stop_tolerance = performance_meta.get("stop_tolerance") or self._extract_tolerance_from_config()

        convergence = {
            "iterations": iterations,
            "gradient_norm": gradient_norm,
            "hessian_diag": hessian_diag,
            "stop_reason": stop_reason,
            "stop_tolerance": stop_tolerance,
            "evaluations": performance_meta.get("evaluations") if isinstance(performance_meta.get("evaluations"), (int, float)) else iterations,
            "performance": performance_meta or None,
        }
        model_data["convergence"] = convergence
        if performance_meta:
            model_data["performance"] = performance_meta

    def _ensure_science_config(self) -> ScienceRunConfig | None:
        if self._science_config is not None:
            return self._science_config
        config_path = self.run_dir / "config_used.json"
        if not config_path.exists():
            return None
        try:
            self._science_config = ScienceRunConfig.from_path(config_path)
            return self._science_config
        except Exception as exc:
            self.logger.warning("Unable to load science config for convergence metrics: %s", exc)
            self._science_config = None
            return None

    def _get_joint_config_path(self) -> Path | None:
        if self._joint_config_path and self._joint_config_path.exists():
            return self._joint_config_path
        config = self._ensure_science_config()
        if not config:
            return None
        try:
            path = config.get_joint_config_path()
            self._joint_config_path = path
            return path
        except Exception as exc:
            self.logger.debug("Could not materialize joint config path: %s", exc)
            return None

    def _get_model_evaluator(self, model_name: str) -> Callable[[Dict[str, float]], float] | None:
        if model_name in self._model_evaluators:
            return self._model_evaluators[model_name]
        joint_path = self._get_joint_config_path()
        if not joint_path:
            return None
        try:
            evaluator = _make_joint_evaluator(model_name, joint_path)
            self._model_evaluators[model_name] = evaluator
            return evaluator
        except Exception as exc:
            self.logger.warning("Unable to build chi² evaluator for %s: %s", model_name, exc)
            return None

    def _approximate_gradient_and_hessian(
        self, model_name: str, parameters: Dict[str, Any]
    ) -> tuple[float | None, Dict[str, float] | None]:
        evaluator = self._get_model_evaluator(model_name)
        if evaluator is None or not parameters:
            return None, None

        param_items = sorted(parameters.items())
        if not param_items:
            return None, None

        keys = [name for name, _ in param_items]
        base_values = np.array([float(value) for _, value in param_items], dtype=float)
        try:
            center = float(evaluator({key: float(val) for key, val in zip(keys, base_values.tolist())}))
        except Exception as exc:
            self.logger.debug("Center evaluation failed for %s: %s", model_name, exc)
            return None, None

        grads: Dict[str, float] = {}
        hess: Dict[str, float] = {}
        for idx, key in enumerate(keys):
            delta = max(1e-6, abs(base_values[idx]) * 1e-6)
            plus_vals = base_values.copy()
            minus_vals = base_values.copy()
            plus_vals[idx] += delta
            minus_vals[idx] -= delta
            try:
                plus_val = float(evaluator({k: float(v) for k, v in zip(keys, plus_vals.tolist())}))
                minus_val = float(evaluator({k: float(v) for k, v in zip(keys, minus_vals.tolist())}))
            except Exception as exc:
                self.logger.debug("Finite-difference eval failed for %s/%s: %s", model_name, key, exc)
                return None, None
            grads[key] = (plus_val - minus_val) / (2 * delta)
            hess[key] = (plus_val - 2 * center + minus_val) / (delta * delta)

        gradient_norm = float(np.linalg.norm(np.array(list(grads.values()), dtype=float)))
        return gradient_norm, hess

    def _extract_tolerance_from_config(self) -> float | str | None:
        config = self._ensure_science_config()
        if not config:
            return None
        settings = config.engine_settings or {}
        for key in ("tolerance", "tol", "stop_tolerance", "convergence_tol"):
            if key in settings:
                return settings[key]
        return None
