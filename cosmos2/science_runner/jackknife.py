"""Jackknife resampling functionality for cosmos2 science runner."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional, Tuple
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from collections import defaultdict

import numpy as np


@dataclass
class JackknifeConfig:
    """Configuration for jackknife resampling."""
    enabled: bool = False
    n_draws: int = 100
    fraction_removed: float = 0.1
    random_seed: int | None = None
    datasets_to_test: List[str] = field(default_factory=lambda: ["sn", "bao", "cmb", "cc", "rsd"])
    dataset_weights: Dict[str, float] = field(default_factory=dict)

    @property
    def is_enabled(self) -> bool:
        return self.enabled
    
    def __post_init__(self) -> None:
        if self.enabled:
            if self.n_draws < 1:
                raise ValueError("n_draws must be >= 1")
            if not 0 < self.fraction_removed < 1:
                raise ValueError("fraction_removed must be between 0 and 1")
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any] | None) -> "JackknifeConfig":
        if not config:
            return cls(enabled=False)
        
        return cls(
            enabled=bool(config.get("enabled", False)),
            n_draws=int(config.get("n_draws", 100)),
            fraction_removed=float(config.get("fraction_removed", 0.1)),
            random_seed=config.get("random_seed"),
            datasets_to_test=list(config.get("datasets_to_test", ["sn", "bao", "cmb", "cc", "rsd"])),
            dataset_weights=dict(config.get("dataset_weights", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "n_draws": self.n_draws,
            "fraction_removed": self.fraction_removed,
            "random_seed": self.random_seed,
            "datasets_to_test": self.datasets_to_test,
            "dataset_weights": self.dataset_weights,
        }


@dataclass
class JackknifeMask:
    """Data mask for a single jackknife draw."""
    draw_index: int
    dataset_masks: Dict[str, np.ndarray]  # Boolean mask for each dataset
    random_seed: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "draw_index": self.draw_index,
            "dataset_masks": {
                dataset: mask.tolist() for dataset, mask in self.dataset_masks.items()
            },
            "random_seed": self.random_seed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JackknifeMask":
        return cls(
            draw_index=data["draw_index"],
            dataset_masks={
                dataset: np.array(mask, dtype=bool)
                for dataset, mask in data["dataset_masks"].items()
            },
            random_seed=data["random_seed"],
        )


class JackknifeResampler:
    """Jackknife resampler for dataset masking."""
    
    def __init__(self, config: JackknifeConfig, datasets: Sequence[str]) -> None:
        self.config = config
        self.datasets = list(datasets)
        self.rng = random.Random(config.random_seed)
        self.np_rng = np.random.RandomState(config.random_seed)
        
        # Store dataset sizes (will be populated during masking)
        self.dataset_sizes: Dict[str, int] = {}
        
        # Store generated masks for reproducibility
        self.masks: List[JackknifeMask] = []
    
    def set_dataset_size(self, dataset: str, size: int) -> None:
        """Set the size of a dataset for masking."""
        if dataset not in self.datasets:
            raise ValueError(f"Dataset '{dataset}' not in configured datasets: {self.datasets}")
        self.dataset_sizes[dataset] = size
    
    def generate_masks(self) -> List[JackknifeMask]:
        """Generate jackknife masks for all draws."""
        if not self.dataset_sizes:
            raise ValueError("Dataset sizes must be set before generating masks")
        
        self.masks = []
        for draw_idx in range(self.config.n_draws):
            mask = self._generate_single_mask(draw_idx)
            self.masks.append(mask)
        
        return self.masks
    
    def _generate_single_mask(self, draw_index: int) -> JackknifeMask:
        """Generate a mask for a single jackknife draw."""
        dataset_masks: Dict[str, np.ndarray] = {}
        
        # Use different seed for each draw but ensure reproducibility
        seed = self.config.random_seed + draw_index if self.config.random_seed is not None else draw_index
        draw_rng = np.random.RandomState(seed)
        
        for dataset in self.datasets:
            size = self.dataset_sizes[dataset]
            if size == 0:
                # Empty dataset - all False mask
                dataset_masks[dataset] = np.zeros(size, dtype=bool)
                continue
            
            # Calculate number of points to remove
            n_remove = max(1, int(size * self.config.fraction_removed))
            
            # Generate random mask
            mask = np.ones(size, dtype=bool)
            if n_remove >= size:
                # Remove all points - edge case
                mask[:] = False
            else:
                # Randomly select points to remove
                remove_indices = draw_rng.choice(size, n_remove, replace=False)
                mask[remove_indices] = False
            
            dataset_masks[dataset] = mask
        
        return JackknifeMask(
            draw_index=draw_index,
            dataset_masks=dataset_masks,
            random_seed=seed,
        )
    
    def save_masks(self, path: Path) -> None:
        """Save all masks to a JSON file for reproducibility."""
        masks_data = {
            "config": {
                "enabled": self.config.enabled,
                "n_draws": self.config.n_draws,
                "fraction_removed": self.config.fraction_removed,
                "random_seed": self.config.random_seed,
                "dataset_weights": self.config.dataset_weights,
            },
            "datasets": self.datasets,
            "dataset_sizes": self.dataset_sizes,
            "masks": [mask.to_dict() for mask in self.masks],
        }
        
        path.write_text(json.dumps(masks_data, indent=2), encoding="utf-8")
    
    @classmethod
    def load_masks(cls, path: Path) -> "JackknifeResampler":
        """Load masks from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        
        config = JackknifeConfig.from_dict(data["config"])
        resampler = cls(config, data["datasets"])
        resampler.dataset_sizes = dict(data["dataset_sizes"])
        resampler.masks = [JackknifeMask.from_dict(mask_data) for mask_data in data["masks"]]
        
        return resampler


def apply_mask_to_dataset(dataset_data: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
    """Apply a boolean mask to dataset data."""
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask, dtype=bool)
    
    masked_data = {}
    
    # Handle different dataset formats
    for key, value in dataset_data.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            # Convert to numpy array for boolean indexing
            arr = np.asarray(value)
            arr_len = arr.shape[0] if arr.ndim > 0 else None
            if arr_len == len(mask):
                masked_arr = arr[mask]
                # Handle covariance matrix masking
                if key in ["cov", "inv_cov"] and masked_arr.shape != arr.shape:
                    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                        masked_cov = arr[np.ix_(mask, mask)]
                        masked_data[key] = masked_cov
                    else:
                        masked_data[key] = masked_arr
                else:
                    masked_data[key] = masked_arr.tolist() if isinstance(value, (list, tuple)) else masked_arr
            else:
                # Keep as-is if dimensions don't match (e.g., metadata)
                masked_data[key] = value
        else:
            # Keep non-array data as-is
            masked_data[key] = value
    
    return masked_data


def create_masked_datasets(
    original_datasets: Dict[str, Any], 
    mask: JackknifeMask
) -> Dict[str, Any]:
    """Create masked versions of all datasets for a jackknife draw."""
    masked_datasets = {}
    
    for dataset_name, dataset_data in original_datasets.items():
        if dataset_name in mask.dataset_masks:
            dataset_mask = mask.dataset_masks[dataset_name]
            masked_datasets[dataset_name] = apply_mask_to_dataset(dataset_data, dataset_mask)
        else:
            # Keep dataset as-is if no mask for it
            masked_datasets[dataset_name] = dataset_data
    
    return masked_datasets

@dataclass
class ModelResult:
    """Result for a single model in a jackknife draw."""
    model_name: str
    parameters: Dict[str, float]
    chi_squared: float
    aic: Optional[float] = None
    bic: Optional[float] = None
    dof: Optional[int] = None
    convergence_status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "parameters": self.parameters,
            "chi_squared": self.chi_squared,
            "aic": self.aic,
            "bic": self.bic,
            "dof": self.dof,
            "convergence_status": self.convergence_status
        }


@dataclass
class JackknifeDraw:
    """Results from a single jackknife draw."""
    draw_index: int
    removed_datasets: Dict[str, Any]
    original_models: Dict[str, ModelResult]
    jackknife_models: Dict[str, ModelResult]
    best_model_full: str
    best_model_jackknife: str
    success: bool
    error_message: Optional[str] = None
    random_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "draw_index": self.draw_index,
            "removed_datasets": self.removed_datasets,
            "original_models": {name: res.to_dict() for name, res in self.original_models.items()},
            "jackknife_models": {name: res.to_dict() for name, res in self.jackknife_models.items()},
            "best_model_full": self.best_model_full,
            "best_model_jackknife": self.best_model_jackknife,
            "success": self.success,
            "error_message": self.error_message,
            "random_seed": self.random_seed
        }


class RemovalStrategy(Enum):
    """Selective removal strategies for jackknife."""
    RANDOM_PERCENTAGE = "random_percentage"
    REDSHIFT_BANDS = "redshift_bands"
    SURVEY_ORIGIN = "survey_origin"
    SPECIFIC_POINTS = "specific_points"
    DATASET_REMOVAL = "dataset_removal"
    K_MODE_RANGES = "k_mode_ranges"
    L_MODE_RANGES = "l_mode_ranges"


@dataclass
class SelectiveRemovalConfig:
    """Configuration for selective data removal strategies."""
    strategy: RemovalStrategy
    dataset_type: str
    fraction: float = 0.1
    specific_indices: List[int] = field(default_factory=list)
    redshift_ranges: List[Tuple[float, float]] = field(default_factory=list)
    survey_names: List[str] = field(default_factory=list)
    k_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))
    l_range: Tuple[int, int] = field(default_factory=lambda: (0, 2500))


class SimpleJackknifeRunner:
    """Simple, reliable jackknife runner that uses file replacement for data masking."""

    def __init__(self, config: JackknifeConfig, models: List[str]):
        self.config = config
        self.models = models
        self.rng = np.random.RandomState(config.random_seed or 42)

    def run_jackknife(
        self,
        original_datasets: Dict[str, Any],
        baseline_models: Dict[str, ModelResult],
        fit_function
    ) -> List[JackknifeDraw]:
        """Run jackknife analysis with multiple draws."""
        print(f"[jackknife] Running {self.config.n_draws} jackknife draws with {len(self.models)} models")
        print(f"[jackknife] Models: {', '.join(self.models)}")
        print(f"[jackknife] Sequential execution enabled (file replacement requires sequential processing)")

        draws = []

        # Run sequentially - parallel execution conflicts with file replacement
        for draw_idx in range(self.config.n_draws):
            try:
                draw = self._run_single_draw(draw_idx, original_datasets, baseline_models, fit_function)
                draws.append(draw)

                status = "SUCCESS" if draw.success else "FAILED"
                model_change = "→" if draw.best_model_full != draw.best_model_jackknife else "="
                print(f"[jackknife] Draw {draw_idx + 1}/{self.config.n_draws}: {status} ({draw.best_model_full} {model_change} {draw.best_model_jackknife})")

            except Exception as e:
                print(f"[jackknife] Draw {draw_idx + 1} failed: {e}")
                # Create failed draw
                draw = JackknifeDraw(
                    draw_index=draw_idx,
                    removed_datasets={},
                    original_models=baseline_models,
                    jackknife_models={},
                    best_model_full=list(baseline_models.keys())[0] if baseline_models else "unknown",
                    best_model_jackknife="unknown",
                    success=False,
                    error_message=str(e)
                )
                draws.append(draw)

        return draws

    def _run_single_draw(
        self,
        draw_idx: int,
        original_datasets: Dict[str, Any],
        original_models: Dict[str, ModelResult],
        fit_function
    ) -> JackknifeDraw:
        """Run a single jackknife draw."""
        # Use deterministic seed for reproducibility
        seed = (self.config.random_seed or 0) + draw_idx * 1000
        draw_rng = np.random.RandomState(seed)

        # Create jackknife datasets
        jackknife_datasets = {}
        removed_info = {}

        # Simple random removal for now (can be extended with selective strategies later)
        for dataset_name in self.config.datasets_to_test:
            if dataset_name not in original_datasets:
                continue

            dataset = original_datasets[dataset_name]
            jackknife_dataset, n_removed = self._create_jackknife_dataset(
                dataset, self.config.fraction_removed, draw_rng
            )

            jackknife_datasets[dataset_name] = jackknife_dataset
            removed_info[dataset_name] = n_removed

        # Keep datasets not being tested unchanged
        for dataset_name, dataset in original_datasets.items():
            if dataset_name not in jackknife_datasets:
                jackknife_datasets[dataset_name] = dataset

        # Run optimization with masked datasets
        jackknife_models = {}
        success = True
        error_msg = None

        try:
            jackknife_results = fit_function(jackknife_datasets, self.models)

            # Convert results to ModelResult objects
            for model_name in self.models:
                if model_name in jackknife_results:
                    model_data = jackknife_results[model_name]
                    # Debug: check what we got
                    print(f"[jackknife] Draw {draw_idx}: Got results for {model_name}")
                    print(f"[jackknife] Draw {draw_idx}: Full result keys: {list(model_data.keys())}")
                    print(f"[jackknife] Draw {draw_idx}: Chi2: {model_data.get('chi_squared')}")
                    print(f"[jackknife] Draw {draw_idx}: Parameters type: {type(model_data.get('parameters'))}")
                    print(f"[jackknife] Draw {draw_idx}: Parameters value: {model_data.get('parameters')}")

                    # Ensure parameters is a dict, not None
                    parameters = model_data.get("parameters", {})
                    if parameters is None:
                        print(f"[jackknife] Draw {draw_idx}: Parameters was None, using empty dict")
                        parameters = {}

                    model_result = ModelResult(
                        model_name=model_name,
                        parameters=parameters,
                        chi_squared=model_data.get("chi_squared", float('inf')),
                        aic=model_data.get("aic"),
                        bic=model_data.get("bic"),
                        dof=model_data.get("dof"),
                        convergence_status=model_data.get("convergence_status", "unknown")
                    )
                    jackknife_models[model_name] = model_result
                else:
                    jackknife_models[model_name] = ModelResult(
                        model_name=model_name,
                        parameters={},
                        chi_squared=float('inf'),
                        convergence_status="failed"
                    )

            # Determine best model
            valid_models = [(name, model) for name, model in jackknife_models.items()
                          if model.chi_squared < float('inf')]
            best_model_jackknife = min(valid_models, key=lambda x: x[1].chi_squared)[0] if valid_models else "unknown"

        except Exception as e:
            error_msg = str(e)
            success = False
            best_model_jackknife = "unknown"

        # Use first model as baseline
        best_model_full = list(original_models.keys())[0] if original_models else "unknown"

        return JackknifeDraw(
            draw_index=draw_idx,
            removed_datasets=removed_info,
            original_models=original_models,
            jackknife_models=jackknife_models,
            best_model_full=best_model_full,
            best_model_jackknife=best_model_jackknife,
            success=success,
            error_message=error_msg,
            random_seed=seed
        )

    def _create_jackknife_dataset(
        self,
        dataset: Dict[str, Any],
        fraction_to_remove: float,
        rng: np.random.RandomState
    ) -> tuple[Dict[str, Any], int]:
        """Create a jackknife version of a dataset by removing random points."""
        if not dataset:
            return dataset, 0

        # Find the size of the dataset
        data_size = self._get_dataset_size(dataset)
        if data_size == 0:
            return dataset, 0

        # Calculate how many points to remove
        n_to_remove = max(1, int(data_size * fraction_to_remove))
        n_to_remove = min(n_to_remove, data_size // 2)  # Never remove more than 50%

        # Randomly select indices to remove
        remove_indices = rng.choice(data_size, n_to_remove, replace=False)

        # Create jackknife dataset
        jackknife_dataset = self._apply_mask_to_dataset(dataset, remove_indices)

        return jackknife_dataset, n_to_remove

    def _get_dataset_size(self, dataset: Dict[str, Any]) -> int:
        """Get the size of a dataset."""
        # Look for common data fields
        for field in ['data', 'z', 'mu', 'distances']:
            if field in dataset:
                value = dataset[field]
                if isinstance(value, (list, tuple, np.ndarray)):
                    return len(value)

        # If no standard field found, try to find the largest array
        max_size = 0
        for value in dataset.values():
            if isinstance(value, (list, tuple, np.ndarray)):
                max_size = max(max_size, len(value))

        return max_size

    def _apply_mask_to_dataset(self, dataset: Dict[str, Any], remove_indices: np.ndarray) -> Dict[str, Any]:
        """Apply removal mask to dataset."""
        dataset_size = self._get_dataset_size(dataset)
        print(f"[jackknife] Applying mask: dataset_size={dataset_size}, remove_indices len={len(remove_indices)}")
        if dataset_size == 0 or len(remove_indices) == 0:
            return dataset

        # Create boolean mask (True = keep, False = remove)
        mask = np.ones(dataset_size, dtype=bool)
        mask[remove_indices] = False

        jackknife_dataset = {}
        for key, value in dataset.items():
            try:
                if key in ['z', 'obs', 'err', 'labels'] and hasattr(value, '__len__'):
                    # These are the main data arrays that should be masked
                    if isinstance(value, np.ndarray) and len(value) == dataset_size:
                        jackknife_dataset[key] = value[mask]
                    elif isinstance(value, (list, tuple)) and len(value) == dataset_size:
                        jackknife_dataset[key] = [value[i] for i, keep in enumerate(mask) if keep]
                    else:
                        jackknife_dataset[key] = value  # Keep as-is if size doesn't match
                elif key in ['cov', 'inv_cov'] and hasattr(value, 'shape'):
                    # Handle covariance matrices
                    if len(value.shape) == 2 and value.shape[0] == dataset_size and value.shape[1] == dataset_size:
                        jackknife_dataset[key] = value[np.ix_(mask, mask)]
                    else:
                        jackknife_dataset[key] = value  # Keep as-is
                else:
                    # Keep metadata and other fields unchanged
                    jackknife_dataset[key] = value
            except Exception as e:
                # If anything fails, keep the original value
                jackknife_dataset[key] = value

        return jackknife_dataset

    def _create_masked_fit_function(self, original_fit_fn, masked_dataset):
        """Create a fit function that uses masked data."""
        def masked_fit_fn(model, dataset=None):
            try:
                # Call the original fit function with the masked dataset
                result = original_fit_fn(model, dataset=masked_dataset)
                return result
            except Exception as e:
                print(f"[jackknife] Masked fit error: {e}")
                return float('inf'), {}

        # Preserve the original function name for debugging
        masked_fit_fn.__name__ = getattr(original_fit_fn, '__name__', 'masked_fit')
        return masked_fit_fn


def analyze_jackknife_results(draws: List[JackknifeDraw], config: JackknifeConfig) -> Dict[str, Any]:
    """Analyze jackknife results and compute stability metrics."""
    if not draws:
        return {"error": "No draws to analyze"}

    successful_draws = [d for d in draws if d.success]

    overall_param_values: Dict[str, List[float]] = defaultdict(list)
    overall_chi2_values: List[float] = []

    analysis = {
        "success_rate": len(successful_draws) / len(draws),
        "n_draws_total": len(draws),
        "n_draws_successful": len(successful_draws),
        "stability_metrics": {
            "overall_stability_score": 0.0,
            "model_comparison": {},
            "dataset_impact": {}
        },
        "model_analyses": {},
        "draws": [d.to_dict() for d in draws]
    }

    analysis["stability_metrics"].update({
        "n_draws_total": len(draws),
        "n_draws_successful": len(successful_draws)
    })

    if not successful_draws:
        return analysis

    # Analyze each model
    all_models = set()
    for draw in successful_draws:
        all_models.update(draw.jackknife_models.keys())

    for model_name in all_models:
        model_analysis = {
            "model_name": model_name,
            "n_draws": len(successful_draws),
            "parameter_stability": {"parameter_stats": {}, "stability_score": 0.0},
            "chi2_stability": {},
            "overall_stability_score": 0.0
        }

        # Collect parameter values across draws
        param_values: Dict[str, List[float]] = {}
        chi2_values: List[float] = []

        for draw in successful_draws:
            if model_name in draw.jackknife_models:
                model_result = draw.jackknife_models[model_name]
                chi2_val = model_result.chi_squared
                if isinstance(chi2_val, (int, float)) and np.isfinite(chi2_val):
                    chi2_values.append(float(chi2_val))
                    overall_chi2_values.append(float(chi2_val))

                # Check if parameters exist and are valid
                if model_result.parameters and isinstance(model_result.parameters, dict):
                    for param, value in model_result.parameters.items():
                        if value is None:
                            continue
                        try:
                            float_value = float(value)
                        except (ValueError, TypeError):
                            continue
                        if not np.isfinite(float_value):
                            continue
                        param_values.setdefault(param, []).append(float_value)
                        overall_param_values.setdefault(param, []).append(float_value)

        # Calculate parameter stability
        cv_values = []
        if param_values:
            for param, values in param_values.items():
                values_array = np.asarray(values, dtype=float)
                finite = np.isfinite(values_array)
                values_array = values_array[finite]
                if values_array.size == 0:
                    continue

                mean_val = float(np.mean(values_array))
                std_val = float(np.std(values_array))
                cv = abs(std_val / mean_val) if mean_val != 0 else 0.0
                cv_values.append(cv)
                stability_label = "stable" if cv < 0.1 else "moderate" if cv < 0.2 else "unstable"

                model_analysis["parameter_stability"]["parameter_stats"][param] = {
                    "mean": mean_val,
                    "std": std_val,
                    "coefficient_of_variation": cv,
                    "n_values": int(values_array.size),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "stability": stability_label
                }

            # Calculate overall stability score
            if cv_values:
                score = 1.0 - min(1.0, float(np.mean(cv_values)))
                model_analysis["overall_stability_score"] = score
                model_analysis["parameter_stability"]["stability_score"] = score

        # Record chi2 stability
        if chi2_values:
            chi2_array = np.asarray(chi2_values, dtype=float)
            finite = np.isfinite(chi2_array)
            chi2_array = chi2_array[finite]
            if chi2_array.size > 0:
                model_analysis["chi2_stability"] = {
                    "mean_chi2": float(np.mean(chi2_array)),
                    "std_chi2": float(np.std(chi2_array)),
                    "min_chi2": float(np.min(chi2_array)),
                    "max_chi2": float(np.max(chi2_array)),
                    "n_values": int(chi2_array.size)
                }

        analysis["model_analyses"][model_name] = model_analysis

    # Update overall stability score
    if analysis["model_analyses"]:
        stability_scores = [model["overall_stability_score"] for model in analysis["model_analyses"].values()]
        analysis["stability_metrics"]["overall_stability_score"] = float(np.mean(stability_scores))

    # Aggregate parameter shifts across all models
    parameter_stats: Dict[str, Dict[str, Any]] = {}
    parameter_cv_values: List[float] = []
    for param, values in overall_param_values.items():
        values_array = np.asarray(values, dtype=float)
        finite = np.isfinite(values_array)
        values_array = values_array[finite]
        if values_array.size == 0:
            continue

        mean_val = float(np.mean(values_array))
        std_val = float(np.std(values_array))
        min_val = float(np.min(values_array))
        max_val = float(np.max(values_array))
        cv = abs(std_val / mean_val) if mean_val != 0 else 0.0
        parameter_cv_values.append(cv)

        parameter_stats[param] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "coefficient_of_variation": cv,
        }

    parameter_stability_score = 0.0
    if parameter_cv_values:
        parameter_stability_score = 1.0 - min(1.0, float(np.mean(parameter_cv_values)))

    analysis["parameter_shifts"] = {
        "parameter_stats": parameter_stats,
        "stability_score": parameter_stability_score,
        "n_parameters_analyzed": len(parameter_stats)
    }
    analysis["stability_metrics"]["parameter_stability_score"] = parameter_stability_score

    # Aggregate chi² changes across all models
    chi2_stats: Dict[str, Any] = {}
    chi2_stability_score = 0.0
    if overall_chi2_values:
        chi2_array = np.asarray(overall_chi2_values, dtype=float)
        finite = np.isfinite(chi2_array)
        chi2_array = chi2_array[finite]
        if chi2_array.size > 0:
            min_chi2 = float(np.min(chi2_array))
            max_chi2 = float(np.max(chi2_array))
            mean_chi2 = float(np.mean(chi2_array))
            std_chi2 = float(np.std(chi2_array))
            relative_std = float(std_chi2 / mean_chi2) if mean_chi2 != 0 else 0.0
            chi2_stats = {
                "mean": mean_chi2,
                "std": std_chi2,
                "min": min_chi2,
                "max": max_chi2,
                "n_values": int(chi2_array.size),
                "range": max_chi2 - min_chi2,
                "relative_std": relative_std
            }
            chi2_stability_score = 1.0 - min(1.0, relative_std)

    analysis["chi2_changes"] = {
        "chi2_stats": chi2_stats,
        "stability_score": chi2_stability_score
    }
    analysis["stability_metrics"]["chi2_stability_score"] = chi2_stability_score

    return analysis
