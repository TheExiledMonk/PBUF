# PBUF Cosmology Pipeline - Configuration Examples

This document provides comprehensive examples of configuration files and command-line usage for the PBUF cosmology pipeline, demonstrating the full range of extensibility features.

## Basic Configuration Examples

### 1. Simple Individual Fitting

#### Command Line
```bash
# Basic CMB fit with ΛCDM
python pipelines/fit_cmb.py --model lcdm

# PBUF fit with parameter override
python pipelines/fit_cmb.py --model pbuf --alpha 1e-3 --eps0 0.8

# BAO fit with integrity checks
python pipelines/fit_bao.py --model pbuf --verify-integrity
```

#### Configuration File (basic_cmb.json)
```json
{
    "model": "lcdm",
    "datasets": ["cmb"],
    "parameters": {
        "H0": 67.4,
        "Om0": 0.315,
        "Obh2": 0.02237,
        "ns": 0.9649
    },
    "output": {
        "format": "human",
        "save_file": "cmb_results.txt"
    }
}
```

### 2. Joint Fitting Configuration

#### Multi-Dataset Joint Fit (joint_analysis.json)
```json
{
    "model": "pbuf",
    "datasets": ["cmb", "bao", "sn"],
    "parameters": {
        "H0": 67.4,
        "Om0": 0.315,
        "alpha": 5e-4,
        "eps0": 0.7,
        "k_sat": 0.9762
    },
    "optimizer": {
        "method": "minimize",
        "options": {
            "maxiter": 1000,
            "ftol": 1e-6
        }
    },
    "integrity_checks": {
        "enabled": true,
        "tolerance": 1e-4
    },
    "output": {
        "format": "json",
        "save_file": "pbuf_joint_results.json",
        "include_diagnostics": true,
        "include_predictions": true
    }
}
```

## Advanced Configuration Examples

### 3. Global Optimization Configuration

#### Differential Evolution Setup (global_opt.json)
```json
{
    "model": "pbuf",
    "datasets": ["cmb", "bao", "sn"],
    "optimizer": {
        "method": "differential_evolution",
        "options": {
            "maxiter": 1000,
            "seed": 42,
            "popsize": 15,
            "atol": 1e-6,
            "tol": 1e-6,
            "workers": 4
        },
        "bounds": {
            "H0": [60, 80],
            "Om0": [0.2, 0.4],
            "Obh2": [0.020, 0.025],
            "ns": [0.9, 1.0],
            "alpha": [1e-6, 1e-2],
            "eps0": [0.3, 1.0],
            "k_sat": [0.5, 1.0]
        }
    },
    "output": {
        "format": "json",
        "save_file": "global_optimization_results.json"
    }
}
```

### 4. Model Comparison Analysis

#### Multi-Model Comparison (model_comparison.json)
```json
{
    "analysis_type": "model_comparison",
    "models": ["lcdm", "pbuf"],
    "datasets": ["cmb", "bao", "sn"],
    "comparison_metrics": ["chi2", "aic", "bic", "reduced_chi2"],
    
    "model_configs": {
        "lcdm": {
            "parameters": {
                "H0": 67.4,
                "Om0": 0.315,
                "Obh2": 0.02237,
                "ns": 0.9649
            },
            "optimizer": {
                "method": "minimize",
                "options": {"maxiter": 1000}
            }
        },
        "pbuf": {
            "parameters": {
                "H0": 67.4,
                "Om0": 0.315,
                "alpha": 5e-4,
                "eps0": 0.7
            },
            "optimizer": {
                "method": "differential_evolution",
                "options": {"maxiter": 500, "seed": 42},
                "bounds": {
                    "alpha": [1e-6, 1e-2],
                    "eps0": [0.3, 1.0]
                }
            }
        }
    },
    
    "output": {
        "comparison_table": "model_comparison.csv",
        "detailed_results": "detailed_comparison.json",
        "summary_report": "comparison_summary.txt"
    }
}
```

### 5. Parameter Grid Search

#### Grid Search Configuration (grid_search.json)
```json
{
    "analysis_type": "parameter_grid",
    "model": "pbuf",
    "datasets": ["cmb", "bao"],
    
    "parameter_grid": {
        "H0": [65, 67, 70, 73, 75],
        "Om0": [0.25, 0.30, 0.35],
        "alpha": [1e-4, 5e-4, 1e-3],
        "eps0": [0.5, 0.7, 0.9]
    },
    
    "fixed_parameters": {
        "Obh2": 0.02237,
        "ns": 0.9649,
        "k_sat": 0.9762
    },
    
    "optimizer": {
        "method": "minimize",
        "options": {"maxiter": 500}
    },
    
    "output": {
        "grid_results": "parameter_grid_results.csv",
        "best_fit_summary": "best_fit_parameters.json",
        "contour_data": "parameter_contours.json"
    }
}
```

## Extension Configuration Examples

### 6. Adding New Models

#### w-CDM Model Extension (wcdm_extension.json)
```json
{
    "extensions": {
        "models": {
            "wcdm": {
                "parameters": {
                    "H0": 67.4,
                    "Om0": 0.315,
                    "Obh2": 0.02237,
                    "ns": 0.9649,
                    "w0": -1.0,
                    "wa": 0.0,
                    "model_class": "wcdm"
                },
                "physics_modules": ["wcdm_models"],
                "documentation": {
                    "theory": "documents/wcdm_theory.md",
                    "validation": "documents/wcdm_validation.md"
                },
                "validation_tests": {
                    "lcdm_limit": {
                        "w0": -1.0,
                        "wa": 0.0,
                        "tolerance": 1e-6
                    }
                }
            }
        }
    },
    
    "analysis": {
        "model": "wcdm",
        "datasets": ["cmb", "bao", "sn"],
        "parameters": {
            "w0": -0.9,
            "wa": 0.1
        },
        "optimizer": {
            "method": "differential_evolution",
            "bounds": {
                "w0": [-2.0, -0.3],
                "wa": [-1.0, 1.0]
            }
        }
    }
}
```

### 7. Adding New Datasets

#### Weak Lensing Dataset Extension (weak_lensing_extension.json)
```json
{
    "extensions": {
        "datasets": {
            "weak_lensing": {
                "loader": {
                    "module": "weak_lensing_data",
                    "function": "load_des_y3_shear_correlation"
                },
                "likelihood": {
                    "function": "likelihood_weak_lensing",
                    "models": ["lcdm", "pbuf", "wcdm"]
                },
                "metadata": {
                    "survey": "DES Y3",
                    "data_type": "shear_correlation",
                    "redshift_bins": 4,
                    "angular_bins": 20
                },
                "validation": {
                    "covariance_check": true,
                    "data_format_check": true
                }
            }
        }
    },
    
    "analysis": {
        "model": "lcdm",
        "datasets": ["cmb", "bao", "sn", "weak_lensing"],
        "parameters": {
            "sigma8": 0.8159,
            "As": 2.1e-9
        },
        "output": {
            "save_file": "cosmology_with_wl.json"
        }
    }
}
```

### 8. Custom Optimization Methods

#### MCMC Sampling Configuration (mcmc_config.json)
```json
{
    "extensions": {
        "optimizers": {
            "emcee": {
                "module": "mcmc_optimizers",
                "function": "emcee_sampler",
                "options": {
                    "nwalkers": 100,
                    "nsteps": 5000,
                    "burn_in": 1000,
                    "thin": 10
                }
            },
            "dynesty": {
                "module": "nested_sampling",
                "function": "dynesty_sampler",
                "options": {
                    "nlive": 500,
                    "dlogz": 0.01
                }
            }
        }
    },
    
    "analysis": {
        "model": "pbuf",
        "datasets": ["cmb", "bao", "sn"],
        "optimizer": {
            "method": "emcee",
            "options": {
                "nwalkers": 50,
                "nsteps": 2000,
                "burn_in": 500
            },
            "bounds": {
                "H0": [60, 80],
                "Om0": [0.2, 0.4],
                "alpha": [1e-6, 1e-2]
            }
        },
        "output": {
            "chains": "mcmc_chains.h5",
            "corner_plot": "corner_plot.png",
            "summary_stats": "mcmc_summary.json"
        }
    }
}
```

## Specialized Analysis Configurations

### 9. Physics Validation Configuration

#### Comprehensive Validation Suite (validation_config.json)
```json
{
    "analysis_type": "physics_validation",
    "models": ["lcdm", "pbuf"],
    
    "validation_tests": {
        "h_ratios": {
            "enabled": true,
            "redshifts": [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
            "tolerance": 1e-4,
            "reference_model": "lcdm"
        },
        "recombination": {
            "enabled": true,
            "reference_value": 1089.80,
            "tolerance": 1.0,
            "method": "PLANCK18"
        },
        "sound_horizon": {
            "enabled": true,
            "reference_value": 147.09,
            "tolerance": 0.5,
            "method": "EH98"
        },
        "covariance_matrices": {
            "enabled": true,
            "min_eigenvalue": 1e-10,
            "condition_number_max": 1e12
        }
    },
    
    "parameter_sets": {
        "planck2018": {
            "H0": 67.36,
            "Om0": 0.3153,
            "Obh2": 0.02237,
            "ns": 0.9649
        },
        "pbuf_fiducial": {
            "H0": 67.4,
            "Om0": 0.315,
            "alpha": 5e-4,
            "eps0": 0.7,
            "k_sat": 0.9762
        }
    },
    
    "output": {
        "validation_report": "physics_validation_report.md",
        "test_results": "validation_results.json",
        "plots": "validation_plots/"
    }
}
```

### 10. Batch Processing Configuration

#### Large-Scale Analysis (batch_analysis.json)
```json
{
    "analysis_type": "batch_processing",
    "parallel_jobs": 8,
    
    "job_matrix": {
        "models": ["lcdm", "pbuf"],
        "dataset_combinations": [
            ["cmb"],
            ["bao"],
            ["sn"],
            ["cmb", "bao"],
            ["cmb", "sn"],
            ["bao", "sn"],
            ["cmb", "bao", "sn"]
        ],
        "parameter_variations": {
            "pbuf": [
                {"alpha": 1e-4, "eps0": 0.5},
                {"alpha": 5e-4, "eps0": 0.7},
                {"alpha": 1e-3, "eps0": 0.9}
            ]
        }
    },
    
    "optimization": {
        "method": "minimize",
        "options": {"maxiter": 1000},
        "timeout": 300
    },
    
    "output": {
        "results_directory": "batch_results/",
        "summary_file": "batch_summary.csv",
        "failed_jobs_log": "failed_jobs.log",
        "timing_report": "timing_analysis.json"
    },
    
    "error_handling": {
        "max_retries": 3,
        "retry_delay": 10,
        "continue_on_failure": true
    }
}
```

## Usage Examples

### Command Line Usage with Configurations

```bash
# Use basic configuration
python pipelines/fit_joint.py --config basic_cmb.json

# Override configuration parameters
python pipelines/fit_joint.py --config joint_analysis.json --H0 70.0 --maxiter 2000

# Run model comparison
python pipelines/run_model_comparison.py --config model_comparison.json

# Execute parameter grid search
python pipelines/run_parameter_grid.py --config grid_search.json

# Run with extensions
python pipelines/fit_joint.py --config wcdm_extension.json --extensions extensions.json

# Batch processing
python pipelines/run_batch_analysis.py --config batch_analysis.json --parallel 16
```

### Python API Usage with Configurations

```python
from pipelines.fit_core.config_loader import load_configuration
from pipelines.fit_core.engine import run_fit

# Load and apply configuration
config = load_configuration("joint_analysis.json")

# Run analysis with configuration
result = run_fit(
    model=config["model"],
    datasets_list=config["datasets"],
    overrides=config["parameters"],
    optimizer_config=config["optimizer"]
)

# Apply extensions if specified
if "extensions" in config:
    apply_extensions(config["extensions"])
    
    # Re-run with extended capabilities
    result = run_fit(
        model=config["model"],
        datasets_list=config["datasets"] + config["extensions"]["datasets"],
        overrides=config["parameters"]
    )
```

### Configuration Validation

```python
from pipelines.fit_core.config_validator import validate_configuration

# Validate configuration before use
config_file = "complex_analysis.json"
is_valid, errors = validate_configuration(config_file)

if is_valid:
    print("✓ Configuration is valid")
    config = load_configuration(config_file)
else:
    print("❌ Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

These configuration examples demonstrate the full flexibility and extensibility of the PBUF cosmology pipeline, enabling complex analyses through declarative configuration rather than code modification.