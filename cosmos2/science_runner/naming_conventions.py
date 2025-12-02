"""Naming conventions for cosmos2 science runner outputs."""

from __future__ import annotations

from typing import Dict, List


class NamingConventions:
    """Consistent naming conventions for figures, tables, and outputs."""
    
    # Base prefixes
    FIGURE_PREFIX = "fig"
    TABLE_PREFIX = "tab"
    PLOT_PREFIX = "plot"
    
    # Model naming
    MODEL_NAMES = {
        "lcdm": "ΛCDM",
        "pbuf": "PBUF",
        "wcdm": "wCDM",
        "quintessence": "Quintessence"
    }
    
    # Dataset naming
    DATASET_NAMES = {
        "cmb": "CMB",
        "sn": "SN Ia",
        "bao_iso": "BAO (iso)",
        "bao_aniso": "BAO (aniso)",
        "cc": "Cosmic Chronometers",
        "rsd": "Redshift-Space Distortions",
        "wl": "Weak Lensing",
        "lensing_cross": "Lensing Cross-Correlations"
    }
    
    # Parameter naming (for display)
    PARAMETER_DISPLAY_NAMES = {
        "H0": "H₀",
        "Omega_m0": "Ωₘ,₀",
        "Omega_b0": "Ω_b,₀", 
        "Omega_k0": "Ω_k,₀",
        "Rmax": "R_max",
        "alpha": "α",
        "w": "w",
        "wa": "w_a"
    }
    
    # Derived quantity naming
    DERIVED_QUANTITY_NAMES = {
        "r_d": "r_d",
        "S8": "S₈",
        "q0": "q₀",
        "z_star": "z_*",
        "age_Gyr": "Age (Gyr)",
        "sigma8": "σ₈"
    }
    
    # Figure types and their standard labels
    FIGURE_TYPES = {
        "hubble_diagram": "Hubble Diagram",
        "bao_distance_comparison": "BAO Distance Comparison",
        "hz_comparison": "H(z) Comparison", 
        "growth_rate": "Growth Rate",
        "omega_sigma_evolution": "Ωσ(a) Evolution",
        "temperature_evolution": "Temperature Evolution",
        "deceleration_curve": "Deceleration Curve q(z)",
        "sound_horizon_evolution": "Sound Horizon Evolution",
        "residuals_grid": "Residuals Grid",
        "chi2_stability": "χ² Stability",
        "parameter_stability": "Parameter Stability",
        "qz_evolution": "q(z) Evolution Band"
    }
    
    # Table types and their standard labels
    TABLE_TYPES = {
        "parameters": "Best-fit Parameters",
        "derived_quantities": "Derived Quantities",
        "dataset_statistics": "Dataset Statistics",
        "chi2_statistics": "χ² Statistics",
        "jackknife_summary": "Jackknife Summary",
        "quantum_parameters": "Quantum Parameters",
        "model_comparison": "Model Comparison"
    }
    
    @classmethod
    def get_figure_id(cls, figure_type: str, model: str | None = None, suffix: str = "") -> str:
        """Generate consistent figure ID."""
        parts = [cls.FIGURE_PREFIX, figure_type]
        
        if model:
            parts.append(model)
        
        if suffix:
            parts.append(suffix)
        
        return "_".join(parts)
    
    @classmethod
    def get_table_id(cls, table_type: str, model: str | None = None, suffix: str = "") -> str:
        """Generate consistent table ID."""
        parts = [cls.TABLE_PREFIX, table_type]
        
        if model:
            parts.append(model)
        
        if suffix:
            parts.append(suffix)
        
        return "_".join(parts)
    
    @classmethod
    def get_figure_filename(cls, figure_type: str, model: str | None = None, suffix: str = "") -> str:
        """Generate consistent figure filename."""
        return f"{cls.get_figure_id(figure_type, model, suffix)}.png"
    
    @classmethod
    def get_table_filename(cls, table_type: str, model: str | None = None, suffix: str = "") -> str:
        """Generate consistent table filename."""
        return f"{cls.get_table_id(table_type, model, suffix)}.csv"
    
    @classmethod
    def get_figure_label(cls, figure_type: str, model: str | None = None) -> str:
        """Generate consistent figure label for LaTeX."""
        base_label = cls.get_figure_id(figure_type, model)
        return f"\\label{{{base_label}}}"
    
    @classmethod
    def get_table_label(cls, table_type: str, model: str | None = None) -> str:
        """Generate consistent table label for LaTeX."""
        base_label = cls.get_table_id(table_type, model)
        return f"\\label{{{base_label}}}"
    
    @classmethod
    def get_figure_caption(cls, figure_type: str, model: str | None = None) -> str:
        """Generate consistent figure caption."""
        figure_name = cls.FIGURE_TYPES.get(figure_type, figure_type.replace("_", " ").title())
        
        if model:
            model_display = cls.MODEL_NAMES.get(model.lower(), model)
            return f"{figure_name} for {model_display} model"
        else:
            return figure_name
    
    @classmethod
    def get_table_caption(cls, table_type: str, model: str | None = None) -> str:
        """Generate consistent table caption."""
        table_name = cls.TABLE_TYPES.get(table_type, table_type.replace("_", " ").title())
        
        if model:
            model_display = cls.MODEL_NAMES.get(model.lower(), model)
            return f"{table_name} for {model_display} model"
        else:
            return table_name
    
    @classmethod
    def format_parameter_name(cls, param_name: str) -> str:
        """Format parameter name for display."""
        return cls.PARAMETER_DISPLAY_NAMES.get(param_name, param_name)
    
    @classmethod
    def format_derived_quantity_name(cls, quantity_name: str) -> str:
        """Format derived quantity name for display."""
        return cls.DERIVED_QUANTITY_NAMES.get(quantity_name, quantity_name)
    
    @classmethod
    def format_model_name(cls, model_name: str) -> str:
        """Format model name for display."""
        return cls.MODEL_NAMES.get(model_name.lower(), model_name)
    
    @classmethod
    def format_dataset_name(cls, dataset_name: str) -> str:
        """Format dataset name for display."""
        return cls.DATASET_NAMES.get(dataset_name, dataset_name)
    
    @classmethod
    def get_parameter_unit(cls, param_name: str) -> str:
        """Get unit for parameter."""
        units = {
            "H0": "km s⁻¹ Mpc⁻¹",
            "Omega_m0": "",
            "Omega_b0": "",
            "Omega_k0": "",
            "Rmax": "GeV",
            "alpha": "",
            "w": "",
            "wa": ""
        }
        return units.get(param_name, "")
    
    @classmethod
    def get_derived_quantity_unit(cls, quantity_name: str) -> str:
        """Get unit for derived quantity."""
        units = {
            "r_d": "Mpc",
            "S8": "",
            "q0": "",
            "z_star": "",
            "age_Gyr": "Gyr",
            "sigma8": ""
        }
        return units.get(quantity_name, "")
    
    @classmethod
    def validate_naming(cls, items: Dict[str, str]) -> List[str]:
        """Validate that naming follows conventions."""
        issues = []
        
        for item_id, item_name in items.items():
            # Check for consistent prefixes
            if item_id.startswith(cls.FIGURE_PREFIX):
                expected_type = "figure"
            elif item_id.startswith(cls.TABLE_PREFIX):
                expected_type = "table"
            elif item_id.startswith(cls.PLOT_PREFIX):
                expected_type = "plot"
            else:
                issues.append(f"Item {item_id} lacks proper prefix")
                continue
            
            # Check for snake_case
            if "_" not in item_id and len(item_id.split("_")) < 2:
                issues.append(f"Item {item_id} should use snake_case with descriptive parts")
        
        return issues
    
    @classmethod
    def get_standard_figure_set(cls, models: List[str]) -> Dict[str, str]:
        """Get standard set of figures for given models."""
        figures = {}
        
        # Core figures for all models
        core_figures = [
            "hubble_diagram",
            "bao_distance_comparison", 
            "hz_comparison",
            "growth_rate",
            "omega_sigma_evolution",
            "temperature_evolution",
            "deceleration_curve",
            "residuals_grid"
        ]
        
        for figure_type in core_figures:
            figures[cls.get_figure_id(figure_type)] = cls.get_figure_caption(figure_type)
        
        # Model-specific figures
        for model in models:
            if model.lower() == "pbuf":
                figures[cls.get_figure_id("sound_horizon_evolution", model)] = \
                    cls.get_figure_caption("sound_horizon_evolution", model)
        
        return figures
    
    @classmethod
    def get_standard_table_set(cls, models: List[str]) -> Dict[str, str]:
        """Get standard set of tables for given models."""
        tables = {}
        
        # Core tables
        core_tables = [
            "parameters",
            "derived_quantities", 
            "dataset_statistics",
            "chi2_statistics"
        ]
        
        for table_type in core_tables:
            tables[cls.get_table_id(table_type)] = cls.get_table_caption(table_type)
        
        # Model-specific tables
        for model in models:
            tables[cls.get_table_id("parameters", model)] = \
                cls.get_table_caption("parameters", model)
            tables[cls.get_table_id("derived_quantities", model)] = \
                cls.get_table_caption("derived_quantities", model)
        
        return tables
