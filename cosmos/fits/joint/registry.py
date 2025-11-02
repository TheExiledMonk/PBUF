"""
Dynamic registry for all available dataset χ² modules.

Each dataset module should export a chi_squared_<dataset>()
function that takes a model as argument.
"""

import importlib

# Datasets we know about and want to auto-discover
DEFAULT_DATASETS = [
    "cmb",
    "sn",
    "sn.pantheon",
    "sn.sh0es",
    "bao.iso",
    "bao.aniso",
    "cc",
    "rsd",
]

def _create_model_func_from_model(model_instance):
    """Create a model_func that returns the given model instance."""
    def model_func(params):
        return model_instance
    return model_func


def _extract_params_from_model(model_instance, model_type):
    """Extract parameters from a model instance."""
    params = {}

    # Common parameters
    params["H0"] = model_instance.h * 100.0
    params["Om0"] = model_instance.omega_m
    params["Ok0"] = getattr(model_instance, 'omega_k', 0.0)
    params["Or0"] = getattr(model_instance, 'omega_r', 9.2e-5)

    # LCDM specific
    if model_type == "lcdm":
        params["Ol0"] = getattr(model_instance, 'omega_lambda', 0.0)
        params["Obh2"] = getattr(model_instance, 'omega_b', 0.02237) * (model_instance.h ** 2)

    # PBUF specific
    elif model_type == "pbuf":
        from cosmos.pbuf.model import PBUF
        if isinstance(model_instance, PBUF):
            params["alpha"] = model_instance.alpha
            params["Rmax"] = model_instance.Rmax
            params["k_sat"] = getattr(model_instance, 'k_sat', 1.0)
            params["Ol0"] = 0.0  # No cosmological constant
            params["Obh2"] = getattr(model_instance, 'omega_b', 0.02237) * (model_instance.h ** 2)

    return params


def _import_dataset_module(name):
    """
    Try importing a dataset module, e.g. cosmos.fits.cmb or cosmos.fits.sn.pantheon
    Returns (module, chi2_func) if successful, else None.
    """
    try:
        # Handle dotted module paths like sn.pantheon
        if "." in name:
            parts = name.split(".")
            if len(parts) == 2:
                # Import parent module first
                parent_module = importlib.import_module(f"cosmos.fits.{parts[0]}")
                # Then import the submodule
                module = importlib.import_module(f"cosmos.fits.{name}")
                submodule_name = parts[1]
            else:
                module = importlib.import_module(f"cosmos.fits.{name}")
                submodule_name = name.split(".")[-1]
        else:
            module = importlib.import_module(f"cosmos.fits.{name}")
            submodule_name = name

        # Look for chi2 functions with different naming patterns
        if submodule_name == "pantheon" and hasattr(module, "chi2_sn_pantheon_abs"):
            original_func = getattr(module, "chi2_sn_pantheon_abs")

            def wrapper(model_instance, _original=original_func):
                from cosmos.lcdm.model import LCDM
                from cosmos.pbuf.model import PBUF

                if isinstance(model_instance, LCDM):
                    model_type = "lcdm"
                elif isinstance(model_instance, PBUF):
                    model_type = "pbuf"
                else:
                    model_type = "unknown"

                params = _extract_params_from_model(model_instance, model_type)
                model_func = _create_model_func_from_model(model_instance)
                result = _original(model_func, params)
                return result["chi2"]

            return module, wrapper

        for attr in dir(module):
            # Try the new naming pattern first (chi2_sn_*)
            if attr == f"chi2_sn_{submodule_name}":
                original_func = getattr(module, attr)

                # Create a wrapper that adapts from old interface (model) to new interface (model_func, params)
                def wrapper(model_instance):
                    # Determine model type from the model instance
                    from cosmos.lcdm.model import LCDM
                    from cosmos.pbuf.model import PBUF

                    if isinstance(model_instance, LCDM):
                        model_type = "lcdm"
                    elif isinstance(model_instance, PBUF):
                        model_type = "pbuf"
                    else:
                        model_type = "unknown"

                    # Extract params from model
                    params = _extract_params_from_model(model_instance, model_type)

                    # Create model_func
                    model_func = _create_model_func_from_model(model_instance)

                    # Call the new chi2 function
                    result = original_func(model_func, params)

                    # Return just the chi2 value for compatibility
                    return result["chi2"]

                return module, wrapper

            # Try the old naming pattern (chi_squared_*)
            elif attr.startswith("chi_squared_"):
                return module, getattr(module, attr)

            # Try alternative pattern for pantheon/sh0es
            elif submodule_name in ["pantheon", "sh0es"] and attr == f"chi2_sn_{submodule_name}":
                original_func = getattr(module, attr)

                # Create a wrapper that adapts from old interface (model) to new interface (model_func, params)
                def wrapper(model_instance):
                    # Determine model type from the model instance
                    from cosmos.lcdm.model import LCDM
                    from cosmos.pbuf.model import PBUF

                    if isinstance(model_instance, LCDM):
                        model_type = "lcdm"
                    elif isinstance(model_instance, PBUF):
                        model_type = "pbuf"
                    else:
                        model_type = "unknown"

                    # Extract params from model
                    params = _extract_params_from_model(model_instance, model_type)

                    # Create model_func
                    model_func = _create_model_func_from_model(model_instance)

                    # Call the new chi2 function
                    result = original_func(model_func, params)

                    # Return just the chi2 value for compatibility
                    return result["chi2"]

                return module, wrapper

    except Exception:
        return None
    return None

def available_datasets(verbose=False):
    """
    Discover all available dataset χ² functions.

    Returns
    -------
    dict
        { "cmb": func, "sn": func, ... }
    """
    found = {}
    for name in DEFAULT_DATASETS:
        mod = _import_dataset_module(name)
        if mod:
            module, func = mod
            # For dotted names like sn.pantheon, use the submodule name
            if "." in name:
                shortname = name.split(".")[-1]
            else:
                shortname = name.replace(".", "_")
            found[shortname] = func
            if verbose:
                print(f"✓ Registered dataset: {shortname}")
        elif verbose:
            print(f"✗ Missing dataset: {name}")
    return found
