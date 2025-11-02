🧠 PBUF Data Interface Standard (v1)

Planck-Bound Unified Framework (PBUF)
Developer Guide – Data Handling and Fit Integration
Version 1.0 — October 2025

📘 Overview

All cosmological fits in PBUF (CMB, SN, BAO, CC, RSD, etc.) use a single, unified dataset schema.

Every χ² evaluator, data loader, and joint likelihood must conform to this standard so that models like ΛCDM and PBUF remain directly comparable and modular.

🔖 PBUF Data Object Schema

Each dataset must be a Python dictionary with the following fields:

{
    "name": str,             # Dataset name, e.g. "Planck2018", "Pantheon+", "DR16"
    "type": str,             # One of: "CMB", "SN", "BAO_ISO", "BAO_ANISO", "CC", "RSD"
    "z": np.ndarray | None,  # Redshifts (None for CMB distance priors)
    "obs": np.ndarray,       # Observed values (μ, R, D_V/rd, etc.)
    "err": np.ndarray | None,# 1σ uncertainties (optional if covariance provided)
    "cov": np.ndarray | None,# Covariance matrix (optional)
    "meta": dict             # Metadata (units, reference, source, version)
}

🧩 Schema Enforcement

All datasets must be validated through the helper:

from data_interface.standardize import ensure_standard_dataset


Example:

data = ensure_standard_dataset(data, "SN")


This guarantees:

Correct key names (z, obs, err, cov, meta)

Correct array shapes and numeric types

Dataset type consistency (e.g., “SN” for supernovae)

🧮 χ² Evaluator Pattern

All χ² functions inside cosmos/fits/<dataset>/ must follow this structure:

def chi_squared_<dataset>(model, data=None) -> float:
    data = ensure_standard_dataset(data, "<TYPE>")
    z, obs, err, cov = data["z"], data["obs"], data["err"], data["cov"]

    pred = compute_model_prediction(model, z)
    diff = pred - obs

    if cov is not None:
        chi2 = float(diff.T @ np.linalg.inv(cov) @ diff)
    else:
        chi2 = float(np.sum((diff / err) ** 2))
    return chi2

✅ Requirements
Rule	Description
Always call ensure_standard_dataset()	Enforces schema and type safety
Always return float	χ² must be a scalar
Never hard-code data	Use external loaders if data=None
Covariance > errors	Prefer covariance matrix if available
Consistent naming	Always use obs, err, cov, z, meta
🧱 Example Implementations
🌀 CMB
def chi_squared_cmb(model, data=None):
    data = ensure_standard_dataset(data, "CMB")
    obs = data["obs"]
    cov_inv = np.linalg.inv(data["cov"])

    pred = np.array([
        model.cmb_shift_parameter(),
        model.cmb_acoustic_scale(),
        model.cmb_angular_scale()
    ])

    diff = pred - obs
    return float(diff.T @ cov_inv @ diff)

💥 Supernovae (SN)
def chi_squared_sn(model, data=None):
    data = ensure_standard_dataset(data, "SN")
    z, obs, err, cov = data["z"], data["obs"], data["err"], data["cov"]
    mu = model.distance_modulus(z)
    diff = mu - obs
    return float(diff.T @ np.linalg.inv(cov) @ diff) if cov is not None else float(np.sum((diff / err)**2))

🌌 BAO Isotropic
def chi_squared_bao_iso(model, data=None):
    data = ensure_standard_dataset(data, "BAO_ISO")
    z, obs, err, cov = data["z"], data["obs"], data["err"], data["cov"]
    pred = np.array([model.DV_over_rd(zi) for zi in z])
    diff = pred - obs
    return float(diff.T @ np.linalg.inv(cov) @ diff) if cov is not None else float(np.sum((diff / err)**2))

🔭 BAO Anisotropic
def chi_squared_bao_aniso(model, data=None):
    data = ensure_standard_dataset(data, "BAO_ANISO")
    z, obs, err, cov = data["z"], data["obs"], data["err"], data["cov"]

    pred = []
    for zi in z:
        pred += [model.DM_over_rd(zi), model.DH_over_rd(zi)]
    diff = np.array(pred) - obs
    return float(diff.T @ np.linalg.inv(cov) @ diff) if cov is not None else float(np.sum((diff / err)**2))

Ordering rules  
- `obs` is interleaved per redshift: `[D_M(z_0)/r_d, D_H(z_0)/r_d, D_M(z_1)/r_d, …]`.  
- `cov` must follow the same ordering. If your covariance is block-structured (`[D_M… , D_H…]`), reorder it before loading.  
- When transforming published correlations between `D_M/r_d` and `H(z) r_d / c`, flip the sign after converting to `D_H/r_d = 1 / (H(z) r_d / c)`.

⏳ Cosmic Chronometers (CC)
def chi_squared_cc(model, data=None):
    data = ensure_standard_dataset(data, "CC")
    z, obs, err, cov = data["z"], data["obs"], data["err"], data["cov"]
    H_model = np.array([model.H(z_i) for z_i in z])
    diff = H_model - obs
    return float(diff.T @ np.linalg.inv(cov) @ diff) if cov is not None else float(np.sum((diff / err)**2))

🌐 Redshift-Space Distortions (RSD)
def chi_squared_rsd(model, data=None):
    data = ensure_standard_dataset(data, "RSD")
    z, obs, err, cov = data["z"], data["obs"], data["err"], data["cov"]
    fs8_model = np.array([model.fsigma8(z_i) for z_i in z])
    diff = fs8_model - obs
    return float(diff.T @ np.linalg.inv(cov) @ diff) if cov is not None else float(np.sum((diff / err)**2))

🧩 Joint χ² Integration

All χ² functions can be combined dynamically for multi-probe fits:

def compute_joint_chi2(model, datasets):
    funcs = {
        "cmb": chi_squared_cmb,
        "sn": chi_squared_sn,
        "bao_iso": chi_squared_bao_iso,
        "bao_aniso": chi_squared_bao_aniso,
        "cc": chi_squared_cc,
        "rsd": chi_squared_rsd,
    }

    total = 0.0
    results = {}
    for key, func in funcs.items():
        chi2 = func(model, datasets.get(key))
        results[key] = chi2
        total += chi2

    results["total"] = total
    return results

🧠 Developer Rules
Rule	Description
1	Always validate datasets via ensure_standard_dataset()
2	Use consistent field names (obs, err, cov, z, meta)
3	Do not embed raw data inside modules
4	Covariance takes precedence over simple errors
5	Always return scalar float for χ²
6	Use the same schema for synthetic and real data
7	New datasets must include loader + chi² module + test
⚙️ Adding New Datasets

To add a new observational probe:

Create a loader in data_interface/<new>_loader.py that outputs a schema-compliant dict.

Create a χ² evaluator in cosmos/fits/<new>/chi2.py using the same pattern.

Register the new fit in the joint χ² dictionary.

Add a test in /tests/test_<new>.py.

That’s it — it will automatically integrate with all models and joint fits.

🧭 Example: Loader Output (CMB)
{
    "name": "Planck2018",
    "type": "CMB",
    "z": None,
    "obs": np.array([1.7498, 301.729, 0.0104085]),
    "err": None,
    "cov": np.array([
        [1.296e-05, 2.475e-05, -6.9e-07],
        [2.475e-05, 1.2193e-04, -1.98e-06],
        [-6.9e-07, -1.98e-06, 2e-08]
    ]),
    "meta": {"units": "dimensionless", "source": "Planck 2018 distance priors"}
}

✅ Summary

One data schema governs all probes.

Every χ² function validates its input.

Real and synthetic datasets share identical structures.

The joint fit pipeline auto-integrates all components.

This ensures every comparison between ΛCDM and PBUF is physically meaningful, computationally stable, and reproducible.
