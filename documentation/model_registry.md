# Model Registry

The Cosmology Model Registry lives inside `cosmos/models/__init__.py` and the accompanying factory helpers. It maps the user-facing names passed to the CLI (e.g. `--model lcdm`) to the code bundles that expose the shared expansion, distance, and sanity helpers every fit or optimiser can rely on.

Current entries:

- **lcdm** – the production ΛCDM engine with the full `LCDMModel`, CMB wrappers, basin optimiser, and Phase-6a checks.
- **pbuf** – the elastic spacetime model that adds thermal tables, microphysics wiring, and PBUF-specific phase-7a guards.
- **ede_lcdm** – the new Early Dark Energy variant that exposes `H(a)`, `E(a)`, Ωₑₑₐ(a) hilltop phenomenology, distance helpers, and Phase-6a checks while keeping LCDM extremal behaviour in the limit Ω_EDE → 0.
- **desi_mod** – the DESI-modified ΛCDM helpers that are currently present for pytest coverage and not plugged into the main fits yet.
- **running_lambda** – the phenomenological running-vacuum brother of ΛCDM that keeps the standard distance, growth, and CMB interfaces but evolves Ω_Λ(H) via ν_Λ, admits GR growth, and ships with its own sanity gate.
- **dgp** – the Dvali–Gabadadze–Porrati braneworld reference cosmology (self-accelerating, parameterised by Ω_rc); shares the distance/growth helpers with the other models and is currently exposed for comparison tests only.

Each model bundle registers its sanity gate (Phase-6a / Phase-7a) inside `cosmos.models.__init__.py`, so new candidates such as `ede_lcdm` can be resolved by name once their helper suite is available.

## cosmos2 coverage

The `cosmos2` refactor intentionally keeps a narrow model surface (`lcdm`, `pbuf`). The comparison/experimental variants that live under `cosmos/models`—`ede_lcdm`, `running_lambda`, `dgp`, `mg_lcdm`, `desi_mod`—stay **legacy-only** for now. Attempts to request them through the cosmos2 model factory raise an explicit error; use the original `cosmos` package if you need those models. Porting them would require full kernel translations and is out of scope for the current v11 science run.
