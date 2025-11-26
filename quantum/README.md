# Quantum configuration

 Runtime knobs that exclusively affect the Quantum Engine should live in this
 directory (e.g., regulator tables, thermal scan presets, export policies).
 Keeping them under `configs/quantum/` makes the provenance for quantum-derived
 artifacts explicit and avoids cross-contamination with Cosmos defaults.
